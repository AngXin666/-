"""
整合页面检测器 - 页面分类器 + YOLO模型
Integrated Page Detector - Page Classifier + YOLO Models

工作流程：
1. 使用页面分类器（PyTorch）快速识别页面类型（100%准确率，20-50ms）
2. 根据页面类型自动加载对应的YOLO模型
3. 使用YOLO模型检测页面元素（按钮、输入框等）
"""

import asyncio
import json
import os
from typing import Optional, Dict, List, Tuple
from pathlib import Path
from dataclasses import dataclass
from io import BytesIO
import time

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms, models
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

from .adb_bridge import ADBBridge
from .page_detector import PageState, PageDetectionResult


@dataclass
class PageElement:
    """页面元素检测结果"""
    class_name: str  # 元素类别名称
    confidence: float  # 置信度
    bbox: Tuple[int, int, int, int]  # 边界框 (x1, y1, x2, y2)
    center: Tuple[int, int]  # 中心点 (x, y)


@dataclass
class IntegratedDetectionResult(PageDetectionResult):
    """整合检测结果"""
    elements: List[PageElement] = None  # 检测到的页面元素
    yolo_model_used: str = None  # 使用的YOLO模型
    
    def __post_init__(self):
        if self.elements is None:
            self.elements = []


class PageDetectorIntegrated:
    """整合页面检测器 - 页面分类器 + YOLO模型"""
    
    def __init__(self, adb: ADBBridge, 
                 classifier_model_path='page_classifier_pytorch_best.pth',
                 classes_path='page_classes.json',
                 yolo_registry_path='yolo_model_registry.json',
                 mapping_path='page_yolo_mapping.json',
                 log_callback=None):
        """初始化整合检测器
        
        Args:
            adb: ADB 桥接器实例
            classifier_model_path: 页面分类器模型路径
            classes_path: 类别列表文件路径
            yolo_registry_path: YOLO模型注册表路径
            mapping_path: 页面-YOLO映射配置路径
            log_callback: 日志回调函数
        """
        self.adb = adb
        self._log_callback = log_callback
        
        # 页面分类器相关
        self._classifier_model = None
        self._classes = None
        self._device = None
        self._transform = None
        self._img_size = (224, 224)
        
        # YOLO模型相关
        self._yolo_models = {}  # 缓存已加载的YOLO模型
        self._yolo_registry = {}
        self._page_yolo_mapping = {}
        
        # 类别名称到PageState的映射
        self._class_to_state = {
            '个人页_已登录': PageState.PROFILE_LOGGED,
            '个人页_未登录': PageState.PROFILE,
            '个人页广告': PageState.PROFILE_AD,
            '交易流水': PageState.TRANSACTION_HISTORY,
            '优惠劵': PageState.COUPON,
            '分类页': PageState.CATEGORY,
            '加载页': PageState.LOADING,
            '启动页服务弹窗': PageState.STARTUP_POPUP,
            '广告页': PageState.AD,
            '搜索页': PageState.SEARCH,
            '文章页': PageState.ARTICLE,
            '模拟器桌面': PageState.LAUNCHER,
            '温馨提示': PageState.WARMTIP,
            '登录页': PageState.LOGIN,
            '积分页': PageState.POINTS_PAGE,
            '签到弹窗': PageState.CHECKIN_POPUP,
            '签到页': PageState.CHECKIN,
            '设置页': PageState.SETTINGS,
            '转账页': PageState.TRANSFER,
            '钱包页': PageState.WALLET,
            '首页': PageState.HOME,
            '首页公告': PageState.HOME_NOTICE,
            '首页异常代码弹窗': PageState.HOME_ERROR_POPUP,
        }
        
        # 初始化检测缓存
        from .performance.detection_cache import DetectionCache
        self._detection_cache = DetectionCache(ttl=0.5)  # 缓存0.5秒，足够快速检测页面变化
        
        # 加载配置和模型
        self._load_classifier(classifier_model_path, classes_path)
        self._load_yolo_registry(yolo_registry_path)
        self._load_mapping(mapping_path)
    
    def _log(self, msg: str):
        """输出日志"""
        if self._log_callback:
            self._log_callback(msg)
    
    def _load_classifier(self, model_path: str, classes_path: str):
        """加载页面分类器"""
        if not HAS_TORCH or not HAS_PIL:
            self._log("[整合检测器] ✗ PyTorch或PIL未安装")
            return
        
        try:
            # 加载类别列表（尝试在models目录查找）
            if not os.path.exists(classes_path):
                alt_classes_path = os.path.join('models', classes_path)
                if os.path.exists(alt_classes_path):
                    classes_path = alt_classes_path
                else:
                    self._log(f"[整合检测器] ✗ 类别文件不存在: {classes_path}")
                    return
            
            with open(classes_path, 'r', encoding='utf-8') as f:
                self._classes = json.load(f)
            
            # 加载模型（尝试在models目录查找）
            if not os.path.exists(model_path):
                alt_model_path = os.path.join('models', model_path)
                if os.path.exists(alt_model_path):
                    model_path = alt_model_path
                else:
                    self._log(f"[整合检测器] ✗ 模型文件不存在: {model_path}")
                    return
            
            # 设置设备
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 定义模型架构
            class PageClassifier(nn.Module):
                def __init__(self, num_classes):
                    super(PageClassifier, self).__init__()
                    self.mobilenet = models.mobilenet_v2(weights=None)
                    in_features = self.mobilenet.classifier[1].in_features
                    self.mobilenet.classifier = nn.Sequential(
                        nn.Dropout(0.2),
                        nn.Linear(in_features, 128),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(128, num_classes)
                    )
                
                def forward(self, x):
                    return self.mobilenet(x)
            
            # 创建并加载模型
            num_classes = len(self._classes)
            model = PageClassifier(num_classes)
            
            checkpoint = torch.load(model_path, map_location=self._device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model = model.to(self._device)
            model.eval()
            self._classifier_model = model
            
            # 设置图片预处理
            self._transform = transforms.Compose([
                transforms.Resize(self._img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            self._log(f"[整合检测器] ✓ 页面分类器已加载 (设备: {self._device})")
            
        except Exception as e:
            self._log(f"[整合检测器] ✗ 加载页面分类器失败: {e}")
            self._classifier_model = None
    
    def _load_yolo_registry(self, registry_path: str):
        """加载YOLO模型注册表"""
        try:
            # 如果路径不是绝对路径，尝试在models目录中查找
            if not os.path.isabs(registry_path) and not os.path.exists(registry_path):
                models_registry_path = os.path.join('models', registry_path)
                if os.path.exists(models_registry_path):
                    registry_path = models_registry_path
            
            if not os.path.exists(registry_path):
                self._log(f"[整合检测器] ✗ YOLO注册表不存在: {registry_path}")
                return
            
            with open(registry_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self._yolo_registry = data.get('models', {})
            
            self._log(f"[整合检测器] ✓ YOLO注册表已加载 ({len(self._yolo_registry)} 个模型)")
            
        except Exception as e:
            self._log(f"[整合检测器] ✗ 加载YOLO注册表失败: {e}")
    
    def _load_mapping(self, mapping_path: str):
        """加载页面-YOLO映射配置"""
        try:
            # 如果路径不是绝对路径，尝试在models目录中查找
            if not os.path.isabs(mapping_path) and not os.path.exists(mapping_path):
                models_mapping_path = os.path.join('models', mapping_path)
                if os.path.exists(models_mapping_path):
                    mapping_path = models_mapping_path
            
            if not os.path.exists(mapping_path):
                self._log(f"[整合检测器] ✗ 映射配置不存在: {mapping_path}")
                return
            
            with open(mapping_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self._page_yolo_mapping = data.get('mapping', {})
            
            self._log(f"[整合检测器] ✓ 页面-YOLO映射已加载 ({len(self._page_yolo_mapping)} 个页面)")
            
        except Exception as e:
            self._log(f"[整合检测器] ✗ 加载映射配置失败: {e}")
    
    def _load_yolo_model(self, model_key: str) -> Optional[YOLO]:
        """加载YOLO模型（带缓存）"""
        if not HAS_YOLO:
            return None
        
        # 检查缓存
        if model_key in self._yolo_models:
            return self._yolo_models[model_key]
        
        # 从注册表获取模型路径
        model_info = self._yolo_registry.get(model_key)
        if not model_info:
            self._log(f"[整合检测器] ✗ YOLO模型未注册: {model_key}")
            return None
        
        model_path = model_info.get('model_path')
        if not model_path:
            self._log(f"[整合检测器] ✗ YOLO模型路径为空: {model_key}")
            return None
        
        # 如果路径不是绝对路径，添加models/前缀
        if not os.path.isabs(model_path):
            # 尝试在models目录中查找
            models_path = os.path.join('models', model_path)
            if os.path.exists(models_path):
                model_path = models_path
            # 如果models/路径不存在，尝试原路径（兼容旧版本）
            elif not os.path.exists(model_path):
                self._log(f"[整合检测器] ✗ YOLO模型文件不存在: {model_path} (也尝试了 {models_path})")
                return None
        
        if not os.path.exists(model_path):
            self._log(f"[整合检测器] ✗ YOLO模型文件不存在: {model_path}")
            return None
        
        try:
            model = YOLO(model_path)
            self._yolo_models[model_key] = model
            self._log(f"[整合检测器] ✓ YOLO模型已加载: {model_key} ({model_path})")
            return model
        except Exception as e:
            self._log(f"[整合检测器] ✗ 加载YOLO模型失败 {model_key}: {e}")
            return None
    
    async def _get_screenshot(self, device_id: str) -> Optional[Image.Image]:
        """获取屏幕截图"""
        if not HAS_PIL:
            return None
        
        try:
            screenshot_data = await self.adb.screencap(device_id)
            if not screenshot_data:
                return None
            
            image = Image.open(BytesIO(screenshot_data))
            return image
        except Exception:
            return None
    
    def _classify_page(self, image: Image.Image) -> Tuple[Optional[str], float]:
        """使用页面分类器识别页面类型"""
        if not self._classifier_model or not self._classes:
            return None, 0.0
        
        try:
            # 转换为RGB
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            
            # 预处理和预测
            image_tensor = self._transform(image).unsqueeze(0).to(self._device)
            
            with torch.no_grad():
                outputs = self._classifier_model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                class_name = self._classes[predicted_idx.item()]
                confidence_value = confidence.item()
            
            return class_name, confidence_value
            
        except Exception as e:
            self._log(f"[整合检测器] ✗ 页面分类失败: {e}")
            return None, 0.0
    
    def _detect_elements(self, image: Image.Image, page_class: str) -> List[PageElement]:
        """使用YOLO模型检测页面元素"""
        if not HAS_YOLO:
            print(f"  [_detect_elements] ✗ YOLO库未安装")
            return []
        
        # 调试信息：输出所有可用的映射键
        print(f"  [_detect_elements] 页面类型: {page_class}")
        print(f"  [_detect_elements] 映射配置中的所有页面类型: {list(self._page_yolo_mapping.keys())[:10]}...")  # 只显示前10个
        
        # 获取该页面类型对应的YOLO模型
        mapping = self._page_yolo_mapping.get(page_class, {})
        yolo_models = mapping.get('yolo_models', [])
        
        print(f"  [_detect_elements] 映射的YOLO模型数量: {len(yolo_models)}")
        
        if not yolo_models:
            print(f"  [_detect_elements] ⚠️ 该页面类型没有配置YOLO模型")
            print(f"  [_detect_elements] 调试: mapping = {mapping}")
            print(f"  [_detect_elements] 调试: page_class在映射中? {page_class in self._page_yolo_mapping}")
            return []
        
        elements = []
        
        # 按优先级加载和使用YOLO模型
        for model_info in sorted(yolo_models, key=lambda x: x.get('priority', 999)):
            model_key = model_info.get('model_key')
            if not model_key:
                continue
            
            print(f"  [_detect_elements] 尝试加载模型: {model_key}")
            model = self._load_yolo_model(model_key)
            if not model:
                print(f"  [_detect_elements] ✗ 模型加载失败: {model_key}")
                continue
            
            print(f"  [_detect_elements] ✓ 模型加载成功: {model_key}")
            
            try:
                # 使用YOLO检测
                results = model.predict(image, conf=0.25, verbose=False)
                
                for result in results:
                    boxes = result.boxes
                    print(f"  [_detect_elements] 检测到 {len(boxes)} 个目标")
                    
                    for box in boxes:
                        # 提取检测信息
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        class_name = result.names[cls]
                        
                        # 计算中心点
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        
                        element = PageElement(
                            class_name=class_name,
                            confidence=conf,
                            bbox=(int(x1), int(y1), int(x2), int(y2)),
                            center=(center_x, center_y)
                        )
                        elements.append(element)
                        print(f"  [_detect_elements]   - {class_name}: 置信度={conf:.2f}")
                
            except Exception as e:
                self._log(f"[整合检测器] ✗ YOLO检测失败 {model_key}: {e}")
                print(f"  [_detect_elements] ✗ YOLO检测异常: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"  [_detect_elements] 总共检测到 {len(elements)} 个元素")
        return elements
    
    async def detect_page(self, device_id: str, use_cache: bool = True, 
                         detect_elements: bool = True,
                         use_ocr: bool = False,  # 兼容参数，整合检测器不使用OCR
                         use_template: bool = True,  # 兼容参数
                         use_dl: bool = True) -> IntegratedDetectionResult:  # 兼容参数
        """检测当前页面状态和元素
        
        Args:
            device_id: 设备 ID
            use_cache: 是否使用缓存
            detect_elements: 是否检测页面元素（使用YOLO）
            use_ocr: 兼容参数（整合检测器不使用OCR，忽略此参数）
            use_template: 兼容参数（整合检测器不使用模板匹配，忽略此参数）
            use_dl: 兼容参数（整合检测器始终使用深度学习，忽略此参数）
            
        Returns:
            整合检测结果
        """
        start_time = time.time()
        
        # 检查缓存
        if use_cache:
            cached_result = self._detection_cache.get(device_id)
            if cached_result is not None:
                cached_result.cached = True
                cached_result.detection_time = time.time() - start_time
                return cached_result
        
        # 检查分类器是否加载
        if not self._classifier_model or not self._classes:
            return IntegratedDetectionResult(
                state=PageState.UNKNOWN,
                confidence=0.0,
                details="页面分类器未加载",
                detection_method="integrated",
                detection_time=time.time() - start_time
            )
        
        # 获取截图
        screenshot_start = time.time()
        image = await self._get_screenshot(device_id)
        screenshot_time = time.time() - screenshot_start
        
        if not image:
            return IntegratedDetectionResult(
                state=PageState.UNKNOWN,
                confidence=0.0,
                details="无法截取屏幕",
                detection_method="integrated",
                detection_time=time.time() - start_time
            )
        
        # 1. 使用页面分类器识别页面类型
        classify_start = time.time()
        page_class, confidence = self._classify_page(image)
        classify_time = time.time() - classify_start
        
        # 性能日志（仅在检测时间超过0.5秒时输出）
        total_time = time.time() - start_time
        if total_time > 0.5:
            print(f"  [性能警告] detect_page耗时{total_time:.3f}秒 (截图:{screenshot_time:.3f}秒, 分类:{classify_time:.3f}秒)")
        
        if not page_class:
            return IntegratedDetectionResult(
                state=PageState.UNKNOWN,
                confidence=0.0,
                details="页面分类失败",
                detection_method="integrated",
                detection_time=time.time() - start_time
            )
        
        # 映射到PageState
        state = self._class_to_state.get(page_class, PageState.UNKNOWN)
        
        # 2. 使用YOLO检测页面元素（可选）
        elements = []
        yolo_model_used = None
        if detect_elements:
            elements = self._detect_elements(image, page_class)
            if elements:
                # 记录使用的YOLO模型
                mapping = self._page_yolo_mapping.get(page_class, {})
                yolo_models = mapping.get('yolo_models', [])
                if yolo_models:
                    yolo_model_used = yolo_models[0].get('model_key')
        
        # 构建结果
        details = f"页面分类: {page_class} (置信度: {confidence:.2%})"
        if elements:
            details += f", 检测到 {len(elements)} 个元素"
        
        result = IntegratedDetectionResult(
            state=state,
            confidence=confidence,
            details=details,
            detection_method="integrated",
            detection_time=time.time() - start_time,
            cached=False,
            elements=elements,
            yolo_model_used=yolo_model_used
        )
        
        # 更新缓存
        if use_cache:
            self._detection_cache.set(device_id, result)
        
        return result
    
    async def get_element(self, device_id: str, element_name: str) -> Optional[PageElement]:
        """获取指定名称的页面元素
        
        Args:
            device_id: 设备 ID
            element_name: 元素名称（如"每日签到按钮"）
            
        Returns:
            页面元素或None
        """
        result = await self.detect_page(device_id, detect_elements=True)
        
        for element in result.elements:
            if element.class_name == element_name:
                return element
        
        return None
    
    async def click_element(self, device_id: str, element_name: str) -> bool:
        """点击指定名称的页面元素
        
        Args:
            device_id: 设备 ID
            element_name: 元素名称
            
        Returns:
            是否成功点击
        """
        element = await self.get_element(device_id, element_name)
        if not element:
            self._log(f"[整合检测器] ✗ 未找到元素: {element_name}")
            return False
        
        # 点击元素中心点
        x, y = element.center
        await self.adb.tap(device_id, x, y)
        self._log(f"[整合检测器] ✓ 点击元素: {element_name} at ({x}, {y})")
        return True
    
    async def detect_page_with_priority(self, device_id: str, expected_pages: List[str], use_cache: bool = True) -> IntegratedDetectionResult:
        """使用优先级检测页面（兼容混合检测器的接口）
        
        整合检测器不使用模板匹配，所以忽略expected_pages参数，直接调用detect_page
        
        Args:
            device_id: 设备 ID
            expected_pages: 期望的页面模板列表（忽略）
            use_cache: 是否使用缓存
            
        Returns:
            整合检测结果
        """
        return await self.detect_page(device_id, use_cache=use_cache, detect_elements=False)
    
    def clear_cache(self, device_id: str = None):
        """清除缓存（兼容混合检测器的接口）
        
        Args:
            device_id: 设备ID，如果为None则清除所有缓存
        """
        if hasattr(self, '_detection_cache'):
            self._detection_cache.clear(device_id)
    
    async def find_button_yolo(self, device_id: str, page_type: str, button_name: str,
                              conf_threshold: float = 0.5) -> Optional[Tuple[int, int]]:
        """使用YOLO查找指定按钮的坐标
        
        Args:
            device_id: 设备ID
            page_type: 页面类型（如 'checkin' 表示签到页，'homepage' 表示首页）
            button_name: 按钮名称（如 '签到按钮'、'每日签到按钮'）
            conf_threshold: 置信度阈值
            
        Returns:
            按钮中心点坐标 (x, y)，如果未找到返回None
        """
        if not HAS_YOLO:
            self._log("[整合检测器] ✗ YOLO未安装")
            return None
        
        try:
            # 获取截图
            image = await self._get_screenshot(device_id)
            if not image:
                self._log("[整合检测器] ✗ 无法获取截图")
                return None
            
            # 直接使用 page_type 作为 model_key（注册表中的键）
            self._log(f"[整合检测器] 尝试加载模型: {page_type}")
            model = self._load_yolo_model(page_type)
            
            if not model:
                self._log(f"[整合检测器] ✗ 无法加载模型: {page_type}")
                return None
            
            self._log(f"[整合检测器] ✓ 模型已加载，开始检测...")
            
            # 使用YOLO检测
            results = model.predict(image, conf=conf_threshold, verbose=False)
            
            # 查找指定按钮
            for result in results:
                boxes = result.boxes
                self._log(f"[整合检测器] 检测到 {len(boxes)} 个对象")
                
                for box in boxes:
                    cls = int(box.cls[0])
                    class_name = result.names[cls]
                    conf = float(box.conf[0])
                    
                    self._log(f"[整合检测器] 检测到: {class_name} (置信度: {conf:.2%})")
                    
                    # 检查是否是目标按钮
                    if button_name in class_name or class_name in button_name:
                        # 提取边界框
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # 计算中心点
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        
                        self._log(f"[整合检测器] ✓ YOLO检测到按钮: {class_name} at ({center_x}, {center_y}), 置信度: {conf:.2%}")
                        
                        return (center_x, center_y)
            
            # 未找到按钮
            self._log(f"[整合检测器] ✗ 未找到按钮: {button_name}")
            return None
            
        except Exception as e:
            self._log(f"[整合检测器] ✗ YOLO按钮检测失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def close_popup(self, device_id: str, timeout: float = 15.0) -> bool:
        """自动关闭弹窗（带超时保护和重试机制）
        
        从混合检测器复制的完整实现，适配整合检测器
        
        Args:
            device_id: 设备ID
            timeout: 总超时时间（秒），默认15秒
        
        Returns:
            是否成功关闭
        """
        import asyncio
        
        try:
            # 使用 asyncio.wait_for 为整个关闭流程添加超时
            return await asyncio.wait_for(
                self._close_popup_impl(device_id),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            self._log(f"[整合检测器] ✗ 关闭弹窗超时（{timeout}秒）")
            return False
    
    async def _close_popup_impl(self, device_id: str) -> bool:
        """关闭弹窗的实际实现（从混合检测器复制）"""
        from .retry_helper import retry_until_success
        from .ocr_thread_pool import get_ocr_pool
        
        # 弹窗按钮坐标 (540x960)
        POPUP_BUTTONS = {
            'user_agreement': (270, 600),      # 服务协议弹窗"同意并接受"
            'user_agreement_alt': (270, 608),  # 服务协议弹窗备用坐标
            'home_announcement': (270, 690),   # 主页公告弹窗（底部中央按钮）
            'login_error': (436, 557),         # 登录错误确定按钮
            'generic': (270, 600),             # 通用弹窗
        }
        
        # 签到弹窗关闭按钮坐标（MuMu模拟器 540x960）
        CHECKIN_POPUP_CLOSE = [
            (270, 812),  # 中心位置
            (278, 811),  # 右偏
            (274, 811),  # 中右
        ]
        
        # 优先使用页面分类结果来判断弹窗类型
        popup_type = None
        button_pos = None
        
        # 检查当前页面状态
        result = await self.detect_page(device_id, use_cache=False)
        if result.state != PageState.POPUP and result.state != PageState.CHECKIN_POPUP:
            self._log(f"[整合检测器] 当前不是弹窗页面，无需关闭")
            return True
        
        # 获取当前截图用于OCR识别
        screenshot_data = await self.adb.screencap(device_id)
        current_screenshot = None
        if screenshot_data and HAS_PIL:
            current_screenshot = Image.open(BytesIO(screenshot_data))
        
        # 使用OCR检测弹窗类型
        ocr_pool = get_ocr_pool()
        if ocr_pool and current_screenshot:
            try:
                texts = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: ocr_pool.ocr_image(current_screenshot)
                )
                
                if texts:
                    text_str = " ".join(texts) if texts else ""
                    self._log(f"[整合检测器] OCR识别到: {texts[:5] if texts else '无'}...")
                    
                    # 登录错误弹窗（最高优先级）
                    if "友情提示" in text_str:
                        popup_type = "login_error"
                        button_pos = POPUP_BUTTONS['login_error']
                        self._log(f"[整合检测器] 类型: {popup_type} (OCR检测)")
                    # 用户协议弹窗
                    elif any(kw in text_str for kw in ["用户协议", "隐私政策", "服务协议", "隐私协议"]):
                        if "登录" not in text_str or "同意并接受" in text_str:
                            popup_type = "user_agreement"
                            button_pos = POPUP_BUTTONS['user_agreement']
                            self._log(f"[整合检测器] 类型: {popup_type} (OCR检测)")
                    # 主页公告弹窗
                    elif any(kw in text_str for kw in ["公告", "活动", "恭喜", "领取", "×"]):
                        popup_type = "home_announcement"
                        button_pos = POPUP_BUTTONS['home_announcement']
                        self._log(f"[整合检测器] 类型: {popup_type} (OCR检测)")
                    # 通用弹窗
                    elif any(kw in text_str for kw in ["确定", "关闭", "取消", "知道了", "我知道了"]):
                        popup_type = "generic"
                        button_pos = POPUP_BUTTONS['generic']
                        self._log(f"[整合检测器] 类型: {popup_type} (OCR检测)")
                    else:
                        popup_type = "unknown"
                        button_pos = POPUP_BUTTONS['generic']
                        self._log(f"[整合检测器] 类型: {popup_type} (OCR检测)")
            except Exception as e:
                self._log(f"[整合检测器] OCR检测失败: {e}")
                popup_type = "generic"
                button_pos = POPUP_BUTTONS['generic']
        else:
            popup_type = "generic"
            button_pos = POPUP_BUTTONS['generic']
        
        # 如果是首页公告弹窗，点击弹窗外部关闭
        if popup_type == "home_announcement":
            self._log(f"[整合检测器] 首页公告弹窗，点击外部区域关闭...")
            await self.adb.tap(device_id, 270, 100)
            await asyncio.sleep(2)
            
            result = await self.detect_page(device_id)
            if result.state != PageState.POPUP:
                self._log(f"[整合检测器] ✓ 成功关闭首页公告弹窗")
                return True
            else:
                self._log(f"[整合检测器] ⚠️ 点击外部未关闭，尝试其他位置...")
                await self.adb.tap(device_id, 270, 850)
                await asyncio.sleep(2)
                
                result = await self.detect_page(device_id)
                if result.state != PageState.POPUP:
                    self._log(f"[整合检测器] ✓ 成功关闭首页公告弹窗")
                    return True
        
        # 检查是否是签到奖励弹窗
        is_checkin_popup = (popup_type == "checkin_popup" or result.state == PageState.CHECKIN_POPUP)
        if not is_checkin_popup and ocr_pool and current_screenshot:
            try:
                texts = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: ocr_pool.ocr_image(current_screenshot)
                )
                if texts:
                    text_str = ''.join(texts)
                    if ("恭喜" in text_str and "成功" in text_str) or "知道了" in text_str:
                        is_checkin_popup = True
                        self._log(f"[整合检测器] 检测到签到奖励弹窗 (OCR确认)")
            except:
                pass
        
        # 如果是签到弹窗，使用专用坐标
        if is_checkin_popup:
            self._log(f"[整合检测器] 使用签到弹窗专用坐标...")
            for i, (x, y) in enumerate(CHECKIN_POPUP_CLOSE, 1):
                self._log(f"[整合检测器] 尝试位置 {i}/3: ({x}, {y})")
                await self.adb.tap(device_id, x, y)
                await asyncio.sleep(2)
                
                result = await self.detect_page(device_id)
                if result.state != PageState.POPUP and result.state != PageState.CHECKIN_POPUP:
                    self._log(f"[整合检测器] ✓ 成功关闭签到弹窗")
                    return True
            
            self._log(f"[整合检测器] ⚠️ 签到弹窗专用坐标都失败，尝试其他方法...")
        
        # 使用预设位置
        if button_pos:
            await self.adb.tap(device_id, button_pos[0], button_pos[1])
            await asyncio.sleep(2)
            
            result = await self.detect_page(device_id)
            if result.state != PageState.POPUP:
                self._log(f"[整合检测器] ✓ 成功关闭")
                return True
            else:
                self._log(f"[整合检测器] ⚠️ 预设位置点击失败，仍是弹窗")
            
            # 尝试其他预设位置
            if popup_type in ["unknown", "home_announcement", "user_agreement"]:
                self._log(f"[整合检测器] 尝试其他预设位置...")
                alternative_positions = [
                    (270, 608),  # 备用位置1
                    (270, 620),  # 稍微靠下
                    (270, 650),  # 更靠下的位置
                    (270, 550),  # 更靠上
                ]
                
                for pos in alternative_positions:
                    await self.adb.tap(device_id, pos[0], pos[1])
                    await asyncio.sleep(1.5)
                    
                    result = await self.detect_page(device_id)
                    if result.state != PageState.POPUP:
                        self._log(f"[整合检测器] ✓ 成功关闭（位置: {pos}）")
                        return True
                
                # 尝试按返回键
                self._log(f"[整合检测器] 所有位置都失败，尝试按返回键...")
                await self.adb.press_back(device_id)
                await asyncio.sleep(1.5)
                
                result = await self.detect_page(device_id)
                if result.state != PageState.POPUP:
                    self._log(f"[整合检测器] ✓ 成功关闭（返回键）")
                    return True
                else:
                    self._log(f"[整合检测器] ✗ 返回键也失败，弹窗无法关闭")
                    return False
        
        self._log(f"[整合检测器] ✗ 无法关闭弹窗")
        return False
