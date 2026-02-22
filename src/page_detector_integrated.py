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
from .page_detector import PageDetectionResult
from .page_state_dynamic import PageState, PageStateType


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
                 state_mapping_path='page_state_mapping.json',
                 log_callback=None):
        """初始化智能检测器
        
        Args:
            adb: ADB 桥接器实例
            classifier_model_path: 页面分类器模型路径
            classes_path: 类别列表文件路径
            yolo_registry_path: YOLO模型注册表路径
            mapping_path: 页面-YOLO映射配置路径
            state_mapping_path: 页面状态映射配置路径
            log_callback: 日志回调函数
        """
        # 强制输出，确保能看到初始化过程
        import sys
        print("[智能检测器] __init__ 开始初始化...")
        sys.stdout.flush()
        
        # 【优化】移除PyTorch线程限制，让PyTorch自动管理线程
        # 检测器是单例，多个账号共享同一个实例，不需要限制线程数
        # PyTorch会根据CPU核心数自动分配线程，支持多账号并发调用
        
        self.adb = adb
        self._log_callback = log_callback
        self._verbose = True  # [2026-02-21] 临时启用详细日志，调试YOLO模型加载
        
        print("[智能检测器] 初始化成员变量...")
        sys.stdout.flush()
        
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
        
        # 类别名称到PageState的映射（从配置文件动态加载）
        self._class_to_state = {}
        self._state_mapping_config = {}
        
        # 初始化检测缓存
        from .performance.detection_cache import DetectionCache
        self._detection_cache = DetectionCache(ttl=0.5)  # 缓存0.5秒，足够快速检测页面变化
        
        print("[智能检测器] 开始加载配置和模型...")
        sys.stdout.flush()
        
        # 加载配置和模型
        self._load_state_mapping(state_mapping_path)  # 先加载状态映射
        print("[智能检测器] 状态映射加载完成，开始加载分类器...")
        sys.stdout.flush()
        
        self._load_classifier(classifier_model_path, classes_path)
        print("[智能检测器] 分类器加载完成，开始加载YOLO注册表...")
        sys.stdout.flush()
        
        self._load_yolo_registry(yolo_registry_path)
        self._load_mapping(mapping_path)
        
        print(f"[智能检测器] __init__ 完成 (model={self._classifier_model is not None}, classes={self._classes is not None})")
        sys.stdout.flush()
    
    def _log(self, msg: str, level: str = "debug"):
        """输出日志
        
        Args:
            msg: 日志消息
            level: 日志级别 ("info" 或 "debug")
                - "info": 关键信息，总是输出
                - "debug": 调试信息，只在verbose模式下输出
        """
        # 默认禁用所有智能检测器的详细日志
        # 如果需要调试，可以设置 self._verbose = True
        if not self._verbose:
            return
        
        if level == "info" or self._verbose:
            if self._log_callback:
                self._log_callback(msg)
            else:
                # 如果没有回调函数，使用标准logger
                from .logger import get_logger
                logger = get_logger()
                if level == "info":
                    logger.info(msg)
                else:
                    logger.debug(msg)
    
    def set_verbose(self, verbose: bool):
        """设置是否输出详细日志
        
        Args:
            verbose: True=输出详细日志，False=只输出关键信息
        """
        self._verbose = verbose
    
    def _load_state_mapping(self, mapping_path: str):
        """加载页面状态映射配置
        
        Args:
            mapping_path: 映射配置文件路径
        """
        # 动态 PageState 会自动从配置文件加载
        # 这里只需要构建类别名称到 PageState 的映射
        try:
            # 尝试在config目录查找
            if not os.path.exists(mapping_path):
                alt_mapping_path = os.path.join('config', mapping_path)
                if os.path.exists(alt_mapping_path):
                    mapping_path = alt_mapping_path
            
            # 强制重新加载 PageState 配置（确保使用最新配置）
            if os.path.exists(mapping_path):
                # 先重置加载状态，强制重新加载
                PageState._loaded = False
                PageState.load_from_config(Path(mapping_path))
            
            # 构建类别名称到 PageState 的映射
            # 从 PageState 的所有状态中构建映射
            self._class_to_state = {}
            
            # 同时加载配置文件，获取原始类别名称
            if os.path.exists(mapping_path):
                with open(mapping_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    mappings = config.get('mappings', {})
                    
                    # 使用原始类别名称作为键
                    for class_name, state_config in mappings.items():
                        state_name = state_config.get('state', 'UNKNOWN')
                        state_obj = PageState.get_by_name(state_name)
                        if state_obj:
                            self._class_to_state[class_name] = state_obj
            
            print(f"[智能检测器] ✓ 已加载 {len(self._class_to_state)} 个页面状态映射")
            
        except Exception as e:
            print(f"[智能检测器] ✗ 加载状态映射失败: {e}")
            # 使用默认映射
            self._class_to_state = {}
    
    def _load_classifier(self, model_path: str, classes_path: str):
        """加载页面分类器"""
        # [2026-02-21] 修复：将Path对象转换为字符串
        model_path = str(model_path)
        classes_path = str(classes_path)
        
        # 强制输出到文件，确保能看到
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        
        print("[智能检测器] 开始加载页面分类器...")
        sys.stdout.flush()
        
        if not HAS_TORCH or not HAS_PIL:
            # 关键错误，强制输出
            msg = "[智能检测器] ✗ PyTorch或PIL未安装"
            print(msg)
            sys.stdout.flush()
            sys.stderr.write(msg + "\n")
            sys.stderr.flush()
            return
        
        try:
            print("[智能检测器] 开始加载类别列表...")
            sys.stdout.flush()
            
            # 加载类别列表（尝试在models目录查找）
            if not os.path.exists(classes_path):
                alt_classes_path = os.path.join('models', classes_path)
                if os.path.exists(alt_classes_path):
                    classes_path = alt_classes_path
                else:
                    # 关键错误，强制输出
                    msg = f"[智能检测器] ✗ 类别文件不存在: {classes_path}"
                    print(msg)
                    sys.stdout.flush()
                    sys.stderr.write(msg + "\n")
                    sys.stderr.flush()
                    return
            
            with open(classes_path, 'r', encoding='utf-8') as f:
                self._classes = json.load(f)
            
            print(f"[智能检测器] ✓ 类别列表已加载: {len(self._classes)} 个类别")
            sys.stdout.flush()
            
            # 加载模型（尝试在models目录查找）
            print(f"[智能检测器] 开始加载模型文件: {model_path}")
            sys.stdout.flush()
            
            if not os.path.exists(model_path):
                alt_model_path = os.path.join('models', model_path)
                if os.path.exists(alt_model_path):
                    model_path = alt_model_path
                else:
                    # 关键错误，强制输出
                    msg = f"[智能检测器] ✗ 模型文件不存在: {model_path}"
                    print(msg)
                    sys.stdout.flush()
                    sys.stderr.write(msg + "\n")
                    sys.stderr.flush()
                    return
            
            # 设置设备
            print("[智能检测器] 检测计算设备...")
            sys.stdout.flush()
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"[智能检测器] 使用设备: {self._device}")
            sys.stdout.flush()
            
            # 定义模型架构
            print("[智能检测器] 定义模型架构...")
            sys.stdout.flush()
            
            class PageClassifier(nn.Module):
                def __init__(self, num_classes):
                    super(PageClassifier, self).__init__()
                    # 使用 MobileNetV2 架构（匹配训练脚本）
                    self.mobilenet = models.mobilenet_v2(weights=None)
                    # 替换分类器
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
            print(f"[智能检测器] 创建模型实例 ({num_classes} 个类别)...")
            sys.stdout.flush()
            model = PageClassifier(num_classes)
            
            print("[智能检测器] 加载模型权重...")
            sys.stdout.flush()
            checkpoint = torch.load(model_path, map_location=self._device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            print("[智能检测器] 将模型移至设备...")
            sys.stdout.flush()
            model = model.to(self._device)
            model.eval()
            self._classifier_model = model
            
            print("[智能检测器] 设置图片预处理...")
            sys.stdout.flush()
            # 设置图片预处理
            self._transform = transforms.Compose([
                transforms.Resize(self._img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            self._log(f"[智能检测器] ✓ 页面分类器已加载 (设备: {self._device})")
            
        except Exception as e:
            # 关键错误，强制输出
            import sys
            msg = f"[智能检测器] ✗ 加载页面分类器失败: {e}"
            print(msg)
            sys.stdout.flush()
            sys.stderr.write(msg + "\n")
            sys.stderr.flush()
            
            import traceback
            traceback.print_exc()
            sys.stderr.flush()
            
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
                self._log(f"[智能检测器] ✗ YOLO注册表不存在: {registry_path}")
                return
            
            with open(registry_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self._yolo_registry = data.get('models', {})
            
            self._log(f"[智能检测器] ✓ YOLO注册表已加载 ({len(self._yolo_registry)} 个模型)")
            
        except Exception as e:
            self._log(f"[智能检测器] ✗ 加载YOLO注册表失败: {e}")
    
    def _load_mapping(self, mapping_path: str):
        """加载页面-YOLO映射配置"""
        try:
            # [2026-02-21] 调试：强制输出，确保能看到
            print(f"[智能检测器] 开始加载映射配置: {mapping_path}")
            import sys
            sys.stdout.flush()
            
            # 如果路径不是绝对路径，尝试在多个目录中查找
            if not os.path.isabs(mapping_path) and not os.path.exists(mapping_path):
                # 尝试在config目录中查找
                config_mapping_path = os.path.join('config', mapping_path)
                print(f"[智能检测器] 尝试config目录: {config_mapping_path}, 存在={os.path.exists(config_mapping_path)}")
                sys.stdout.flush()
                if os.path.exists(config_mapping_path):
                    mapping_path = config_mapping_path
                else:
                    # 尝试在models目录中查找
                    models_mapping_path = os.path.join('models', mapping_path)
                    print(f"[智能检测器] 尝试models目录: {models_mapping_path}, 存在={os.path.exists(models_mapping_path)}")
                    sys.stdout.flush()
                    if os.path.exists(models_mapping_path):
                        mapping_path = models_mapping_path
            
            print(f"[智能检测器] 最终路径: {mapping_path}, 存在={os.path.exists(mapping_path)}")
            sys.stdout.flush()
            
            if not os.path.exists(mapping_path):
                print(f"[智能检测器] ✗ 映射配置不存在: {mapping_path}")
                sys.stdout.flush()
                self._log(f"[智能检测器] ✗ 映射配置不存在: {mapping_path}")
                return
            
            with open(mapping_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self._page_yolo_mapping = data.get('mapping', {})
            
            print(f"[智能检测器] ✓ 页面-YOLO映射已加载 ({len(self._page_yolo_mapping)} 个页面)")
            sys.stdout.flush()
            self._log(f"[智能检测器] ✓ 页面-YOLO映射已加载 ({len(self._page_yolo_mapping)} 个页面)")
            
        except Exception as e:
            self._log(f"[智能检测器] ✗ 加载映射配置失败: {e}")
    
    def _load_yolo_model(self, model_key: str) -> Optional[YOLO]:
        """加载YOLO模型（带缓存）
        
        # [2026-02-21] 修复：YOLO模型文件不存在时优雅降级
        # 原因：YOLO模型文件(.pt)被.gitignore排除，可能未训练或被误删
        # 解决：返回None而不是抛异常，让页面分类器继续工作
        """
        if not HAS_YOLO:
            return None
        
        # 检查缓存
        if model_key in self._yolo_models:
            return self._yolo_models[model_key]
        
        # 从注册表获取模型路径
        model_info = self._yolo_registry.get(model_key)
        if not model_info:
            # [2026-02-21] 降级：模型未注册时只记录警告，不影响页面分类
            import logging
            logging.getLogger(__name__).warning(f"YOLO模型未注册: {model_key}，将使用OCR降级方案")
            return None
        
        model_path = model_info.get('model_path')
        if not model_path:
            import logging
            logging.getLogger(__name__).warning(f"YOLO模型路径为空: {model_key}，将使用OCR降级方案")
            return None
        
        # 如果路径不是绝对路径，添加models/前缀
        if not os.path.isabs(model_path):
            # 尝试在models目录中查找
            models_path = os.path.join('models', model_path)
            if os.path.exists(models_path):
                model_path = models_path
            # 如果models/路径不存在，尝试原路径（兼容旧版本）
            elif not os.path.exists(model_path):
                # [2026-02-21] 降级：文件不存在时只记录警告，不影响页面分类
                import logging
                logging.getLogger(__name__).warning(
                    f"YOLO模型文件不存在: {model_path} (也尝试了 {models_path})，将使用OCR降级方案"
                )
                return None
        
        if not os.path.exists(model_path):
            # [2026-02-21] 降级：文件不存在时只记录警告，不影响页面分类
            import logging
            logging.getLogger(__name__).warning(f"YOLO模型文件不存在: {model_path}，将使用OCR降级方案")
            return None
        
        try:
            model = YOLO(model_path)
            self._yolo_models[model_key] = model
            self._log(f"[智能检测器] ✓ YOLO模型已加载: {model_key} ({model_path})")
            return model
        except Exception as e:
            # [2026-02-21] 降级：加载失败时只记录警告，不影响页面分类
            import logging
            logging.getLogger(__name__).warning(f"加载YOLO模型失败 {model_key}: {e}，将使用OCR降级方案")
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
            self._log(f"[智能检测器] ✗ 页面分类失败: {e}")
            return None, 0.0
    
    async def _ocr_assisted_detection(self, device_id: str, image: Image.Image, 
                                     predicted_class: str, predicted_confidence: float) -> Optional[Tuple[str, float, str]]:
        """使用OCR辅助判断页面类型（当置信度低于85%时）
        
        Args:
            device_id: 设备ID
            image: 截图
            predicted_class: 分类器预测的类别
            predicted_confidence: 分类器预测的置信度
            
        Returns:
            (页面类别, 置信度, OCR识别文本) 或 None（如果OCR无法辅助判断）
        """
        try:
            # OCR关键词映射（用于辅助判断页面类型）
            ocr_keywords = {
                '首页': ['首页', '推荐', '热门'],
                '首页公告': ['公告', '通知', '温馨提示', '确认', '知道了'],
                '首页异常代码弹窗': ['异常', '错误代码', '重试'],
                '手机号码错误': ['手机有误', '请重填', '手机号码错误', '手机号不存在'],
                '用户名或密码错误': ['用户名或密码错误', '密码错误', '友情提示'],
                '签到页': ['签到', '每日签到', '立即签到', '已签到'],
                '签到弹窗': ['签到成功', '获得', '积分'],
                '个人页_已登录': ['我的', '个人中心', '账户', '设置'],
                '个人页_未登录': ['登录', '注册', '立即登录'],
                '分类页': ['分类', '全部分类'],
                '搜索页': ['搜索', '请输入关键词'],
                '钱包页': ['钱包', '余额', '充值'],
                '转账页': ['转账', '收款人', '转账金额'],
                '转账确认弹窗': ['确认转账', '转账确认'],
            }
            
            # 使用OCR线程池识别屏幕文字
            from .ocr_thread_pool import get_ocr_pool
            from .ocr_image_processor import enhance_for_ocr
            
            ocr_pool = get_ocr_pool()
            if not ocr_pool:
                return None
            
            # 增强图像并进行OCR识别（超时3秒）
            enhanced_image = enhance_for_ocr(image)
            ocr_result = await ocr_pool.recognize(enhanced_image, timeout=3.0)
            
            if not ocr_result or not ocr_result.texts:
                return None
            
            # 提取所有识别到的文字
            all_text = ' '.join(ocr_result.texts)
            
            # 统计每个页面类型的关键词匹配数
            match_scores = {}
            for page_type, keywords in ocr_keywords.items():
                score = 0
                for keyword in keywords:
                    if keyword in all_text:
                        score += 1
                if score > 0:
                    match_scores[page_type] = score
            
            # 如果没有匹配到任何关键词，返回原始预测
            if not match_scores:
                return None
            
            # 找到匹配分数最高的页面类型
            best_match = max(match_scores.items(), key=lambda x: x[1])
            ocr_predicted_class = best_match[0]
            ocr_match_score = best_match[1]
            
            # 如果OCR预测与分类器预测一致，提升置信度到90%
            if ocr_predicted_class == predicted_class:
                new_confidence = max(predicted_confidence, 0.90)
                return (predicted_class, new_confidence, all_text)
            
            # 如果OCR预测不一致，但OCR匹配分数较高（>=2个关键词），使用OCR结果
            if ocr_match_score >= 2:
                return (ocr_predicted_class, 0.90, all_text)
            
            # 否则返回原始预测
            return None
            
        except Exception as e:
            self._log(f"[智能检测器] ✗ OCR辅助判断失败: {e}")
            return None
    
    def _detect_elements(self, image: Image.Image, page_class: str) -> List[PageElement]:
        """使用YOLO模型检测页面元素"""
        if not HAS_YOLO:
            return []
        
        # 获取该页面类型对应的YOLO模型
        mapping = self._page_yolo_mapping.get(page_class, {})
        yolo_models = mapping.get('yolo_models', [])
        
        if not yolo_models:
            return []
        
        elements = []
        
        # 按优先级加载和使用YOLO模型
        for model_info in sorted(yolo_models, key=lambda x: x.get('priority', 999)):
            model_key = model_info.get('model_key')
            if not model_key:
                continue
            
            model = self._load_yolo_model(model_key)
            if not model:
                continue
            
            try:
                # 使用YOLO检测
                results = model.predict(image, conf=0.25, verbose=False)
                
                for result in results:
                    boxes = result.boxes
                    
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
                
            except Exception as e:
                self._log(f"[智能检测器] ✗ YOLO检测失败 {model_key}: {e}")
                import traceback
                traceback.print_exc()
        
        return elements
    
    async def detect_page(self, device_id: str, use_cache: bool = True, 
                         detect_elements: bool = True,
                         use_ocr: bool = False,  # 兼容参数，智能检测器不使用OCR
                         use_template: bool = True,  # 兼容参数
                         use_dl: bool = True) -> IntegratedDetectionResult:  # 兼容参数
        """检测当前页面状态和元素
        
        Args:
            device_id: 设备 ID
            use_cache: 是否使用缓存
            detect_elements: 是否检测页面元素（使用YOLO）
            use_ocr: 兼容参数（智能检测器不使用OCR，忽略此参数）
            use_template: 兼容参数（智能检测器不使用模板匹配，忽略此参数）
            use_dl: 兼容参数（智能检测器始终使用深度学习，忽略此参数）
            
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
            print(f"[智能检测器] ✗ 页面分类器未加载: model={self._classifier_model is not None}, classes={self._classes is not None}")
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
            print(f"[智能检测器] ✗ 页面分类失败: page_class=None, confidence={confidence}")
            return IntegratedDetectionResult(
                state=PageState.UNKNOWN,
                confidence=0.0,
                details="页面分类失败",
                detection_method="integrated",
                detection_time=time.time() - start_time
            )
        
        # 如果置信度低于70%，使用OCR辅助判断
        ocr_text = None
        if confidence < 0.70:
            print(f"[智能检测器] ⚠️ 置信度较低 ({confidence:.2%})，使用OCR辅助判断...")
            try:
                # 使用asyncio.wait_for添加超时保护（5秒，必须大于内层OCR的3秒超时）
                ocr_result = await asyncio.wait_for(
                    self._ocr_assisted_detection(device_id, image, page_class, confidence),
                    timeout=5.0
                )
                if ocr_result:
                    page_class, confidence, ocr_text = ocr_result
                    print(f"[智能检测器] ✓ OCR辅助判断完成: {page_class} (置信度: {confidence:.2%})")
            except asyncio.TimeoutError:
                print(f"[智能检测器] ⚠️ OCR辅助判断超时（5秒），使用原始预测")
            except Exception as e:
                print(f"[智能检测器] ⚠️ OCR辅助判断出错: {e}，使用原始预测")
        
        # 映射到PageState
        state = self._class_to_state.get(page_class, PageState.UNKNOWN)
        
        # 如果映射失败，输出警告日志
        if state == PageState.UNKNOWN:
            print(f"[智能检测器] ⚠️ 未找到页面类别映射: '{page_class}' (置信度: {confidence:.2%})")
            print(f"[智能检测器] 当前已加载的映射数量: {len(self._class_to_state)}")
            print(f"[智能检测器] 提示: 请检查 config/page_state_mapping.json 中是否包含此类别")
            print(f"[智能检测器] 或点击'🔄 注册新模型'按钮自动注册")
        
        # 2. 使用YOLO检测页面元素（可选）
        elements = []
        yolo_model_used = None
        if detect_elements:
            elements = self._detect_elements(image, page_class)
            
            # [2026-02-22] 修复：当页面被误识别为广告页时，尝试使用转账页YOLO模型
            # 广告页不应该有元素检测需求，如果detect_elements=True但识别为广告页，
            # 很可能是页面分类错误，尝试用转账页YOLO模型检测
            if not elements and page_class == '广告页':
                self._log("[智能检测器] ⚠️ 广告页但需要检测元素，尝试使用转账页YOLO模型...")
                elements = self._detect_elements(image, '转账页')
                if elements:
                    self._log(f"[智能检测器] ✓ 转账页YOLO检测到 {len(elements)} 个元素")
            
            if elements:
                # 记录使用的YOLO模型
                mapping = self._page_yolo_mapping.get(page_class, {})
                yolo_models = mapping.get('yolo_models', [])
                if yolo_models:
                    yolo_model_used = yolo_models[0].get('model_key')
        
        # 构建结果
        if state == PageState.UNKNOWN:
            details = f"⚠️ 未映射的页面类别: {page_class} (置信度: {confidence:.2%})"
        elif state == PageState.LOGIN_ERROR and ocr_text:
            # 登录错误页面，提取具体错误信息
            details = f"登录错误: {ocr_text}"
        else:
            details = f"页面分类: {page_class} (置信度: {confidence:.2%})"
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
            self._log(f"[智能检测器] ✗ 未找到元素: {element_name}")
            return False
        
        # 点击元素中心点
        x, y = element.center
        await self.adb.tap(device_id, x, y)
        self._log(f"[智能检测器] ✓ 点击元素: {element_name} at ({x}, {y})")
        return True
    
    async def detect_page_with_priority(self, device_id: str, expected_pages: List[str], use_cache: bool = True) -> IntegratedDetectionResult:
        """使用优先级检测页面（兼容混合检测器的接口）
        
        智能检测器不使用模板匹配，所以忽略expected_pages参数，直接调用detect_page
        
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
            self._log("[智能检测器] ✗ YOLO未安装")
            return None
        
        try:
            # 获取截图
            image = await self._get_screenshot(device_id)
            if not image:
                self._log("[智能检测器] ✗ 无法获取截图")
                return None
            
            # 直接使用 page_type 作为 model_key（注册表中的键）
            self._log(f"[智能检测器] 尝试加载模型: {page_type}")
            model = self._load_yolo_model(page_type)
            
            if not model:
                self._log(f"[智能检测器] ✗ 无法加载模型: {page_type}")
                return None
            
            self._log(f"[智能检测器] ✓ 模型已加载，开始检测...")
            
            # 使用YOLO检测
            results = model.predict(image, conf=conf_threshold, verbose=False)
            
            # 查找指定按钮
            for result in results:
                boxes = result.boxes
                self._log(f"[智能检测器] 检测到 {len(boxes)} 个对象")
                
                for box in boxes:
                    cls = int(box.cls[0])
                    class_name = result.names[cls]
                    conf = float(box.conf[0])
                    
                    self._log(f"[智能检测器] 检测到: {class_name} (置信度: {conf:.2%})")
                    
                    # 检查是否是目标按钮
                    if button_name in class_name or class_name in button_name:
                        # 提取边界框
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # 计算中心点
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        
                        self._log(f"[智能检测器] ✓ YOLO检测到按钮: {class_name} at ({center_x}, {center_y}), 置信度: {conf:.2%}")
                        
                        return (center_x, center_y)
            
            # 未找到按钮
            self._log(f"[智能检测器] ✗ 未找到按钮: {button_name}")
            return None
            
        except Exception as e:
            self._log(f"[智能检测器] ✗ YOLO按钮检测失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def close_popup(self, device_id: str, timeout: float = 15.0, known_popup_type: str = None, max_attempts: int = 3) -> bool:
        """自动关闭弹窗（带超时保护和重试机制）
        
        从混合检测器复制的完整实现，适配智能检测器
        
        Args:
            device_id: 设备ID
            timeout: 总超时时间（秒），默认15秒
            known_popup_type: 已知的弹窗类型（可选），如果提供则跳过OCR识别
                            可选值: "home_announcement", "user_agreement", "login_error", "generic"
            max_attempts: 最大重试次数（默认3次，可从GUI配置传入）
        
        Returns:
            是否成功关闭
        """
        import asyncio
        
        try:
            # 使用 asyncio.wait_for 为整个关闭流程添加超时
            return await asyncio.wait_for(
                self._close_popup_impl(device_id, known_popup_type, max_attempts),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            self._log(f"[智能检测器] ✗ 关闭弹窗超时（{timeout}秒）")
            return False
    
    async def _close_popup_impl(self, device_id: str, known_popup_type: str = None, max_attempts: int = 3) -> bool:
        """关闭弹窗的实际实现（从混合检测器复制）
        
        Args:
            device_id: 设备ID
            known_popup_type: 已知的弹窗类型（可选），如果提供则跳过OCR识别
        """
        from .retry_helper import retry_until_success
        from .ocr_thread_pool import get_ocr_pool
        
        # 弹窗按钮坐标 (540x960)
        POPUP_BUTTONS = {
            'user_agreement': (270, 600),      # 服务协议弹窗"同意并接受"
            'user_agreement_alt': (270, 608),  # 服务协议弹窗备用坐标
            'home_announcement': (290, 210),   # 主页广告弹窗关闭按钮（X按钮位置）
            'login_error': (436, 557),         # 登录错误确定按钮
            'generic': (270, 600),             # 通用弹窗
        }
        
        # 签到弹窗关闭按钮坐标（MuMu模拟器 540x960）
        CHECKIN_POPUP_CLOSE = [
            (270, 812),  # 中心位置
            (278, 811),  # 右偏
            (274, 811),  # 中右
        ]
        
        # 优先使用已知的弹窗类型（避免重复OCR识别）
        popup_type = known_popup_type
        button_pos = None
        current_screenshot = None
        
        # 如果已知弹窗类型，直接使用
        if known_popup_type:
            self._log(f"[智能检测器] 使用已知弹窗类型: {known_popup_type}")
            if known_popup_type in POPUP_BUTTONS:
                button_pos = POPUP_BUTTONS[known_popup_type]
        else:
            # 检查当前页面状态
            result = await self.detect_page(device_id, use_cache=False)
            if result.state != PageState.POPUP and result.state != PageState.CHECKIN_POPUP:
                self._log(f"[智能检测器] 当前不是弹窗页面，无需关闭")
                return True
            
            # 如果还没有截图，获取当前截图用于OCR识别
            if not current_screenshot:
                screenshot_data = await self.adb.screencap(device_id)
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
                        self._log(f"[智能检测器] OCR识别到: {texts[:5] if texts else '无'}...")
                        
                        # 登录错误弹窗（最高优先级）
                        if "友情提示" in text_str:
                            popup_type = "login_error"
                            button_pos = POPUP_BUTTONS['login_error']
                            self._log(f"[智能检测器] 类型: {popup_type} (OCR检测)")
                        # 用户协议弹窗
                        elif any(kw in text_str for kw in ["用户协议", "隐私政策", "服务协议", "隐私协议"]):
                            if "登录" not in text_str or "同意并接受" in text_str:
                                popup_type = "user_agreement"
                                button_pos = POPUP_BUTTONS['user_agreement']
                                self._log(f"[智能检测器] 类型: {popup_type} (OCR检测)")
                        # 主页公告弹窗
                        elif any(kw in text_str for kw in ["公告", "活动", "恭喜", "领取", "×"]):
                            popup_type = "home_announcement"
                            button_pos = POPUP_BUTTONS['home_announcement']
                            self._log(f"[智能检测器] 类型: {popup_type} (OCR检测)")
                        # 通用弹窗
                        elif any(kw in text_str for kw in ["确定", "关闭", "取消", "知道了", "我知道了"]):
                            popup_type = "generic"
                            button_pos = POPUP_BUTTONS['generic']
                            self._log(f"[智能检测器] 类型: {popup_type} (OCR检测)")
                        else:
                            popup_type = "unknown"
                            button_pos = POPUP_BUTTONS['generic']
                            self._log(f"[智能检测器] 类型: {popup_type} (OCR检测)")
                except Exception as e:
                    self._log(f"[智能检测器] OCR检测失败: {e}")
                    popup_type = "generic"
                    button_pos = POPUP_BUTTONS['generic']
            else:
                popup_type = "generic"
                button_pos = POPUP_BUTTONS['generic']
        
        # 如果是首页公告弹窗，点击弹窗外部关闭
        if popup_type == "home_announcement":
            self._log(f"[智能检测器] 首页公告弹窗，点击顶部区域关闭...")
            
            # 使用 POPUP_BUTTONS 中配置的坐标（不要硬编码）
            close_x, close_y = button_pos if button_pos else POPUP_BUTTONS['home_announcement']
            
            # 使用传入的重试次数（从GUI配置）
            self._log(f"[智能检测器] 最多尝试 {max_attempts} 次")
            for attempt in range(1, max_attempts + 1):
                self._log(f"[智能检测器] 第 {attempt}/{max_attempts} 次点击 ({close_x}, {close_y})...")
                await self.adb.tap(device_id, close_x, close_y)
                await asyncio.sleep(1.0)  # 等待页面响应
                
                result = await self.detect_page(device_id)
                
                # 成功条件：到达首页
                if result.state == PageState.HOME:
                    # 成功关闭弹窗并到达首页
                    self._log(f"[智能检测器] ✓ 成功关闭首页公告弹窗，当前页面: {result.state.value}")
                    return True
                elif result.state == PageState.CATEGORY:
                    # 如果误点进入分类页，点击首页按钮回到首页
                    self._log(f"[智能检测器] ✗ 点击误触进入分类页，点击首页按钮...")
                    # 分类页必须点击首页按钮，不能按返回键
                    await self.adb.tap(device_id, 90, 920)  # 首页按钮坐标
                    await asyncio.sleep(0.5)  # 等待页面切换
                    # 清除缓存，立即检测是否回到首页
                    self.clear_cache(device_id)
                    result = await self.detect_page(device_id)
                    if result.state == PageState.HOME:
                        self._log(f"[智能检测器] ✓ 已返回首页")
                        return True
                    # 如果还没回到首页，继续下一次尝试
                    continue
                elif result.state == PageState.POPUP:
                    # 仍然是弹窗，继续下一次尝试
                    self._log(f"[智能检测器] 仍是弹窗，继续尝试...")
                    continue
                else:
                    # 其他异常状态（退出应用、UNKNOWN等），返回失败让上层处理
                    self._log(f"[智能检测器] ✗ 点击后页面状态异常: {result.state.value}，返回失败")
                    return False
            
            # 所有尝试都失败，最后尝试按返回键
            self._log(f"[智能检测器] {max_attempts} 次点击都失败，尝试按返回键...")
            await self.adb.press_back(device_id)
            await asyncio.sleep(1.0)
            
            result = await self.detect_page(device_id)
            if result.state == PageState.HOME:
                self._log(f"[智能检测器] ✓ 返回键成功关闭首页公告弹窗，当前页面: {result.state.value}")
                return True
            else:
                self._log(f"[智能检测器] ✗ 无法关闭首页公告弹窗，当前页面: {result.state.value}")
                return False
        
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
                        self._log(f"[智能检测器] 检测到签到奖励弹窗 (OCR确认)")
            except:
                pass
        
        # 如果是签到弹窗，使用专用坐标
        if is_checkin_popup:
            self._log(f"[智能检测器] 使用签到弹窗专用坐标...")
            for i, (x, y) in enumerate(CHECKIN_POPUP_CLOSE, 1):
                self._log(f"[智能检测器] 尝试位置 {i}/3: ({x}, {y})")
                await self.adb.tap(device_id, x, y)
                await asyncio.sleep(2)
                
                result = await self.detect_page(device_id)
                if result.state != PageState.POPUP and result.state != PageState.CHECKIN_POPUP:
                    self._log(f"[智能检测器] ✓ 成功关闭签到弹窗")
                    return True
            
            self._log(f"[智能检测器] ⚠️ 签到弹窗专用坐标都失败，尝试其他方法...")
        
        # 使用预设位置
        if button_pos:
            await self.adb.tap(device_id, button_pos[0], button_pos[1])
            await asyncio.sleep(2)
            
            result = await self.detect_page(device_id)
            if result.state != PageState.POPUP:
                self._log(f"[智能检测器] ✓ 成功关闭")
                return True
            else:
                self._log(f"[智能检测器] ⚠️ 预设位置点击失败，仍是弹窗")
            
            # 尝试其他预设位置
            if popup_type in ["unknown", "home_announcement", "user_agreement"]:
                self._log(f"[智能检测器] 尝试其他预设位置...")
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
                        self._log(f"[智能检测器] ✓ 成功关闭（位置: {pos}）")
                        return True
                
                # 尝试按返回键
                self._log(f"[智能检测器] 所有位置都失败，尝试按返回键...")
                await self.adb.press_back(device_id)
                await asyncio.sleep(1.5)
                
                result = await self.detect_page(device_id)
                if result.state != PageState.POPUP:
                    self._log(f"[智能检测器] ✓ 成功关闭（返回键）")
                    return True
                else:
                    self._log(f"[智能检测器] ✗ 返回键也失败，弹窗无法关闭")
                    return False
        
        self._log(f"[智能检测器] ✗ 无法关闭弹窗")
        return False
