"""
YOLO检测器 - 使用训练好的YOLO模型检测和定位目标（按钮、文本区域等）
YOLO Detector - Detect and locate objects (buttons, text regions, etc.) using trained YOLO models
"""

import json
from typing import Optional, List, Tuple, Dict
from io import BytesIO
from pathlib import Path

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

from .adb_bridge import ADBBridge


class Detection:
    """检测结果"""
    def __init__(self, class_name: str, confidence: float, bbox: Tuple[int, int, int, int]):
        self.class_name = class_name  # 目标类别名称
        self.confidence = confidence  # 置信度
        self.bbox = bbox  # 边界框 (x1, y1, x2, y2)
        self.center = self._calculate_center()
        self.click_point = self._calculate_click_point()  # 更准确的点击位置
    
    def _calculate_center(self) -> Tuple[int, int]:
        """计算目标中心点坐标"""
        x1, y1, x2, y2 = self.bbox
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        return (center_x, center_y)
    
    def _calculate_click_point(self) -> Tuple[int, int]:
        """计算更准确的点击位置
        
        使用边界框上方偏移位置（经过测试，这个位置最准确）
        """
        x1, y1, x2, y2 = self.bbox
        center_x = int((x1 + x2) / 2)
        # 使用边界框上方20像素的位置（经过实际测试验证）
        click_y = y1 + 20
        return (center_x, click_y)
    
    def __repr__(self):
        return f"Detection(class={self.class_name}, conf={self.confidence:.2f}, center={self.center}, click={self.click_point})"


class YoloDetector:
    """YOLO检测器
    
    使用训练好的YOLO模型检测页面上的目标（按钮、文本区域等）并返回坐标
    """
    
    def __init__(self, adb: ADBBridge, registry_path='yolo_model_registry.json', log_callback=None):
        """初始化YOLO检测器
        
        Args:
            adb: ADB桥接对象
            registry_path: 模型注册表路径
            log_callback: 日志回调函数
        """
        self.adb = adb
        self._log_callback = log_callback
        self._models = {}  # 缓存已加载的模型 {page_type: YOLO_model}
        self._registry = self._load_registry(registry_path)
        
        # GPU加速配置
        self._device = 'cpu'  # 默认使用CPU
        self._enable_gpu_if_available()
        
        # 只在缺少依赖时输出错误日志
        if not HAS_YOLO:
            self._log("[YOLO] ✗ ultralytics未安装")
        if not HAS_PIL:
            self._log("[YOLO] ✗ PIL未安装")
    
    def _log(self, msg: str):
        """输出日志"""
        if self._log_callback:
            self._log_callback(msg)
        else:
            print(msg)
    
    def _enable_gpu_if_available(self):
        """启用GPU加速（如果可用）"""
        try:
            import torch
            
            # 检查CUDA是否可用
            if torch.cuda.is_available():
                self._device = 'cuda'
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self._log(f"[YOLO] ✓ GPU加速已启用: {gpu_name} ({gpu_memory:.1f}GB)")
                
                # 设置CUDA优化
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
            else:
                self._log(f"[YOLO] ⚠️ 未检测到CUDA GPU，使用CPU")
        except ImportError:
            self._log(f"[YOLO] ⚠️ PyTorch未安装，无法使用GPU加速")
        except Exception as e:
            self._log(f"[YOLO] ⚠️ GPU初始化失败: {e}")
    
    def _load_registry(self, registry_path: str) -> dict:
        """加载模型注册表"""
        try:
            with open(registry_path, 'r', encoding='utf-8') as f:
                registry = json.load(f)
            # 静默加载，不输出日志
            return registry
        except Exception as e:
            # 只在失败时输出错误
            self._log(f"[YOLO] ✗ 加载注册表失败: {e}")
            return {'models': {}}
    
    def _load_model(self, page_type: str) -> Optional[YOLO]:
        """加载指定页面类型的YOLO模型
        
        Args:
            page_type: 页面类型（如 'homepage', 'login', 'checkin'）
            
        Returns:
            YOLO模型对象，如果加载失败返回None
        """
        if not HAS_YOLO:
            self._log(f"[YOLO] ✗ ultralytics未安装，无法加载模型 {page_type}")
            return None
        
        # 检查缓存
        if page_type in self._models:
            return self._models[page_type]
        
        # 从注册表获取模型信息
        model_info = self._registry.get('models', {}).get(page_type)
        if not model_info:
            self._log(f"[YOLO] ✗ 模型 {page_type} 未在注册表中")
            return None
        
        model_path = model_info.get('model_path')
        if not model_path:
            self._log(f"[YOLO] ✗ 模型 {page_type} 缺少 model_path")
            return None
        
        if not Path(model_path).exists():
            self._log(f"[YOLO] ✗ 模型文件不存在: {model_path}")
            return None
        
        try:
            self._log(f"[YOLO] 正在加载模型: {model_path}")
            model = YOLO(model_path)
            
            # 将模型移动到GPU（如果可用）
            if self._device == 'cuda':
                model.to('cuda')
                self._log(f"[YOLO] ✓ 模型已加载到GPU: {page_type}")
            else:
                self._log(f"[YOLO] ✓ 模型加载成功（CPU）: {page_type}")
            
            self._models[page_type] = model
            return model
        except Exception as e:
            # 只在异常时输出错误
            self._log(f"[YOLO] ✗ 加载模型失败: {e}")
            return None
    
    async def detect(self, device_id: str, page_type: str, 
                    conf_threshold: float = 0.5) -> List[Detection]:
        """检测页面上的目标（按钮、文本区域等）
        
        Args:
            device_id: 设备ID
            page_type: 页面类型（如 'homepage', 'login', 'checkin'）
            conf_threshold: 置信度阈值（默认0.5）
            
        Returns:
            检测结果列表
        """
        if not HAS_YOLO or not HAS_PIL:
            self._log(f"[YOLO] ✗ 缺少依赖库，无法检测")
            return []
        
        # 加载模型
        model = self._load_model(page_type)
        if not model:
            self._log(f"[YOLO] ✗ 模型 {page_type} 加载失败，跳过检测")
            return []
        
        # 获取截图
        screenshot_data = await self.adb.screencap(device_id)
        if not screenshot_data:
            self._log(f"[YOLO] ✗ 截图失败")
            return []
        
        try:
            # 转换为PIL图像
            image = Image.open(BytesIO(screenshot_data))
            
            # YOLO检测（使用GPU加速）
            self._log(f"[YOLO] 开始检测 {page_type}，置信度阈值={conf_threshold}，设备={self._device}")
            results = model.predict(image, conf=conf_threshold, verbose=False, device=self._device)
            
            # 解析检测结果
            detections = []
            if results and len(results) > 0:
                result = results[0]
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    # 获取边界框坐标
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    
                    # 获取置信度
                    conf = float(boxes.conf[i].cpu().numpy())
                    
                    # 获取类别名称
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    class_name = model.names[cls_id]
                    
                    detection = Detection(class_name, conf, (x1, y1, x2, y2))
                    detections.append(detection)
            
            if detections:
                self._log(f"[YOLO] ✓ 检测到 {len(detections)} 个目标")
            else:
                self._log(f"[YOLO] ⚠️ 未检测到任何目标（可能页面不匹配或置信度阈值过高）")
            
            return detections
            
        except Exception as e:
            # 只在异常时输出错误
            self._log(f"[YOLO] ✗ 检测失败: {e}")
            return []
    
    async def find_object(self, device_id: str, page_type: str, object_name: str,
                         conf_threshold: float = 0.5) -> Optional[Tuple[int, int]]:
        """查找指定目标的坐标
        
        Args:
            device_id: 设备ID
            page_type: 页面类型
            object_name: 目标名称（如 "每日签到按钮", "登陆按钮", "昵称文本"）
            conf_threshold: 置信度阈值
            
        Returns:
            目标点击位置坐标 (x, y)，如果未找到返回None
        """
        detections = await self.detect(device_id, page_type, conf_threshold)
        
        # 查找匹配的目标
        for det in detections:
            if object_name in det.class_name or det.class_name in object_name:
                # 返回更准确的点击位置（边界框上方1/3处）
                return det.click_point
        
        # 静默失败
        return None
    
    async def click_object(self, device_id: str, page_type: str, object_name: str,
                          conf_threshold: float = 0.5) -> bool:
        """检测并点击指定目标
        
        Args:
            device_id: 设备ID
            page_type: 页面类型
            object_name: 目标名称
            conf_threshold: 置信度阈值
            
        Returns:
            是否成功点击
        """
        # 查找目标
        object_pos = await self.find_object(device_id, page_type, object_name, conf_threshold)
        
        if object_pos:
            # 点击目标
            x, y = object_pos
            await self.adb.tap(device_id, x, y)
            # 静默点击，不输出日志
            return True
        
        return False
    
    def get_available_models(self) -> Dict[str, dict]:
        """获取所有可用的模型信息
        
        Returns:
            模型信息字典 {page_type: model_info}
        """
        return self._registry.get('models', {})
    
    def get_model_classes(self, page_type: str) -> List[str]:
        """获取指定页面类型模型可以检测的类别
        
        Args:
            page_type: 页面类型
            
        Returns:
            类别名称列表
        """
        model_info = self._registry.get('models', {}).get(page_type, {})
        return model_info.get('classes', [])
