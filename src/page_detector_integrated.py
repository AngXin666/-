"""
YOLO识别器
YOLO Detection Only

只负责 YOLO 元素检测，不进行页面类型分类
页面类型分类使用专用模型（PageDetectorDL）
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
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

from .adb_bridge import ADBBridge
from .page_detector import PageDetectionResult
from .page_state_dynamic import PageState, PageStateType


@dataclass
class PageElement:
    """YOLO检测结果"""
    class_name: str  # 元素类别名称
    confidence: float  # 置信度
    bbox: Tuple[int, int, int, int]  # 边界框 (x1, y1, x2, y2)
    center: Tuple[int, int]  # 中心点 (x, y)


@dataclass
class IntegratedDetectionResult(PageDetectionResult):
    """YOLO识别结果"""
    elements: List[PageElement] = None  # YOLO检测到的元素
    yolo_model_used: str = None  # 使用的YOLO模型
    
    def __post_init__(self):
        if self.elements is None:
            self.elements = []


class PageDetectorIntegrated:
    """YOLO识别器"""
    
    def __init__(self, adb: ADBBridge, 
                 classifier_model_path=None,  # 已废弃，保留以兼容旧代码
                 classes_path=None,  # 已废弃，保留以兼容旧代码
                 # [2026-03-12] 修改路径：YOLO注册表文件移动到models目录
                 yolo_registry_path='models/yolo_model_registry.json',
                 mapping_path=None,  # 已废弃，保留以兼容旧代码
                 state_mapping_path=None,  # 已废弃，保留以兼容旧代码
                 log_callback=None):
        """初始化YOLO识别器（仅YOLO元素检测，不包含页面分类）"""
        self.adb = adb
        self._log_callback = log_callback
        self._verbose = False
        
        # YOLO模型相关
        self._yolo_models = {}  # 缓存已加载的YOLO模型
        self._yolo_registry = {}
        
        # 初始化检测缓存
        from .performance.detection_cache import DetectionCache
        self._detection_cache = DetectionCache(ttl=0.5)
        
        self._load_yolo_registry(yolo_registry_path)
    
    def _log(self, msg: str, level: str = "debug"):
        """输出日志"""
        if not self._verbose:
            return
        
        if level == "info" or self._verbose:
            if self._log_callback:
                self._log_callback(msg)
    
    def set_verbose(self, verbose: bool):
        """设置是否输出详细日志"""
        self._verbose = verbose
    
    def _load_yolo_registry(self, registry_path: str):
        """加载YOLO模型注册表"""
        try:
            if not os.path.isabs(registry_path) and not os.path.exists(registry_path):
                models_registry_path = os.path.join('models', registry_path)
                if os.path.exists(models_registry_path):
                    registry_path = models_registry_path
            
            if not os.path.exists(registry_path):
                self._log(f"[YOLO] ✗ YOLO注册表不存在: {registry_path}")
                return
            
            with open(registry_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self._yolo_registry = data.get('models', {})
            
        except Exception as e:
            pass
    
    def _load_yolo_model(self, model_key: str) -> Optional[YOLO]:
        """加载YOLO模型（带缓存）"""
        # [2026-03-11] 优化日志：移除所有控制台DEBUG输出
        if not HAS_YOLO:
            return None
        
        if model_key in self._yolo_models:
            return self._yolo_models[model_key]
        
        model_info = self._yolo_registry.get(model_key)
        if not model_info:
            return None
        
        model_path = model_info.get('model_path')
        if not model_path:
            return None
        
        # [2026-03-10] 修复原因：尝试在models目录下查找模型文件
        if not Path(model_path).exists():
            models_path = f"models/{model_path}"
            if Path(models_path).exists():
                model_path = models_path
            else:
                return None
        
        if not os.path.exists(model_path):
            return None
        
        try:
            model = YOLO(model_path)
            self._yolo_models[model_key] = model
            return model
        except Exception as e:
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
    
    def _detect_elements_by_model(self, image: Image.Image, model_key: str) -> List[PageElement]:
        """使用指定的YOLO模型识别页面元素"""
        # [2026-03-11] 优化日志：移除所有控制台DEBUG输出
        if not HAS_YOLO:
            return []
        
        model = self._load_yolo_model(model_key)
        if not model:
            return []
        
        elements = []
        
        try:
            results = model.predict(image, conf=0.25, verbose=False)
            
            for result in results:
                boxes = result.boxes
                
                if boxes is None or len(boxes) == 0:
                    continue
                
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = result.names[cls]
                    
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
            pass
        
        return elements
    
    async def detect_page(self, device_id: str, use_cache: bool = True, 
                         detect_elements: bool = True,
                         use_ocr: bool = False,
                         use_template: bool = True,
                         use_dl: bool = True) -> IntegratedDetectionResult:
        """检测当前页面状态和元素
        
        注意：此方法只负责 YOLO 元素检测，不进行页面类型分类
        """
        start_time = time.time()
        
        if use_cache:
            cached_result = self._detection_cache.get(device_id)
            if cached_result is not None:
                cached_result.cached = True
                cached_result.detection_time = time.time() - start_time
                return cached_result
        
        image = await self._get_screenshot(device_id)
        
        if not image:
            return IntegratedDetectionResult(
                state=PageState.UNKNOWN,
                confidence=0.0,
                details="无法截取屏幕",
                detection_method="integrated",
                detection_time=time.time() - start_time
            )
        
        elements = []
        yolo_model_used = None
        if detect_elements:
            # [2026-03-11] 修复原因：智能选择YOLO模型，而不是硬编码使用transfer模型
            # 尝试多个可能的模型，按优先级顺序
            model_candidates = [
                'profile_detailed',  # 个人页详细标注检测模型（8区域）
                'balance',          # 余额积分检测模型
                'profile_logged',   # 已登录个人页检测模型（昵称和ID）
                'transfer',         # 转账页检测模型（作为后备）
            ]
            
            for model_key in model_candidates:
                elements = self._detect_elements_by_model(image, model_key)
                if elements:
                    yolo_model_used = model_key
                    break
        
        details = f"YOLO元素检测"
        if elements:
            details += f": 检测到 {len(elements)} 个元素"
        else:
            details += ": 未检测到元素"
        
        result = IntegratedDetectionResult(
            state=PageState.UNKNOWN,
            confidence=0.0,
            details=details,
            detection_method="integrated",
            detection_time=time.time() - start_time,
            cached=False,
            elements=elements,
            yolo_model_used=yolo_model_used
        )
        
        if use_cache:
            self._detection_cache.set(device_id, result)
        
        return result
    
    async def detect_elements_yolo(self, device_id: str, model_key: str, 
                                  target_classes: List[str] = None,
                                  conf_threshold: float = 0.25) -> Optional[Dict[str, List[Dict]]]:
        """使用YOLO检测指定页面的元素
        
        Args:
            device_id: 设备ID
            model_key: YOLO模型键名（如"签到页"）
            target_classes: 目标检测类别列表（如["签到次数", "签到按钮"]）
            conf_threshold: 置信度阈值
            
        Returns:
            Dict: 检测结果，格式为 {class_name: [{'bbox': (x1,y1,x2,y2), 'confidence': float}]}
        """
        image = await self._get_screenshot(device_id)
        if not image:
            return None
        
        # 使用指定模型检测元素
        elements = self._detect_elements_by_model(image, model_key)
        if not elements:
            return None
        
        # 按类别组织结果
        result = {}
        for element in elements:
            if element.confidence < conf_threshold:
                continue
                
            # 如果指定了目标类别，只返回匹配的类别
            if target_classes and element.class_name not in target_classes:
                continue
            
            if element.class_name not in result:
                result[element.class_name] = []
            
            result[element.class_name].append({
                'bbox': element.bbox,
                'confidence': element.confidence,
                'center': element.center
            })
        
        return result if result else None

    async def find_button_yolo(self, device_id: str, page_type: str, button_name: str,
                              conf_threshold: float = 0.5) -> Optional[Tuple[int, int]]:
        """查找指定按钮的坐标（兼容性方法）
        
        Args:
            device_id: 设备ID
            page_type: 页面类型
            button_name: 按钮名称
            conf_threshold: 置信度阈值
            
        Returns:
            按钮点击位置坐标 (x, y)，如果未找到返回None
        """
        # [2026-03-11] 修复原因：添加find_button_yolo方法以兼容Navigator等模块的调用
        image = await self._get_screenshot(device_id)
        if not image:
            return None
        
        # 使用transfer模型检测元素
        elements = self._detect_elements_by_model(image, 'transfer')
        
        # 查找匹配的按钮
        for element in elements:
            if button_name in element.class_name or element.class_name in button_name:
                return element.center
        
        return None
    
    def clear_cache(self, device_id: str = None):
        """清除缓存"""
        if hasattr(self, '_detection_cache'):
            self._detection_cache.clear(device_id)