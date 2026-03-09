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

# [2026-03-02] 删除未使用的导入：torch, nn, transforms, models
# 这些是通用页面分类器使用的，现在已经删除

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
                 yolo_registry_path='yolo_model_registry.json',
                 mapping_path=None,  # 已废弃，保留以兼容旧代码
                 state_mapping_path=None,  # 已废弃，保留以兼容旧代码
                 log_callback=None):
        """初始化YOLO识别器（仅YOLO元素检测，不包含页面分类）
        
        Args:
            adb: ADB 桥接器实例
            classifier_model_path: 已废弃（保留参数以兼容旧代码）
            classes_path: 已废弃（保留参数以兼容旧代码）
            yolo_registry_path: YOLO模型注册表路径
            mapping_path: 已废弃（保留参数以兼容旧代码）
            state_mapping_path: 已废弃（保留参数以兼容旧代码）
            log_callback: 日志回调函数
        """
        self.adb = adb
        self._log_callback = log_callback
        self._verbose = False
        
        # YOLO模型相关
        self._yolo_models = {}  # 缓存已加载的YOLO模型
        self._yolo_registry = {}
        
        # 初始化检测缓存
        from .performance.detection_cache import DetectionCache
        self._detection_cache = DetectionCache(ttl=0.5)
        
        # [2026-03-02] 删除通用页面分类器：不再加载 _load_classifier、_load_state_mapping、_load_mapping
        # 系统现在使用专用模型（PageDetectorDL）：启动专用、登录专用、签到专用、转账专用、个人页专用
        # PageDetectorIntegrated 只负责 YOLO 元素检测
        self._load_yolo_registry(yolo_registry_path)
    
    def _log(self, msg: str, level: str = "debug"):
        """输出日志"""
        if not self._verbose:
            return
        
        if level == "info" or self._verbose:
            if self._log_callback:
                self._log_callback(msg)
            else:
                from .logger import get_logger
                logger = get_logger()
                if level == "info":
                    logger.info(msg)
                else:
                    logger.debug(msg)
    
    def set_verbose(self, verbose: bool):
        """设置是否输出详细日志"""
        self._verbose = verbose
    
    # [2026-03-02] 删除未使用的方法：_load_state_mapping, _load_classifier, _classify_page, _ocr_assisted_detection, _load_mapping
    # 这些方法是通用页面分类器的代码，现在系统使用专用模型（PageDetectorDL）
    
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
        if not HAS_YOLO:
            return None
        
        if model_key in self._yolo_models:
            return self._yolo_models[model_key]
        
        model_info = self._yolo_registry.get(model_key)
        if not model_info:
            import logging
            logging.getLogger(__name__).warning(f"YOLO模型未注册: {model_key}，将使用OCR降级方案")
            return None
        
        model_path = model_info.get('model_path')
        if not model_path:
            import logging
            logging.getLogger(__name__).warning(f"YOLO模型路径为空: {model_key}，将使用OCR降级方案")
            return None
        
        if not os.path.isabs(model_path):
            models_path = os.path.join('models', model_path)
            if os.path.exists(models_path):
                model_path = models_path
            elif not os.path.exists(model_path):
                return None
        
        if not os.path.exists(model_path):
            return None
        
        try:
            model = YOLO(model_path)
            self._yolo_models[model_key] = model
            return model
        except Exception as e:
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
    
    def _detect_elements_by_model(self, image: Image.Image, model_key: str) -> List[PageElement]:
        """使用指定的YOLO模型识别页面元素"""
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
            import traceback
            traceback.print_exc()
        
        return elements
    
    # [2026-03-02] 删除未使用的方法：_detect_elements（已废弃，使用 _detect_elements_by_model 代替）
    
    async def detect_page(self, device_id: str, use_cache: bool = True, 
                         detect_elements: bool = True,
                         use_ocr: bool = False,
                         use_template: bool = True,
                         use_dl: bool = True) -> IntegratedDetectionResult:
        """检测当前页面状态和元素
        
        注意：此方法只负责 YOLO 元素检测，不进行页面类型分类
        页面类型检测应该使用专用模型（启动专用、签到专用、转账专用、个人页专用）
        """
        start_time = time.time()
        
        if use_cache:
            cached_result = self._detection_cache.get(device_id)
            if cached_result is not None:
                cached_result.cached = True
                cached_result.detection_time = time.time() - start_time
                return cached_result
        
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
        
        # [2026-03-02] 移除页面分类逻辑：只负责元素检测，不进行页面类型分类
        elements = []
        yolo_model_used = None
        if detect_elements:
            elements = self._detect_elements_by_model(image, 'transfer')
            if elements:
                yolo_model_used = 'transfer'
        
        details = f"YOLO元素检测"
        if elements:
            details += f": 检测到 {len(elements)} 个元素"
        else:
            details += ": 未检测到元素（当前检测器不支持元素检测）"
        
        # [2026-03-02] 返回 UNKNOWN 状态，因为此检测器不负责页面类型分类
        result = IntegratedDetectionResult(
            state=PageState.UNKNOWN,
            confidence=0.0,  # 不进行页面分类，置信度为0
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
    
    async def get_element(self, device_id: str, element_name: str) -> Optional[PageElement]:
        """获取指定名称的页面元素"""
        result = await self.detect_page(device_id, detect_elements=True)
        
        for element in result.elements:
            if element.class_name == element_name:
                return element
        
        return None
    
    async def click_element(self, device_id: str, element_name: str) -> bool:
        """点击指定名称的页面元素"""
        element = await self.get_element(device_id, element_name)
        if not element:
            return False
        
        x, y = element.center
        await self.adb.tap(device_id, x, y)
        return True
    
    async def detect_page_with_priority(self, device_id: str, expected_pages: List[str], use_cache: bool = True) -> IntegratedDetectionResult:
        """使用优先级检测页面"""
        return await self.detect_page(device_id, use_cache=use_cache, detect_elements=False)
    
    def clear_cache(self, device_id: str = None):
        """清除缓存"""
        if hasattr(self, '_detection_cache'):
            self._detection_cache.clear(device_id)
    
    async def find_button_yolo(self, device_id: str, page_type: str, button_name: str,
                              conf_threshold: float = 0.5) -> Optional[Tuple[int, int]]:
        """使用YOLO查找指定按钮的坐标"""
        if not HAS_YOLO:
            return None
        
        try:
            image = await self._get_screenshot(device_id)
            if not image:
                return None
            
            model = self._load_yolo_model(page_type)
            
            if not model:
                return None
            
            results = model.predict(image, conf=conf_threshold, verbose=False)
            
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    cls = int(box.cls[0])
                    class_name = result.names[cls]
                    conf = float(box.conf[0])
                    
                    if button_name in class_name or class_name in button_name:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        
                        return (center_x, center_y)
            
            return None
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None

    async def close_popup(self, device_id: str, timeout: float = 15.0, known_popup_type: str = None, max_attempts: int = 3) -> bool:
        """自动关闭弹窗（带超时保护和重试机制）"""
        import asyncio
        
        try:
            return await asyncio.wait_for(
                self._close_popup_impl(device_id, known_popup_type, max_attempts),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            self._log(f"[弹窗处理] ✗ 关闭弹窗超时（{timeout}秒）")
            return False
    
    async def _close_popup_impl(self, device_id: str, known_popup_type: str = None, max_attempts: int = 3) -> bool:
        """关闭弹窗的实际实现
        
        [2026-03-03] 修复原因：YOLO检测器始终返回UNKNOWN，不应该触发弹窗关闭逻辑
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
            self._log(f"[弹窗处理] 使用已知弹窗类型: {known_popup_type}")
            if known_popup_type in POPUP_BUTTONS:
                button_pos = POPUP_BUTTONS[known_popup_type]
        else:
            # [2026-03-03] 修复：YOLO检测器始终返回UNKNOWN，不应该触发弹窗关闭逻辑
            # 如果没有指定弹窗类型，直接返回True（不执行弹窗关闭逻辑）
            self._log(f"[弹窗处理] 未指定弹窗类型，跳过弹窗检测（避免YOLO检测器误判）")
            return True
        
        # 如果指定了弹窗类型，执行弹窗关闭逻辑
        if popup_type == "home_announcement":
            self._log(f"[弹窗处理] 首页公告弹窗，点击顶部区域关闭...")
            
            close_x, close_y = button_pos if button_pos else POPUP_BUTTONS['home_announcement']
            
            self._log(f"[弹窗处理] 最多尝试 {max_attempts} 次")
            for attempt in range(1, max_attempts + 1):
                self._log(f"[弹窗处理] 第 {attempt}/{max_attempts} 次点击 ({close_x}, {close_y})...")
                await self.adb.tap(device_id, close_x, close_y)
                await asyncio.sleep(1.0)
                
                result = await self.detect_page(device_id)
                
                if result.state == PageState.HOME:
                    self._log(f"[弹窗处理] ✓ 成功关闭首页公告弹窗，当前页面: {result.state.value}")
                    return True
                elif result.state == PageState.CATEGORY:
                    self._log(f"[弹窗处理] ✗ 点击误触进入分类页，点击首页按钮...")
                    await self.adb.tap(device_id, 90, 920)
                    await asyncio.sleep(0.5)
                    self.clear_cache(device_id)
                    result = await self.detect_page(device_id)
                    if result.state == PageState.HOME:
                        self._log(f"[弹窗处理] ✓ 已返回首页")
                        return True
                    continue
                elif result.state == PageState.POPUP:
                    self._log(f"[弹窗处理] 仍是弹窗，继续尝试...")
                    continue
                else:
                    self._log(f"[弹窗处理] ✗ 点击后页面状态异常: {result.state.value}，返回失败")
                    return False
            
            self._log(f"[弹窗处理] {max_attempts} 次点击都失败，尝试按返回键...")
            await self.adb.press_back(device_id)
            await asyncio.sleep(1.0)
            
            result = await self.detect_page(device_id)
            if result.state == PageState.HOME:
                self._log(f"[弹窗处理] ✓ 返回键成功关闭首页公告弹窗，当前页面: {result.state.value}")
                return True
            else:
                self._log(f"[弹窗处理] ✗ 无法关闭首页公告弹窗，当前页面: {result.state.value}")
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
                        self._log(f"[弹窗处理] 检测到签到奖励弹窗 (OCR确认)")
            except:
                pass
        
        if is_checkin_popup:
            self._log(f"[弹窗处理] 使用签到弹窗专用坐标...")
            for i, (x, y) in enumerate(CHECKIN_POPUP_CLOSE, 1):
                self._log(f"[弹窗处理] 尝试位置 {i}/3: ({x}, {y})")
                await self.adb.tap(device_id, x, y)
                await asyncio.sleep(2)
                
                result = await self.detect_page(device_id)
                if result.state != PageState.POPUP and result.state != PageState.CHECKIN_POPUP:
                    self._log(f"[弹窗处理] ✓ 成功关闭签到弹窗")
                    return True
            
            self._log(f"[弹窗处理] ⚠️ 签到弹窗专用坐标都失败，尝试其他方法...")
        
        if button_pos:
            await self.adb.tap(device_id, button_pos[0], button_pos[1])
            await asyncio.sleep(2)
            
            result = await self.detect_page(device_id)
            if result.state != PageState.POPUP:
                self._log(f"[弹窗处理] ✓ 成功关闭")
                return True
            else:
                self._log(f"[弹窗处理] ⚠️ 预设位置点击失败，仍是弹窗")
            
            if popup_type in ["unknown", "home_announcement", "user_agreement"]:
                self._log(f"[弹窗处理] 尝试其他预设位置...")
                alternative_positions = [
                    (270, 608),
                    (270, 620),
                    (270, 650),
                    (270, 550),
                ]
                
                for pos in alternative_positions:
                    await self.adb.tap(device_id, pos[0], pos[1])
                    await asyncio.sleep(1.5)
                    
                    result = await self.detect_page(device_id)
                    if result.state != PageState.POPUP:
                        self._log(f"[弹窗处理] ✓ 成功关闭（位置: {pos}）")
                        return True
                
                self._log(f"[弹窗处理] 所有位置都失败，尝试按返回键...")
                await self.adb.press_back(device_id)
                await asyncio.sleep(1.5)
                
                result = await self.detect_page(device_id)
                if result.state != PageState.POPUP:
                    self._log(f"[弹窗处理] ✓ 成功关闭（返回键）")
                    return True
                else:
                    self._log(f"[弹窗处理] ✗ 返回键也失败，弹窗无法关闭")
                    return False
        
        self._log(f"[弹窗处理] ✗ 无法关闭弹窗")
        return False
