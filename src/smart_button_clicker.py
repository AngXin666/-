"""
智能按钮点击器 - 集成学习器的通用按钮点击工具
Smart Button Clicker - Universal button clicking tool with learner integration
"""

import asyncio
from typing import Optional, Tuple, Callable, List
from io import BytesIO

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from .adb_bridge import ADBBridge
from .button_position_learner import ButtonPositionLearner


class SmartButtonClicker:
    """智能按钮点击器
    
    集成按钮位置学习器，提供统一的按钮点击接口
    支持多种检测方式的智能降级策略
    """
    
    def __init__(self, adb: ADBBridge, detector=None, ocr_pool=None):
        """初始化智能按钮点击器
        
        Args:
            adb: ADB桥接对象
            detector: 页面检测器（可选）
            ocr_pool: OCR线程池（可选）
        """
        self.adb = adb
        self.detector = detector
        self.ocr_pool = ocr_pool
    
    async def click_button(
        self,
        device_id: str,
        button_name: str,
        valid_range: Optional[Tuple[int, int, int, int]] = None,
        default_position: Optional[Tuple[int, int]] = None,
        yolo_detector: Optional[Callable] = None,
        ocr_detector: Optional[Callable] = None,
        cached_position: Optional[Tuple[int, int]] = None,
        log_callback: Optional[Callable] = None
    ) -> Tuple[bool, Optional[Tuple[int, int]]]:
        """智能点击按钮
        
        智能降级策略：
        1. YOLO检测 → 坐标合理性验证
        2. 如果不合理 → 使用学习器推荐坐标
        3. 如果学习器无数据 → 使用缓存坐标
        4. 如果缓存无效 → OCR识别 → 坐标合理性验证
        5. 如果OCR失败 → 使用默认坐标
        
        Args:
            device_id: 设备ID
            button_name: 按钮名称（用于学习器记录）
            valid_range: 合理坐标范围 (x_min, x_max, y_min, y_max)
            default_position: 默认坐标
            yolo_detector: YOLO检测函数（返回 (x, y, confidence)）
            ocr_detector: OCR检测函数（返回 (x, y)）
            cached_position: 缓存坐标
            log_callback: 日志回调函数
            
        Returns:
            (成功标志, 使用的坐标)
        """
        def log(msg: str):
            if log_callback:
                log_callback(msg)
        
        # 创建设备专属的学习器
        learner = ButtonPositionLearner(device_id=device_id)
        
        detected_position = None
        detection_confidence = 0.0
        
        # 步骤1: YOLO检测
        if yolo_detector:
            try:
                log(f"  [智能点击] 使用YOLO检测'{button_name}'按钮...")
                result = await yolo_detector()
                if result:
                    if len(result) == 3:
                        detected_position = (result[0], result[1])
                        detection_confidence = result[2]
                    else:
                        detected_position = result
                        detection_confidence = 1.0
                    log(f"  [智能点击] YOLO检测到: {detected_position} (置信度: {detection_confidence:.2%})")
            except Exception as e:
                log(f"  [智能点击] YOLO检测失败: {e}")
        
        # 步骤2: 坐标合理性验证
        if detected_position and valid_range:
            x, y = detected_position
            x_min, x_max, y_min, y_max = valid_range
            
            if x_min <= x <= x_max and y_min <= y <= y_max:
                log(f"  [智能点击] ✓ 坐标合理性验证通过")
                # 记录成功坐标
                learner.record_success(button_name, detected_position, detection_confidence)
                # 点击
                await self.adb.tap(device_id, x, y)
                return (True, detected_position)
            else:
                log(f"  [智能点击] ⚠️ 坐标不合理: {detected_position}，超出范围 {valid_range}")
                log(f"  [智能点击] 尝试使用学习器推荐坐标...")
        elif detected_position:
            # 没有合理范围限制，直接使用检测到的坐标
            log(f"  [智能点击] ✓ 使用YOLO检测坐标（无范围限制）")
            learner.record_success(button_name, detected_position, detection_confidence)
            await self.adb.tap(device_id, detected_position[0], detected_position[1])
            return (True, detected_position)
        
        # 步骤3: 使用学习器推荐坐标
        learner_position = learner.get_best_position(
            button_name,
            min_samples=5,
            prefer_device=True
        )
        
        if learner_position:
            log(f"  [智能点击] ✓ 学习器推荐坐标: {learner_position}")
            
            # 验证学习器推荐的坐标是否合理
            if valid_range:
                x, y = learner_position
                x_min, x_max, y_min, y_max = valid_range
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    log(f"  [智能点击] ✓ 学习器坐标验证通过")
                    await self.adb.tap(device_id, x, y)
                    return (True, learner_position)
                else:
                    log(f"  [智能点击] ⚠️ 学习器坐标不合理，继续降级...")
            else:
                # 没有范围限制，直接使用
                await self.adb.tap(device_id, learner_position[0], learner_position[1])
                return (True, learner_position)
        else:
            log(f"  [智能点击] 学习器无足够数据（需要至少5个样本）")
        
        # 步骤4: 使用缓存坐标
        if cached_position:
            log(f"  [智能点击] 使用缓存坐标: {cached_position}")
            
            # 验证缓存坐标是否合理
            if valid_range:
                x, y = cached_position
                x_min, x_max, y_min, y_max = valid_range
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    log(f"  [智能点击] ✓ 缓存坐标验证通过")
                    await self.adb.tap(device_id, x, y)
                    return (True, cached_position)
                else:
                    log(f"  [智能点击] ⚠️ 缓存坐标不合理，继续降级...")
            else:
                await self.adb.tap(device_id, cached_position[0], cached_position[1])
                return (True, cached_position)
        
        # 步骤5: OCR识别
        if ocr_detector:
            try:
                log(f"  [智能点击] 降级到OCR识别...")
                ocr_position = await ocr_detector()
                if ocr_position:
                    log(f"  [智能点击] OCR找到按钮: {ocr_position}")
                    
                    # 验证OCR坐标是否合理
                    if valid_range:
                        x, y = ocr_position
                        x_min, x_max, y_min, y_max = valid_range
                        if x_min <= x <= x_max and y_min <= y <= y_max:
                            log(f"  [智能点击] ✓ OCR坐标验证通过")
                            learner.record_success(button_name, ocr_position, 1.0)
                            await self.adb.tap(device_id, x, y)
                            return (True, ocr_position)
                        else:
                            log(f"  [智能点击] ⚠️ OCR坐标不合理，继续降级...")
                    else:
                        learner.record_success(button_name, ocr_position, 1.0)
                        await self.adb.tap(device_id, ocr_position[0], ocr_position[1])
                        return (True, ocr_position)
            except Exception as e:
                log(f"  [智能点击] OCR识别失败: {e}")
        
        # 步骤6: 使用默认坐标
        if default_position:
            log(f"  [智能点击] 使用默认坐标: {default_position}")
            await self.adb.tap(device_id, default_position[0], default_position[1])
            return (True, default_position)
        
        # 所有方法都失败
        log(f"  [智能点击] ❌ 无法找到'{button_name}'按钮")
        return (False, None)
    
    async def click_button_simple(
        self,
        device_id: str,
        button_name: str,
        page_name: str,
        element_class_name: str,
        valid_range: Optional[Tuple[int, int, int, int]] = None,
        default_position: Optional[Tuple[int, int]] = None,
        log_callback: Optional[Callable] = None
    ) -> Tuple[bool, Optional[Tuple[int, int]]]:
        """简化版按钮点击（自动使用整合检测器）
        
        Args:
            device_id: 设备ID
            button_name: 按钮名称（用于日志和学习器）
            page_name: 页面名称（用于YOLO检测）
            element_class_name: 元素类名（用于YOLO检测）
            valid_range: 合理坐标范围
            default_position: 默认坐标
            log_callback: 日志回调函数
            
        Returns:
            (成功标志, 使用的坐标)
        """
        def log(msg: str):
            if log_callback:
                log_callback(msg)
        
        # 定义YOLO检测函数
        async def yolo_detect():
            if not self.detector:
                return None
            try:
                button_pos = await self.detector.find_button_yolo(
                    device_id,
                    page_name,
                    element_class_name,
                    conf_threshold=0.5
                )
                if button_pos:
                    # 返回 (x, y, confidence)
                    # find_button_yolo 返回 (x, y)，我们假设置信度为0.8
                    return (button_pos[0], button_pos[1], 0.8)
                return None
            except Exception as e:
                log(f"  [YOLO检测] 失败: {e}")
                return None
        
        # 调用通用点击方法
        return await self.click_button(
            device_id=device_id,
            button_name=button_name,
            valid_range=valid_range,
            default_position=default_position,
            yolo_detector=yolo_detect,
            log_callback=log_callback
        )
