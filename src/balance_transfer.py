"""
余额转账模块 - 处理余额转账功能
Balance Transfer Module - Handle balance transfer functionality
"""

import asyncio
from typing import Optional, Dict, Union
from datetime import datetime

from .adb_bridge import ADBBridge
from .page_detector import PageState
from .ocr_image_processor import enhance_for_ocr
from .models.error_types import ErrorType
from .timeouts_config import TimeoutsConfig


class BalanceTransfer:
    """余额转账处理器"""
    
    # 坐标定义 (540x960 分辨率)
    BALANCE_BUTTON_FALLBACK = (91, 228)  # 余额按钮备用坐标（OCR失败时使用，基于实际测试）
    TRANSFER_BUTTON = (456, 89)  # 转赠按钮 - 钱包页面左上角
    AMOUNT_INPUT = (270, 205)  # 转账金额输入框 - 实测可输入
    ALL_TRANSFER_BUTTON = (250, 275)  # 全部转账按钮 - 实测有效(右侧位置)
    RECIPIENT_INPUT = (270, 381)  # 收款用户ID输入框 - 实测可输入
    SUBMIT_BUTTON = (271, 521)  # 提交申请按钮 - 实测有效
    CONFIRM_BUTTON = (271, 630)  # 确认提交按钮 - 已校准坐标
    
    def __init__(self, adb: ADBBridge, detector: 'PageDetectorIntegrated'):
        """初始化转账处理器
        
        Args:
            adb: ADB桥接对象
            detector: 页面检测器（YOLO识别器）
        """
        self.adb = adb
        
        # [2026-03-03] 修改：从 ModelManager 获取转账专用检测器（避免重复创建）
        # 保存YOLO检测器（用于元素检测、按钮点击等）
        self.detector = detector
        
        # 导入日志记录器和ModelManager
        import logging
        from .model_manager import ModelManager
        
        logger = logging.getLogger(__name__)
        model_manager = ModelManager.get_instance()
        
        # 获取转账专用检测器（MobileNetV3，仅用于页面分类）
        self.page_classifier = model_manager.get_transfer_detector()
        if not self.page_classifier:
            # 降级使用YOLO检测器
            self.page_classifier = detector
            logger.warning(f"⚠️ 转账专用模型未加载，降级使用YOLO检测器")
        
        # 检查是否有转账相关的YOLO模型
        self.has_wallet_yolo = self._check_yolo_model('钱包页')
        self.has_transfer_yolo = self._check_yolo_model('transfer')
        self.has_transfer_confirm_yolo = self._check_yolo_model('transfer_confirm')
        
        if self.has_wallet_yolo:
            logger.debug("  [转账] ✓ 钱包页YOLO模型已加载")
        if self.has_transfer_yolo:
            logger.debug("  [转账] ✓ 转账页YOLO模型已加载")
        if self.has_transfer_confirm_yolo:
            logger.debug("  [转账] ✓ 转账确认弹窗YOLO模型已加载")
    
    def _check_yolo_model(self, page_type: str) -> bool:
        """检查指定页面类型的YOLO模型是否存在
        
        Args:
            page_type: 页面类型
            
        Returns:
            bool: 模型是否存在
        """
        try:
            import json
            from pathlib import Path
            
            # [2026-03-12] 修改路径：YOLO注册表文件移动到models目录
            registry_path = Path("models/yolo_model_registry.json")
            if not registry_path.exists():
                return False
            
            with open(registry_path, 'r', encoding='utf-8') as f:
                registry = json.load(f)
            
            if page_type in registry:
                model_path = registry[page_type].get('model_path')
                if model_path and Path(model_path).exists():
                    return True
            
            return False
        except Exception:
            return False
    
    async def _find_balance_button_by_ocr(self, device_id: str, log_callback=None) -> Optional[tuple]:
        """使用OCR或YOLO查找余额按钮位置
        
        优先使用YOLO检测（更准确），如果失败则降级到OCR
        
        Args:
            device_id: 设备ID
            log_callback: 日志回调函数
            
        Returns:
            tuple: (x, y) 坐标，如果未找到返回None
        """
        def log(msg):
            if log_callback:
                log_callback(msg)
        
        # 优先使用YOLO检测余额数字
        if self.has_wallet_yolo:
            try:
                log("  [转账] 使用YOLO检测余额按钮...")
                buttons = await self.detector.detect_buttons_yolo(device_id, "钱包页")
                
                if buttons:
                    # 查找"余额数字"按钮
                    for btn in buttons:
                        if btn.class_name == '余额数字':
                            center_x, center_y = btn.center
                            log(f"  [YOLO] ✓ 检测到余额数字，位置: ({center_x}, {center_y}), 置信度: {btn.confidence:.2f}")
                            return (center_x, center_y)
                
                log("  [YOLO] 未检测到余额数字，降级到OCR...")
            except Exception as e:
                log(f"  [YOLO] YOLO检测失败: {e}，降级到OCR...")
        
        # 降级到OCR检测
        try:
            from .screen_capture import ScreenCapture
            from .ocr_thread_pool import OCRThreadPool
            from PIL import Image
            import cv2
            import re
            
            # 截图
            screen_capture = ScreenCapture(self.adb)
            screenshot = await screen_capture.capture(device_id)
            screenshot_pil = Image.fromarray(cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB))
            
            # 图像预处理以提高OCR识别准确率
            # 1. 转换为灰度图 - 减少颜色干扰，让OCR更专注于文字形状
            gray_screenshot = screenshot_pil.convert('L')
            
            # 2. 增强对比度2倍 - 提高文字边缘清晰度，特别是灰色背景上的文字
            enhanced_screenshot = enhance_for_ocr(gray_screenshot)
            
            # OCR识别（使用全局单例）
            ocr_pool = OCRThreadPool()
            ocr_result = await ocr_pool.recognize(enhanced_screenshot)
            
            if not ocr_result or not ocr_result.texts:
                return None
            
            # 策略1（最稳健）: 通过"余额"文字定位，然后找最近的数字
            # 这个方法最可靠，因为"余额"文字的位置相对固定
            balance_label_index = None
            for i, text in enumerate(ocr_result.texts):
                if "余额" in text:
                    balance_label_index = i
                    if log:
                        log(f"  [OCR策略1] 找到'余额'标签在索引 {i}")
                    break
            
            if balance_label_index is not None:
                # 在"余额"前后查找数字（优先前面，因为通常数字在上方）
                search_range = list(range(max(0, balance_label_index - 5), balance_label_index))
                search_range.extend(range(balance_label_index + 1, min(len(ocr_result.texts), balance_label_index + 3)))
                
                for j in search_range:
                    match = re.search(r'^(\d+\.?\d*)$', ocr_result.texts[j].strip())
                    if match:
                        try:
                            balance = float(match.group(1))
                            if 0 <= balance <= 10000:
                                if ocr_result.boxes is not None and j < len(ocr_result.boxes):
                                    box = ocr_result.boxes[j]
                                    x_coords = [p[0] for p in box]
                                    y_coords = [p[1] for p in box]
                                    center_x = int(sum(x_coords) / 4)
                                    center_y = int(sum(y_coords) / 4)
                                    
                                    if log:
                                        log(f"  [OCR策略1] ✓ 通过'余额'标签找到数字 {balance} 在索引 {j} ({center_x}, {center_y})")
                                    
                                    return (center_x, center_y)
                        except ValueError:
                            pass
            
            # 策略2（次稳健）: 通过ID位置计算余额位置
            # ID和余额的相对位置关系固定，即使OCR顺序变化也能找到
            for i, text in enumerate(ocr_result.texts):
                id_text = text.strip()
                # 检查是否是ID（6-7位纯数字）
                if re.match(r'^\d{6,7}$', id_text):
                    if ocr_result.boxes is not None and i < len(ocr_result.boxes):
                        id_box = ocr_result.boxes[i]
                        id_x_coords = [p[0] for p in id_box]
                        id_y_coords = [p[1] for p in id_box]
                        id_center_x = int(sum(id_x_coords) / 4)
                        id_center_y = int(sum(id_y_coords) / 4)
                        
                        # 余额通常在ID下方约60-80像素
                        balance_x = id_center_x
                        balance_y = id_center_y + 70
                        
                        if log:
                            log(f"  [OCR策略2] ✓ 通过ID位置 ({id_center_x}, {id_center_y}) 计算余额位置 ({balance_x}, {balance_y})")
                        
                        return (balance_x, balance_y)
            
            # 策略3（备选）: 直接使用索引3的位置
            if len(ocr_result.texts) > 3:
                text = ocr_result.texts[3].strip()
                match = re.search(r'^(\d+\.?\d*)$', text)
                if match:
                    try:
                        balance = float(match.group(1))
                        if 0 <= balance <= 10000:
                            if ocr_result.boxes is not None and 3 < len(ocr_result.boxes):
                                box = ocr_result.boxes[3]
                                x_coords = [p[0] for p in box]
                                y_coords = [p[1] for p in box]
                                center_x = int(sum(x_coords) / 4)
                                center_y = int(sum(y_coords) / 4)
                                
                                if log:
                                    log(f"  [OCR策略3] 在索引3找到余额数字 {balance} 在 ({center_x}, {center_y})")
                                
                                return (center_x, center_y)
                    except ValueError:
                        pass
            
            # 策略4（兜底）: 遍历所有文本查找第一个合理的数值
            for i, text in enumerate(ocr_result.texts):
                match = re.search(r'^(\d+\.?\d*)$', text.strip())
                if match:
                    try:
                        balance = float(match.group(1))
                        if 0 <= balance <= 10000:
                            if ocr_result.boxes is not None and i < len(ocr_result.boxes):
                                box = ocr_result.boxes[i]
                                x_coords = [p[0] for p in box]
                                y_coords = [p[1] for p in box]
                                center_x = int(sum(x_coords) / 4)
                                center_y = int(sum(y_coords) / 4)
                                
                                if log:
                                    log(f"  [OCR策略4] 找到数字 {balance} 在索引 {i} ({center_x}, {center_y})（遍历查找）")
                                
                                return (center_x, center_y)
                    except ValueError:
                        pass
            
            return None
            
        except Exception as e:
            if log:
                log(f"  [OCR] 查找余额按钮失败: {e}")
            return None
    
    async def _find_transfer_button_by_ocr(self, device_id: str, log_callback=None) -> Optional[tuple]:
        """使用OCR查找转赠按钮位置
        
        在钱包页面识别"转赠"文字，返回其中心坐标
        
        Args:
            device_id: 设备ID
            log_callback: 日志回调函数
            
        Returns:
            tuple: (x, y) 坐标，如果未找到返回None
        """
        def log(msg):
            if log_callback:
                log_callback(msg)
        
        try:
            from .screen_capture import ScreenCapture
            from .ocr_thread_pool import OCRThreadPool
            from PIL import Image
            import cv2
            
            log("  [转账] 使用OCR识别转赠按钮...")
            
            # 截图
            screen_capture = ScreenCapture(self.adb)
            screenshot = await screen_capture.capture(device_id)
            screenshot_pil = Image.fromarray(cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB))
            
            # 图像预处理
            gray_screenshot = screenshot_pil.convert('L')
            enhanced_screenshot = enhance_for_ocr(gray_screenshot)
            
            # OCR识别
            ocr_pool = OCRThreadPool()
            ocr_result = await ocr_pool.recognize(enhanced_screenshot)
            
            # [2026-03-05] 修复数组比较错误：检查 texts 是否为 None 或长度为 0
            if not ocr_result or ocr_result.texts is None or len(ocr_result.texts) == 0:
                log("  [OCR] 未识别到任何文字")
                return None
            
            # 查找"转赠"文字
            for i, text in enumerate(ocr_result.texts):
                if "转赠" in text:
                    if ocr_result.boxes is not None and i < len(ocr_result.boxes):
                        box = ocr_result.boxes[i]
                        x_coords = [p[0] for p in box]
                        y_coords = [p[1] for p in box]
                        center_x = int(sum(x_coords) / 4)
                        center_y = int(sum(y_coords) / 4)
                        
                        log(f"  [OCR] ✓ 找到'转赠'按钮在索引 {i}，位置: ({center_x}, {center_y})")
                        return (center_x, center_y)
            
            log("  [OCR] 未找到'转赠'文字")
            return None
            
        except Exception as e:
            if log:
                log(f"  [OCR] 查找转赠按钮失败: {e}")
            return None
    
    async def _find_amount_input_by_ocr(self, device_id: str, log_callback=None) -> Optional[tuple]:
        """使用OCR查找金额输入框位置（备选方案）
        
        通过识别"?"或"¥"符号，计算输入框位置
        
        Args:
            device_id: 设备ID
            log_callback: 日志回调函数
            
        Returns:
            tuple: (x, y) 坐标，如果未找到返回None
        """
        def log(msg):
            if log_callback:
                log_callback(msg)
        
        try:
            from .screen_capture import ScreenCapture
            from .ocr_thread_pool import OCRThreadPool
            from PIL import Image
            import cv2
            
            # 截图
            screen_capture = ScreenCapture(self.adb)
            screenshot = await screen_capture.capture(device_id)
            screenshot_pil = Image.fromarray(cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB))
            
            # 图像预处理以提高OCR识别准确率
            # 1. 转换为灰度图 - 减少颜色干扰，让OCR更专注于文字形状
            gray_screenshot = screenshot_pil.convert('L')
            
            # 2. 增强对比度2倍 - 提高文字边缘清晰度，特别是灰色背景上的文字
            enhanced_screenshot = enhance_for_ocr(gray_screenshot)
            
            # OCR识别（使用全局单例）
            ocr_pool = OCRThreadPool()
            ocr_result = await ocr_pool.recognize(enhanced_screenshot)
            
            # [2026-03-05] 修复数组比较错误：检查 texts 是否为 None 或长度为 0
            if not ocr_result or ocr_result.texts is None or len(ocr_result.texts) == 0:
                return None
            
            # 查找"?"或"¥"符号
            for i, text in enumerate(ocr_result.texts):
                if "?" in text or "¥" in text or "元" in text:
                    if ocr_result.boxes is not None and i < len(ocr_result.boxes):
                        box = ocr_result.boxes[i]
                        x_coords = [p[0] for p in box]
                        y_coords = [p[1] for p in box]
                        symbol_x = int(sum(x_coords) / 4)
                        symbol_y = int(sum(y_coords) / 4)
                        
                        # 输入框在符号右侧,使用屏幕中间位置
                        input_x = 270
                        input_y = symbol_y
                        
                        if log:
                            log(f"  [OCR] 找到金额符号 '{text}' 在 ({symbol_x}, {symbol_y})")
                            log(f"  [OCR] 计算输入框位置: ({input_x}, {input_y})")
                        
                        return (input_x, input_y)
            
            return None
            
        except Exception as e:
            if log:
                log(f"  [OCR] 查找金额输入框失败: {e}")
            return None
    
    async def input_amount(self, device_id: str, amount: str, 
                          use_ocr: bool = False, verify: bool = True, 
                          log_callback=None) -> bool:
        """输入转账金额
        
        Args:
            device_id: 设备ID
            amount: 转账金额（字符串格式，如"10.50"）
            use_ocr: 是否使用OCR动态查找输入框（备选方案）
            verify: 是否验证输入是否成功
            log_callback: 日志回调函数
            
        Returns:
            bool: 是否成功
        """
        def log(msg):
            if log_callback:
                log_callback(msg)
            else:
                print(msg)
        
        try:
            # 确定输入框坐标
            if use_ocr:
                log("  [转账] 使用OCR查找金额输入框...")
                coords = await self._find_amount_input_by_ocr(device_id, log_callback)
                if not coords:
                    log("  [转账] OCR查找失败，使用默认坐标")
                    coords = self.AMOUNT_INPUT
            else:
                coords = self.AMOUNT_INPUT
            
            # 点击输入框
            log(f"  [转账] 点击金额输入框 ({coords[0]}, {coords[1]})...")
            await self.adb.tap(device_id, coords[0], coords[1])
            await asyncio.sleep(TimeoutsConfig.WAIT_SHORT)
            
            # 双击清空（如果有内容）
            await self.adb.tap(device_id, coords[0], coords[1])
            await asyncio.sleep(0.3)
            await self.adb.shell(device_id, "input keyevent KEYCODE_DEL")
            await asyncio.sleep(0.3)
            
            # 输入金额
            log(f"  [转账] 输入金额: {amount}...")
            await self.adb.input_text(device_id, amount)
            await asyncio.sleep(TimeoutsConfig.TRANSFER_INPUT_WAIT)
            
            # 验证输入是否成功
            if verify:
                log("  [转账] 验证输入...")
                success = await self._verify_amount_input(device_id, amount, log_callback)
                if success:
                    log("  [转账] ✓ 金额输入成功")
                else:
                    log("  [转账] ⚠️ 金额输入验证失败")
                return success
            
            return True
            
        except Exception as e:
            log(f"  [转账] 输入金额失败: {e}")
            return False
    
    async def _verify_amount_input(self, device_id: str, expected_amount: str, 
                                   log_callback=None) -> bool:
        """验证金额是否输入成功
        
        Args:
            device_id: 设备ID
            expected_amount: 期望的金额
            log_callback: 日志回调函数
            
        Returns:
            bool: 是否验证成功
        """
        def log(msg):
            if log_callback:
                log_callback(msg)
        
        try:
            from .screen_capture import ScreenCapture
            from .ocr_thread_pool import OCRThreadPool
            from PIL import Image
            import cv2
            
            # 截图
            screen_capture = ScreenCapture(self.adb)
            screenshot = await screen_capture.capture(device_id)
            screenshot_pil = Image.fromarray(cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB))
            
            # 图像预处理以提高OCR识别准确率
            # 1. 转换为灰度图 - 减少颜色干扰，让OCR更专注于文字形状
            gray_screenshot = screenshot_pil.convert('L')
            
            # 2. 增强对比度2倍 - 提高文字边缘清晰度，特别是灰色背景上的文字
            enhanced_screenshot = enhance_for_ocr(gray_screenshot)
            
            # OCR识别（使用全局单例）
            ocr_pool = OCRThreadPool()
            ocr_result = await ocr_pool.recognize(enhanced_screenshot)
            
            # [2026-03-05] 修复数组比较错误：检查 texts 是否为 None 或长度为 0
            if not ocr_result or ocr_result.texts is None or len(ocr_result.texts) == 0:
                return False
            
            # 提取期望金额中的数字部分
            # 例如: "10.50" -> ["10", "50"]
            expected_parts = expected_amount.replace(".", "").replace(",", "")
            
            # 在识别到的文字中查找金额
            for text in ocr_result.texts:
                # 检查是否包含期望的金额
                # 可能的格式: "¥10.50", "10.50", "￥10.50"
                if expected_amount in text:
                    if log:
                        log(f"  [验证] 找到完整金额: '{text}'")
                    return True
                
                # 检查是否包含主要数字部分
                # 例如: expected_amount="10.50", 检查是否包含"10"和"50"
                if len(expected_parts) >= 2:
                    # 至少包含主要数字
                    main_part = expected_amount.split(".")[0] if "." in expected_amount else expected_amount
                    if main_part in text and any(c.isdigit() for c in text):
                        if log:
                            log(f"  [验证] 找到金额数字: '{text}'")
                        return True
            
            if log:
                log(f"  [验证] 未找到金额 '{expected_amount}'")
                log(f"  [验证] 识别到的文字: {ocr_result.texts[:5]}")
            
            return False
            
        except Exception as e:
            if log:
                log(f"  [验证] 验证失败: {e}")
            return False
    
    async def parse_confirm_dialog(self, device_id: str, log_callback=None) -> Optional[Dict]:
        """解析确认弹窗信息
        
        提取弹窗中的收款人ID、姓名和转账金额
        
        Args:
            device_id: 设备ID
            log_callback: 日志回调函数
            
        Returns:
            dict: 弹窗信息
                - recipient_id: str, 收款人ID
                - recipient_name: str, 收款人姓名
                - amount: str, 转账金额
            如果解析失败返回None
        """
        def log(msg):
            if log_callback:
                log_callback(msg)
        
        try:
            from .screen_capture import ScreenCapture
            from .ocr_thread_pool import OCRThreadPool
            from PIL import Image
            import cv2
            import re
            
            # 截图
            screen_capture = ScreenCapture(self.adb)
            screenshot = await screen_capture.capture(device_id)
            screenshot_pil = Image.fromarray(cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB))
            
            # 图像预处理以提高OCR识别准确率
            # 1. 转换为灰度图 - 减少颜色干扰，让OCR更专注于文字形状
            gray_screenshot = screenshot_pil.convert('L')
            
            # 2. 增强对比度2倍 - 提高文字边缘清晰度，特别是灰色背景上的文字
            enhanced_screenshot = enhance_for_ocr(gray_screenshot)
            
            # OCR识别（使用全局单例）
            ocr_pool = OCRThreadPool()
            ocr_result = await ocr_pool.recognize(enhanced_screenshot)
            
            # [2026-03-05] 修复数组比较错误：检查 texts 是否为 None 或长度为 0
            if not ocr_result or ocr_result.texts is None or len(ocr_result.texts) == 0:
                return None
            
            result = {
                'recipient_id': None,
                'recipient_name': None,
                'amount': None
            }
            
            # 查找收款人信息
            # 格式: "转账至ID：1803229-丁正清"
            for text in ocr_result.texts:
                if "转账至ID" in text or "转账至" in text:
                    # 提取ID和姓名
                    # 匹配模式: ID：数字-姓名
                    match = re.search(r'(\d+)[-—](.+)', text)
                    if match:
                        result['recipient_id'] = match.group(1)
                        result['recipient_name'] = match.group(2).strip()
                        if log:
                            log(f"  [弹窗] 收款人ID: {result['recipient_id']}")
                            log(f"  [弹窗] 收款人姓名: {result['recipient_name']}")
                    break
            
            # 查找转账金额
            # 格式: "￥127.66" 或 "¥127.66"
            for text in ocr_result.texts:
                if ("￥" in text or "¥" in text) and any(c.isdigit() for c in text):
                    # 提取数字部分
                    amount_match = re.search(r'[\d.]+', text)
                    if amount_match:
                        try:
                            # 转换为float类型
                            result['amount'] = float(amount_match.group(0))
                            if log:
                                log(f"  [弹窗] 转账金额: {result['amount']:.2f}")
                        except ValueError:
                            result['amount'] = None
                        break
            
            return result if result['recipient_id'] else None
            
        except Exception as e:
            if log:
                log(f"  [弹窗] 解析失败: {e}")
            return None
    
    async def verify_confirm_dialog(self, device_id: str, expected_recipient_id: str,
                                    log_callback=None) -> bool:
        """验证确认弹窗中的收款人ID是否正确
        
        Args:
            device_id: 设备ID
            expected_recipient_id: 期望的收款人ID
            log_callback: 日志回调函数
            
        Returns:
            bool: 是否匹配
        """
        def log(msg):
            if log_callback:
                log_callback(msg)
        
        try:
            log("  [验证] 解析确认弹窗...")
            dialog_info = await self.parse_confirm_dialog(device_id, log_callback)
            
            if not dialog_info:
                log("  [验证] ❌ 无法解析弹窗信息")
                return False
            
            if not dialog_info['recipient_id']:
                log("  [验证] ❌ 未找到收款人ID")
                return False
            
            # 对比ID
            if dialog_info['recipient_id'] == expected_recipient_id:
                log(f"  [验证] ✓ 收款人ID匹配: {dialog_info['recipient_id']}")
                if dialog_info['recipient_name']:
                    log(f"  [验证]   收款人姓名: {dialog_info['recipient_name']}")
                return True
            else:
                log(f"  [验证] ❌ 收款人ID不匹配!")
                log(f"  [验证]   期望: {expected_recipient_id}")
                log(f"  [验证]   实际: {dialog_info['recipient_id']}")
                return False
            
        except Exception as e:
            log(f"  [验证] 验证失败: {e}")
            return False
    
    async def _verify_page_by_ocr(self, device_id: str, keywords: list, 
                                  min_matches: int = 1, log_callback=None) -> bool:
        """使用OCR验证当前页面是否包含关键字
        
        Args:
            device_id: 设备ID
            keywords: 关键字列表
            min_matches: 最少匹配数量（默认1，即找到任意一个关键字即可）
            log_callback: 日志回调函数
            
        Returns:
            bool: 是否找到足够的关键字
        """
        def log(msg):
            if log_callback:
                log_callback(msg)
        
        try:
            from .screen_capture import ScreenCapture
            from .ocr_thread_pool import OCRThreadPool
            from PIL import Image
            import cv2
            
            # 截图
            screen_capture = ScreenCapture(self.adb)
            screenshot = await screen_capture.capture(device_id)
            screenshot_pil = Image.fromarray(cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB))
            
            # 图像预处理以提高OCR识别准确率
            # 1. 转换为灰度图 - 减少颜色干扰，让OCR更专注于文字形状
            gray_screenshot = screenshot_pil.convert('L')
            
            # 2. 增强对比度2倍 - 提高文字边缘清晰度，特别是灰色背景上的文字
            enhanced_screenshot = enhance_for_ocr(gray_screenshot)
            
            # OCR识别（使用全局单例，不要关闭）
            ocr_pool = OCRThreadPool()
            ocr_result = await ocr_pool.recognize(enhanced_screenshot)
            
            # [2026-03-05] 修复数组比较错误：检查 texts 是否为 None 或长度为 0
            if not ocr_result or ocr_result.texts is None or len(ocr_result.texts) == 0:
                log(f"  [验证] ⚠️ OCR未识别到任何文字")
                return False
            
            # 输出识别到的所有文字（用于调试）
            log(f"  [验证] OCR识别到 {len(ocr_result.texts)} 个文本")
            if len(ocr_result.texts) <= 10:
                # 如果文本不多，全部输出
                for i, text in enumerate(ocr_result.texts):
                    log(f"  [验证]   [{i}] {text}")
            else:
                # 如果文本很多，只输出前10个
                for i, text in enumerate(ocr_result.texts[:10]):
                    log(f"  [验证]   [{i}] {text}")
                log(f"  [验证]   ... 还有 {len(ocr_result.texts) - 10} 个文本")
            
            # 查找关键字（去重）
            found_keywords = []
            for text in ocr_result.texts:
                for keyword in keywords:
                    if keyword in text and keyword not in found_keywords:
                        found_keywords.append(keyword)
            
            # 检查是否达到最少匹配数量
            if len(found_keywords) >= min_matches:
                log(f"  [验证] ✓ 找到 {len(found_keywords)}/{len(keywords)} 个关键字: {', '.join(found_keywords)}")
                return True
            else:
                log(f"  [验证] ❌ 只找到 {len(found_keywords)}/{len(keywords)} 个关键字: {', '.join(found_keywords) if found_keywords else '无'}")
                log(f"  [验证] 需要至少 {min_matches} 个关键字")
                return False
                
        except Exception as e:
            log(f"  [验证] OCR验证异常: {e}")
            return False
    
    async def transfer_balance(self, device_id: str, recipient_id: str, 
                               initial_balance: Optional[float] = None,
                               log_callback=None, transfer_chain: list = None,
                               step_number: int = 1, gui_logger=None) -> Dict[str, any]:
        """执行余额转账（使用转账专用模型进行页面检测）
        
        Args:
            device_id: 设备ID
            recipient_id: 收款用户ID
            initial_balance: 转账前的余额（用于验证转账是否成功）
            log_callback: 日志回调函数（可选）
            transfer_chain: 转账链条（用于防止循环转账）
            step_number: 步骤编号（用于简洁日志）
            gui_logger: GUI日志记录器（用于简洁日志）
            
        Returns:
            dict: 转账结果
                - success: bool, 是否成功
                - message: str, 结果消息
                - amount: float, 转账金额（如果成功）
                - chain: list, 转账链条
        """
        # 创建简洁日志记录器
        from .concise_logger import ConciseLogger
        import logging
        
        # 获取文件日志记录器
        file_logger = logging.getLogger(__name__)
        # [2026-03-12] 修复原因：恢复转账过程的GUI日志显示，让用户看到转账进度
        # 创建GUI日志记录器包装
        if gui_logger:
            gui_logger_obj = gui_logger
        elif log_callback:
            class GuiLogger:
                def __init__(self, callback):
                    self.callback = callback
                def info(self, msg):
                    self.callback(msg)
                def error(self, msg):
                    self.callback(msg)
            gui_logger_obj = GuiLogger(log_callback)
        else:
            gui_logger_obj = None
        
        concise = ConciseLogger("balance_transfer", gui_logger_obj, file_logger)
        
        def log(msg):
            if log_callback:
                log_callback(msg)
            else:
                print(msg)
        
        # 初始化转账链条
        if transfer_chain is None:
            transfer_chain = []
        
        result = {
            'success': False,
            'message': '',
            'amount': 0.0,
            'chain': transfer_chain,
            'error_type': None,  # 错误类型（ErrorType枚举）
            'recipient_id': recipient_id,  # 收款人ID
            'recipient_name': None  # 收款人姓名（从确认弹窗解析）
        }
        
        try:
            # 添加步骤日志
            concise.step(step_number, "转账")
            
            # 检查循环转账
            if recipient_id in transfer_chain:
                file_logger.warning(f"[转账] 检测到循环转账，停止转账链条")
                concise.error("检测到循环转账")
                result['message'] = "检测到循环转账"
                result['error_type'] = ErrorType.TRANSFER_FAILED  # 转账失败
                return result
            
            # 记录详细日志到文件
            file_logger.info(f"[转账] 开始执行转账流程（使用YOLO）")
            
            # 1. 进入钱包页面
            concise.action("检查转账条件")
            file_logger.info(f"[转账] 步骤1: 使用YOLO检测个人页面")
            from .page_detector import PageState
            
            # [2026-03-02] 修复：使用转账专用模型进行页面检测
            # 清除页面检测缓存
            if hasattr(self.page_classifier, 'clear_cache'):
                self.page_classifier.clear_cache(device_id)
            
            # 使用转账专用模型进行页面检测
            page_result = await self.page_classifier.detect_page(
                device_id, 
                use_cache=False
            )
            
            file_logger.info(f"[转账] 页面检测结果（转账专用模型）: {page_result.state.value if page_result else 'None'}, 置信度: {page_result.confidence if page_result else 'N/A'}")
            
            if not page_result or page_result.state != PageState.PROFILE_LOGGED:
                file_logger.warning(f"[转账] 当前不在个人页面（已登录），当前页面: {page_result.state.value if page_result else 'unknown'}")
                concise.error("不在个人页面")
                result['message'] = "不在个人页面"
                result['error_type'] = ErrorType.TRANSFER_FAILED
                return result
            
            file_logger.info(f"[转账] 当前在个人页面（已登录）")
            concise.action("满足条件，开始转账")
            
            # 使用智能按钮点击器点击余额按钮
            concise.action("进入钱包页面")
            file_logger.info("[转账] 点击余额按钮")
            
            # [2026-03-02] 修复：使用YOLO检测器进行元素检测
            # 转账专用模型只做页面分类，元素检测需要用YOLO检测器
            element_result = await self.detector.detect_page(
                device_id,
                use_cache=False,
                detect_elements=True
            )
            
            # [2026-02-22] 优化：直接使用element_result中的余额按钮元素
            # [2026-03-05] 修复数组比较错误：使用 is not None 和 len() 检查
            balance_element = None
            if element_result and element_result.elements is not None and len(element_result.elements) > 0:
                for element in element_result.elements:
                    if element.class_name == '余额数字':
                        balance_element = element
                        file_logger.info(f"[转账] YOLO检测到余额按钮: {element.center}, 置信度{element.confidence:.2f}")
                        break
            
            if balance_element:
                await self.adb.tap(device_id, balance_element.center[0], balance_element.center[1])
                file_logger.info(f"[转账] 成功点击余额按钮，位置: {balance_element.center}")
            else:
                # [2026-02-22] 降级方案：YOLO失败 → OCR检测 → 默认坐标
                file_logger.warning("[转账] YOLO未检测到余额按钮，尝试OCR检测...")
                balance_pos = await self._find_balance_button_by_ocr(device_id, log_callback=lambda msg: file_logger.info(msg))
                
                if balance_pos:
                    file_logger.info(f"[转账] OCR检测到余额按钮: {balance_pos}")
                    await self.adb.tap(device_id, balance_pos[0], balance_pos[1])
                else:
                    # 最终降级：使用默认坐标
                    file_logger.warning("[转账] OCR也未检测到余额按钮，使用默认坐标")
                    await self.adb.tap(device_id, self.BALANCE_BUTTON_FALLBACK[0], self.BALANCE_BUTTON_FALLBACK[1])
                    file_logger.info(f"[转账] 使用默认坐标: {self.BALANCE_BUTTON_FALLBACK}")
            
            # [2026-02-21] 增加等待时间，确保页面开始跳转
            await asyncio.sleep(0.5)
            
            # 2. 使用智能等待器等待钱包页面
            file_logger.info("[转账] 步骤2: 等待进入钱包页面...")
            from .performance.smart_waiter import wait_for_page
            
            # [2026-03-02] 修复：使用转账专用模型作为检测器
            page_result = await wait_for_page(
                device_id=device_id,
                detector=self.page_classifier,  # 使用转账专用模型
                expected_states=[PageState.WALLET],
                log_callback=lambda msg: file_logger.info(f"  [等待] {msg}")
            )
            
            if not page_result or page_result.state != PageState.WALLET:
                # [2026-02-21] 输出详细的失败信息
                current_state = page_result.state.value if page_result and page_result.state else 'unknown'
                current_confidence = page_result.confidence if page_result else 0.0
                current_details = page_result.details if page_result else 'N/A'
                
                file_logger.error(f"[转账] 未能进入钱包页面")
                file_logger.error(f"  当前页面状态: {current_state}")
                file_logger.error(f"  置信度: {current_confidence:.2%}")
                file_logger.error(f"  详细信息: {current_details}")
                file_logger.error(f"  期望页面: wallet")
                
                # 尝试手动检测一次，看看是什么页面
                file_logger.info("[转账] 尝试手动检测当前页面...")
                manual_result = await self.detector.detect_page(device_id, use_cache=False, detect_elements=True)
                if manual_result:
                    file_logger.info(f"  手动检测结果: {manual_result.state.value} (置信度: {manual_result.confidence:.2%})")
                    file_logger.info(f"  详细信息: {manual_result.details}")
                    # [2026-03-05] 修复数组比较错误：使用 is not None 和 len() 检查
                    if manual_result.elements is not None and len(manual_result.elements) > 0:
                        file_logger.info(f"  检测到 {len(manual_result.elements)} 个元素:")
                        for elem in manual_result.elements:
                            file_logger.info(f"    - {elem.class_name} (置信度: {elem.confidence:.2f}, 位置: {elem.center})")
                
                concise.error(f"未能进入钱包页面（当前: {current_state}）")
                result['message'] = f"未能进入钱包页面（当前: {current_state}）"
                result['error_type'] = ErrorType.TRANSFER_FAILED
                return result
            
            file_logger.info(f"[转账] ✓ 成功进入钱包页面")
            
            # 3. 点击转赠按钮（优先使用OCR识别，失败则使用固定坐标）
            concise.action("进入转账页面")
            file_logger.info("[转账] 步骤3: 点击转赠按钮")
            
            # 尝试使用OCR识别转赠按钮位置
            transfer_button_pos = await self._find_transfer_button_by_ocr(device_id, log_callback=lambda msg: file_logger.info(msg))
            
            if transfer_button_pos:
                file_logger.info(f"[转账] ✓ OCR识别到转赠按钮: {transfer_button_pos}")
                await self.adb.tap(device_id, transfer_button_pos[0], transfer_button_pos[1])
            else:
                file_logger.info(f"[转账] OCR未识别到转赠按钮，使用默认坐标: {self.TRANSFER_BUTTON}")
                await self.adb.tap(device_id, self.TRANSFER_BUTTON[0], self.TRANSFER_BUTTON[1])
            
            # 4. 使用智能等待器等待转账页面
            file_logger.info("[转账] 步骤4: 等待进入转账页面...")
            
            # [2026-03-02] 修复：使用转账专用模型作为检测器
            page_result = await wait_for_page(
                device_id=device_id,
                detector=self.page_classifier,  # 使用转账专用模型
                expected_states=[PageState.TRANSFER],
                log_callback=lambda msg: file_logger.info(f"  [等待] {msg}")
            )
            
            if not page_result or page_result.state != PageState.TRANSFER:
                file_logger.error(f"[转账] 未能进入转账页面，当前页面: {page_result.state.value if page_result else 'unknown'}")
                concise.error("未能进入转账页面")
                result['message'] = "未能进入转账页面"
                result['error_type'] = ErrorType.TRANSFER_FAILED
                return result
            
            file_logger.info(f"[转账] ✓ 成功进入转账页面")
            
            # 5. 检测转账页面元素（YOLO）
            file_logger.info("[转账] 步骤5: 检测转账页面元素...")
            # [2026-03-12] 修复原因：直接使用转账专用YOLO模型检测元素
            element_result = await self.detector.detect_elements_yolo(
                device_id, 
                model_key='transfer',  # 直接指定使用转账专用模型
                conf_threshold=0.25
            )
            
            # [2026-03-12] 修复原因：处理detect_elements_yolo返回的字典格式
            if not element_result:
                file_logger.error("[转账] 未检测到转账页面元素")
                # [2026-03-12] 添加详细调试日志：诊断YOLO检测失败的原因
                file_logger.error(f"[转账调试] element_result: {element_result}")
                concise.error("未检测到转账页面元素")
                result['message'] = "未检测到转账页面元素"
                result['error_type'] = ErrorType.TRANSFER_FAILED
                return result
            
            # 转换字典格式为元素列表，方便后续处理
            elements = []
            total_elements = 0
            for class_name, detections in element_result.items():
                total_elements += len(detections)
                for detection in detections:
                    # 创建临时对象来模拟PageElement
                    element = type('Element', (), {
                        'class_name': class_name,
                        'confidence': detection['confidence'],
                        'center': detection['center'],
                        'bbox': detection['bbox']
                    })()
                    elements.append(element)
            
            file_logger.info(f"[转账] ✓ 检测到 {total_elements} 个元素")
            for class_name, detections in element_result.items():
                for detection in detections:
                    file_logger.info(f"  - {class_name} (置信度: {detection['confidence']:.2f}, 位置: {detection['center']})")
            
            # 6. 查找并点击全部转账按钮
            concise.action("输入转账信息")
            file_logger.info("[转账] 步骤6: 点击全部转账按钮")
            
            all_transfer_element = None
            for element in elements:
                if element.class_name == '全部转账按钮':
                    all_transfer_element = element
                    break
            
            # [2026-03-12] 修复原因：添加详细调试日志，诊断全部转账按钮点击失败的原因
            if all_transfer_element:
                file_logger.info(f"[转账] ✓ YOLO检测到全部转账按钮: {all_transfer_element.center}")
                file_logger.info(f"[转账] 按钮置信度: {all_transfer_element.confidence:.2f}")
                file_logger.info(f"[转账] 按钮边界框: {all_transfer_element.bbox}")
                await self.adb.tap(device_id, all_transfer_element.center[0], all_transfer_element.center[1])
                file_logger.info(f"[转账] 已点击YOLO检测的全部转账按钮")
            else:
                file_logger.warning(f"[转账] 未检测到全部转账按钮，使用默认坐标: {self.ALL_TRANSFER_BUTTON}")
                file_logger.warning(f"[转账] 当前检测到的元素:")
                for elem in elements:
                    file_logger.warning(f"  - {elem.class_name} (置信度: {elem.confidence:.2f})")
                await self.adb.tap(device_id, self.ALL_TRANSFER_BUTTON[0], self.ALL_TRANSFER_BUTTON[1])
                file_logger.info(f"[转账] 已点击默认坐标的全部转账按钮")
            
            # 等待按钮点击生效
            await asyncio.sleep(TimeoutsConfig.WAIT_SHORT)
            
            # [2026-03-12] 添加验证：检查点击后页面是否有变化
            file_logger.info("[转账] 验证全部转账按钮点击是否生效...")
            verification_result = await self.detector.detect_elements_yolo(
                device_id, 
                model_key='transfer',  # 直接指定使用转账专用模型
                conf_threshold=0.25
            )
            
            if verification_result:
                total_after_click = sum(len(detections) for detections in verification_result.values())
                file_logger.info(f"[转账] 点击后检测到 {total_after_click} 个元素:")
                for class_name, detections in verification_result.items():
                    for detection in detections:
                        file_logger.info(f"  - {class_name} (置信度: {detection['confidence']:.2f})")
            else:
                file_logger.error("[转账] 点击后未检测到任何元素，可能点击失败")
                concise.error("全部转账按钮点击失败")
                result['message'] = "全部转账按钮点击失败"
                result['error_type'] = ErrorType.TRANSFER_FAILED
                return result
            
            # 7. 重新检测页面元素（点击全部转账后页面内容变化）
            # [2026-02-22] 添加详细日志：诊断YOLO元素检测失败问题
            file_logger.info("[转账] 步骤7: 重新检测转账页面元素...")
            file_logger.info(f"[转账] 调用detect_elements_yolo: device_id={device_id}, model_key=transfer")
            
            # [2026-03-12] 修复原因：直接使用转账专用YOLO模型检测元素
            element_result = await self.detector.detect_elements_yolo(
                device_id, 
                model_key='transfer',  # 直接指定使用转账专用模型
                conf_threshold=0.25
            )
            
            # [2026-02-22] 详细记录检测结果
            file_logger.info(f"[转账] detect_elements_yolo返回: element_result={element_result is not None}")
            if element_result:
                total_elements = sum(len(detections) for detections in element_result.values())
                file_logger.info(f"[转账] - 总元素数量: {total_elements}")
                for class_name, detections in element_result.items():
                    file_logger.info(f"[转账] - {class_name}: {len(detections)} 个")
                file_logger.info(f"[转账] - 使用模型: transfer")
            
            # [2026-03-12] 修复原因：处理detect_elements_yolo返回的字典格式
            if not element_result:
                file_logger.error("[转账] 重新检测失败，未检测到元素")
                # [2026-03-12] 添加详细调试日志：诊断重新检测失败的原因
                file_logger.error(f"[转账调试] 重新检测结果: {element_result}")
                concise.error("重新检测失败")
                result['message'] = "重新检测失败"
                result['error_type'] = ErrorType.TRANSFER_FAILED
                return result
            
            # 转换字典格式为元素列表，方便后续处理
            elements = []
            total_elements = 0
            for class_name, detections in element_result.items():
                total_elements += len(detections)
                for detection in detections:
                    # 创建临时对象来模拟PageElement
                    element = type('Element', (), {
                        'class_name': class_name,
                        'confidence': detection['confidence'],
                        'center': detection['center'],
                        'bbox': detection['bbox']
                    })()
                    elements.append(element)
            
            file_logger.info(f"[转账] ✓ 重新检测到 {total_elements} 个元素")
            for class_name, detections in element_result.items():
                for detection in detections:
                    file_logger.info(f"  - {class_name} (置信度: {detection['confidence']:.2f}, 位置: {detection['center']})")
            
            # 8. 查找并点击ID输入框
            concise.action(f"输入收款人ID: {recipient_id}")
            file_logger.info(f"[转账] 步骤8: 输入收款人ID")
            
            recipient_input_element = None
            for element in elements:
                if element.class_name == 'ID输入框':
                    recipient_input_element = element
                    break
            
            if recipient_input_element:
                file_logger.info(f"[转账] ✓ YOLO检测到ID输入框: {recipient_input_element.center}")
                await self.adb.tap(device_id, recipient_input_element.center[0], recipient_input_element.center[1])
            else:
                file_logger.warning(f"[转账] 未检测到ID输入框，使用默认坐标")
                await self.adb.tap(device_id, self.RECIPIENT_INPUT[0], self.RECIPIENT_INPUT[1])
            
            await asyncio.sleep(TimeoutsConfig.WAIT_SHORT)
            file_logger.info(f"[转账] 输入收款用户ID: {recipient_id}")
            
            # [2026-03-12] 修复原因：添加详细调试日志，诊断ID输入失败的原因
            # 先清空输入框（可能有默认内容）
            file_logger.info("[转账] 清空ID输入框...")
            await self.adb.clear_input(device_id)
            await asyncio.sleep(0.3)
            
            # 输入收款人ID
            file_logger.info(f"[转账] 开始输入收款人ID: {recipient_id}")
            input_success = await self.adb.input_text(device_id, recipient_id)
            file_logger.info(f"[转账] ID输入结果: {'成功' if input_success else '失败'}")
            
            if not input_success:
                file_logger.warning("[转账] ID输入失败，尝试逐字符输入...")
                # 降级方案：逐字符输入
                for char in recipient_id:
                    char_success = await self.adb.input_text(device_id, char)
                    if not char_success:
                        file_logger.error(f"[转账] 输入字符 '{char}' 失败")
                    await asyncio.sleep(0.1)
            
            await asyncio.sleep(TimeoutsConfig.TRANSFER_INPUT_WAIT)
            
            # 9. 查找并点击提交按钮
            file_logger.info("[转账] 步骤9: 点击提交按钮")
            
            submit_button_element = None
            for element in elements:
                if element.class_name == '提交按钮':
                    submit_button_element = element
                    break
            
            if submit_button_element:
                file_logger.info(f"[转账] ✓ YOLO检测到提交按钮: {submit_button_element.center}")
                await self.adb.tap(device_id, submit_button_element.center[0], submit_button_element.center[1])
            else:
                file_logger.warning(f"[转账] 未检测到提交按钮，使用默认坐标")
                await self.adb.tap(device_id, self.SUBMIT_BUTTON[0], self.SUBMIT_BUTTON[1])
            
            await asyncio.sleep(TimeoutsConfig.WAIT_MEDIUM)
            
            # 10. 验证确认弹窗并点击确认
            concise.action("确认转账信息")
            file_logger.info("[转账] 步骤10: 验证确认弹窗")
            
            # 解析弹窗信息
            dialog_info = await self.parse_confirm_dialog(device_id, log_callback)
            transfer_amount = None
            recipient_name = None
            if dialog_info and dialog_info.get('amount'):
                transfer_amount = dialog_info['amount']
                file_logger.info(f"[转账] 记录转账金额: {transfer_amount} 元")
                # 保存收款人姓名到返回值
                if dialog_info.get('recipient_name'):
                    result['recipient_name'] = dialog_info['recipient_name']
                    recipient_name = dialog_info['recipient_name']
                    file_logger.info(f"[转账] 记录收款人姓名: {result['recipient_name']}")
                
                # 显示确认信息
                if recipient_name:
                    concise.action(f"收款人: {recipient_name}")
                concise.action(f"金额: {transfer_amount:.2f}元")
            
            # 提交转账
            concise.action("提交转账")
            file_logger.info("[转账] 使用智能按钮点击器点击确认按钮")
            
            # 定义OCR检测函数
            async def detect_confirm_button_ocr():
                try:
                    from .screen_capture import ScreenCapture
                    from .ocr_thread_pool import OCRThreadPool
                    from PIL import Image
                    import cv2
                    
                    screen_capture = ScreenCapture(self.adb)
                    
                    # 截图
                    screenshot = await screen_capture.capture(device_id)
                    screenshot_pil = Image.fromarray(cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB))
                    
                    # 图像预处理
                    gray_screenshot = screenshot_pil.convert('L')
                    enhanced_screenshot = enhance_for_ocr(gray_screenshot)
                    
                    # OCR识别
                    ocr_pool = OCRThreadPool()
                    ocr_result = await ocr_pool.recognize(enhanced_screenshot)
                    
                    # [2026-03-05] 修复数组比较错误：检查 texts 是否为 None 并且长度大于 0
                    if ocr_result and ocr_result.texts is not None and len(ocr_result.texts) > 0:
                        file_logger.debug(f"[转账] OCR识别到 {len(ocr_result.texts)} 个文本")
                        
                        # 查找"确认提交"或"确认"文字
                        for i, text in enumerate(ocr_result.texts):
                            file_logger.debug(f"[转账] OCR文本[{i}]: {text}")
                            if "确认提交" in text or "确认" in text or "提交" in text:
                                if ocr_result.boxes is not None and i < len(ocr_result.boxes):
                                    box = ocr_result.boxes[i]
                                    x_coords = [p[0] for p in box]
                                    y_coords = [p[1] for p in box]
                                    center_x = int(sum(x_coords) / 4)
                                    center_y = int(sum(y_coords) / 4)
                                    file_logger.info(f"[转账] OCR检测到'{text}'按钮，位置: ({center_x}, {center_y})")
                                    return (center_x, center_y)
                    
                    return None
                except Exception as e:
                    file_logger.error(f"[转账] OCR检测失败: {e}", exc_info=True)
                    return None
            
            # 使用智能点击器
            # [2026-02-21] 修复：使用正确的页面类型名称 '转账确认弹窗'
            # 降级方案：YOLO → OCR → 默认坐标
            button_pos = await self.detector.find_button_yolo(
                device_id,
                '转账确认弹窗',  # 修复：使用配置文件中注册的正确名称
                '确认按钮',
                conf_threshold=0.5
            )
            
            if button_pos:
                file_logger.info(f"[转账] YOLO检测到确认按钮: {button_pos}")
            else:
                # 降级1：使用OCR识别确认按钮
                file_logger.warning("[转账] YOLO未检测到确认按钮，尝试OCR识别...")
                button_pos = await detect_confirm_button_ocr()
                
                if button_pos:
                    file_logger.info(f"[转账] OCR检测到确认按钮: {button_pos}")
                else:
                    # 降级2：使用默认坐标
                    file_logger.warning(f"[转账] OCR也未检测到确认按钮，使用默认坐标: {self.CONFIRM_BUTTON}")
                    button_pos = self.CONFIRM_BUTTON
            
            # 点击确认按钮
            await self.adb.tap(device_id, button_pos[0], button_pos[1])
            file_logger.info(f"[转账] 成功点击确认按钮，位置: {button_pos}")
            
            # [2026-03-02] 修复：使用智能等待器等待转账完成
            file_logger.info("[转账] 等待转账完成...")
            page_result = await wait_for_page(
                device_id=device_id,
                detector=self.page_classifier,  # 使用转账专用模型
                expected_states=[PageState.WALLET, PageState.PROFILE_LOGGED],
                log_callback=lambda msg: file_logger.info(f"  [等待] {msg}")
            )
            
            # 11. 验证转账结果并获取转账后余额
            file_logger.info("[转账] 步骤11: 验证转账结果")
            
            # [2026-03-02] 修复：如果不在钱包页或个人页，尝试按返回键返回
            if not page_result:
                file_logger.warning("[转账] 超时，手动检测当前页面...")
                page_result = await self.page_classifier.detect_page(device_id, use_cache=False)
            
            # 如果不在钱包页或个人页，尝试按返回键返回
            max_back_attempts = 3
            back_attempt = 0
            while page_result and page_result.state not in [PageState.WALLET, PageState.PROFILE_LOGGED] and back_attempt < max_back_attempts:
                file_logger.warning(f"[转账] 当前页面异常（{page_result.state.value}），按返回键尝试恢复")
                await self.adb.press_back(device_id)
                await asyncio.sleep(TimeoutsConfig.WAIT_MEDIUM)
                page_result = await self.page_classifier.detect_page(device_id, use_cache=False)
                back_attempt += 1
            
            if page_result and page_result.state == PageState.WALLET:
                file_logger.info(f"[转账] 检测到钱包页面，转账成功")
                
                # 获取转账后余额（用于计算转账金额）
                final_balance = None
                if initial_balance is not None:
                    file_logger.info(f"[转账] 返回个人页面获取转账后余额")
                    
                    # 按返回键回到个人页面
                    file_logger.info("[转账] 按返回键回到个人页面...")
                    await self.adb.press_back(device_id)
                    await asyncio.sleep(1.0)  # 等待页面跳转
                    
                    # [2026-03-02] 修复原因：使用循环检测 + OCR 降级，避免长时间等待
                    file_logger.info("[转账] 检测是否返回个人页...")
                    max_attempts = 3  # 最多尝试3次，每次1秒
                    page_result = None
                    
                    for attempt in range(max_attempts):
                        # 先用转账专用模型检测
                        page_result = await self.page_classifier.detect_page(device_id, use_cache=False)
                        
                        if page_result and page_result.state == PageState.PROFILE_LOGGED:
                            file_logger.info(f"[转账] ✓ 转账专用模型确认在个人页（第{attempt+1}次尝试）")
                            break
                        
                        # 如果专用模型识别失败，立即降级使用 OCR
                        file_logger.info(f"[转账] 转账专用模型识别失败（第{attempt+1}次），降级使用 OCR...")
                        from .page_detector import PageDetector
                        ocr_detector = PageDetector(self.adb)
                        ocr_result = await ocr_detector.detect_page(device_id, use_ocr=True, use_dl=False)
                        
                        if ocr_result and ocr_result.state == PageState.PROFILE_LOGGED:
                            file_logger.info(f"[转账] ✓ OCR 确认在个人页（第{attempt+1}次尝试）")
                            page_result = ocr_result
                            break
                        
                        if attempt < max_attempts - 1:
                            file_logger.info(f"[转账] 未检测到个人页，等待1秒后重试...")
                            await asyncio.sleep(1.0)
                    
                    # 检查最终结果
                    # 检查最终结果
                    if not page_result or page_result.state != PageState.PROFILE_LOGGED:
                        file_logger.warning(f"[转账] ⚠️ 3次尝试后仍未检测到个人页")
                        if page_result:
                            file_logger.warning(f"  当前页面: {page_result.state.value}")
                        file_logger.error(f"  ✗ 无法获取转账后余额，将使用其他策略计算转账金额")
                        final_balance = None
                    
                    if page_result and page_result.state == PageState.PROFILE_LOGGED:
                        file_logger.info(f"[转账] ✓ 已在个人页，开始获取转账后余额")
                        
                        # 检测页面元素（使用个人页面专用YOLO模型）
                        file_logger.info(f"[转账] 检测页面元素...")
                        element_result = await self.detector.detect_elements_yolo(
                            device_id, 
                            model_key='profile_detailed',  # 使用个人页面专用模型检测余额
                            conf_threshold=0.25
                        )
                        
                        # [2026-03-12] 修复原因：处理detect_elements_yolo返回的字典格式
                        if element_result and '余额数字' in element_result:
                            for detection in element_result['余额数字']:
                                file_logger.info(f"[转账] 检测到余额元素，位置: {detection['center']}, 置信度: {detection['confidence']:.2f}")
                                
                                # 截图并OCR识别余额区域
                                try:
                                    from .screen_capture import ScreenCapture
                                    from .ocr_thread_pool import OCRThreadPool
                                    from PIL import Image
                                    import cv2
                                    import re
                                    
                                    screen_capture = ScreenCapture(self.adb)
                                    screenshot = await screen_capture.capture(device_id)
                                    
                                    # 裁剪余额区域
                                    x1, y1, x2, y2 = detection['bbox']
                                    padding = 10
                                    x1 = max(0, x1 - padding)
                                    y1 = max(0, y1 - padding)
                                    x2 = min(screenshot.shape[1], x2 + padding)
                                    y2 = min(screenshot.shape[0], y2 + padding)
                                    
                                    balance_region = screenshot[y1:y2, x1:x2]
                                    balance_pil = Image.fromarray(cv2.cvtColor(balance_region, cv2.COLOR_BGR2RGB))
                                    
                                    # OCR识别
                                    ocr_pool = OCRThreadPool()
                                    ocr_result = await ocr_pool.recognize(balance_pil)
                                    
                                    # [2026-03-05] 修复数组比较错误：检查 texts 是否为 None 并且长度大于 0
                                    if ocr_result and ocr_result.texts is not None and len(ocr_result.texts) > 0:
                                        for text in ocr_result.texts:
                                            balance_match = re.search(r'[\d.]+', text)
                                            if balance_match:
                                                final_balance = float(balance_match.group(0))
                                                file_logger.info(f"[转账] OCR识别到转账后余额: {final_balance:.2f} 元")
                                                break
                                except Exception as e:
                                    file_logger.error(f"[转账] OCR识别余额失败: {e}", exc_info=True)
                                
                                break
                
                # ========== 转账金额获取策略（三级降级）==========
                calculated_amount = None
                amount_source = None
                
                # 策略1: 优先使用确认弹窗的金额（最准确）
                if transfer_amount and transfer_amount > 0:
                    calculated_amount = transfer_amount
                    amount_source = "确认弹窗OCR"
                    file_logger.info(f"[转账金额] 策略1: 使用确认弹窗金额 = {calculated_amount:.2f}元")
                
                # 策略2: 降级使用余额对比计算
                elif final_balance is not None and initial_balance is not None:
                    balance_change = initial_balance - final_balance
                    if balance_change > 0:
                        calculated_amount = balance_change
                        amount_source = "余额对比计算"
                        file_logger.info(f"[转账金额] 策略2: 使用余额对比 = {calculated_amount:.2f}元")
                        file_logger.info(f"  转账前: {initial_balance:.2f}, 转账后: {final_balance:.2f}")
                    else:
                        file_logger.warning(f"[转账金额] 策略2失败: 余额无变化或增加 ({initial_balance:.2f} -> {final_balance:.2f})")
                
                # 策略3: 最后降级使用配置推算（全部转账）
                if calculated_amount is None and initial_balance is not None:
                    calculated_amount = initial_balance
                    amount_source = "配置推算(全部转账)"
                    file_logger.warning(f"[转账金额] 策略3: 使用配置推算 = {calculated_amount:.2f}元")
                
                # 验证转账是否成功
                if calculated_amount and calculated_amount > 0:
                    file_logger.info(f"[转账] ✓ 转账成功")
                    file_logger.info(f"  转账金额: {calculated_amount:.2f}元 (来源: {amount_source})")
                    if final_balance is not None:
                        file_logger.info(f"  转账后余额: {final_balance:.2f}元")
                    
                    # 添加简洁日志
                    concise.success("转账成功")
                    
                    # 显示转账详细信息
                    if gui_logger:
                        gui_logger.info("=" * 60)
                        gui_logger.info(f"  → 转账金额: {calculated_amount:.2f}元")
                        if final_balance is not None:
                            gui_logger.info(f"  → 转账后余额: {final_balance:.2f}元")
                        if recipient_name:
                            gui_logger.info(f"  → 收款人: {recipient_name}")
                        else:
                            gui_logger.info(f"  → 收款人ID: {recipient_id}")
                    
                    # 添加到转账链条
                    transfer_chain.append(recipient_id)
                    
                    result['success'] = True
                    result['message'] = "转账成功"
                    result['chain'] = transfer_chain
                    result['amount'] = calculated_amount
                    return result
                else:
                    # 无法确定转账金额，转账失败
                    file_logger.error(f"[转账] 无法确定转账金额，转账失败")
                    concise.error("无法确定转账金额")
                    result['success'] = False
                    result['message'] = "转账失败：无法确定转账金额"
                    result['error_type'] = ErrorType.TRANSFER_FAILED
                    return result
            else:
                # 不在钱包页面，转账失败
                file_logger.error(f"[转账] 未检测到钱包页面，当前页面: {page_result.state.value if page_result else 'unknown'}")
                concise.error("转账失败")
                result['success'] = False
                result['message'] = "转账失败：未检测到钱包页面"
                result['error_type'] = ErrorType.TRANSFER_FAILED
                return result
            
        except Exception as e:
            file_logger.error(f"[转账] 转账失败: {e}", exc_info=True)
            concise.error("转账失败", e)
            result['message'] = f"转账失败: {e}"
            result['error_type'] = ErrorType.TRANSFER_FAILED
            return result

