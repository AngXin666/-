"""
个人信息读取模块 - 读取积分、抵扣券、总抽奖次数等
Profile Reader Module - Read points, vouchers, draw times, etc.
"""

import asyncio
import re
from typing import Optional, Dict, List
from io import BytesIO

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# 在导入 RapidOCR 之前设置日志级别
import logging
for logger_name in ['rapidocr', 'RapidOCR', 'ppocr', 'onnxruntime']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

try:
    from rapidocr import RapidOCR
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

from .adb_bridge import ADBBridge
from .account_cache import get_account_cache
from .ocr_image_processor import enhance_for_ocr
from .ocr_thread_pool import get_ocr_pool
from .logger import get_silent_logger


class ProfileReader:
    """个人信息读取器"""
    
    # 定义固定的像素区域（540x960分辨率）
    # 只识别数字区域，不包含标签文字
    # [2026-03-01] 删除优惠券：个人页已经没有优惠券了
    REGIONS = {
        'nickname': (100, 90, 300, 130),     # 昵称区域（顶部，ID上方）
        'balance': (30, 230, 150, 330),      # 余额数字区域（向左扩展20px以包含完整数字）
        'points': (180, 230, 260, 330),      # 积分数字区域
        'vouchers': (265, 230, 360, 330),    # 抵扣券数字区域（左边扩大10px以支持3位数）
    }
    
    def __init__(self, adb: ADBBridge, yolo_detector=None):
        """初始化读取器
        
        Args:
            adb: ADB桥接对象
            yolo_detector: YOLO按钮检测器或PageDetector对象（应该是从ModelManager获取的共享实例）
        """
        self.adb = adb
        
        # 从ModelManager获取OCR线程池
        from .model_manager import ModelManager
        model_manager = ModelManager.get_instance()
        self._ocr_pool = model_manager.get_ocr_thread_pool() if HAS_OCR else None
        
        # 初始化账号缓存
        self._cache = get_account_cache()
        
        # [2026-03-01] 修复原因：不再自己创建检测器，完全依赖传入的检测器
        # 初始化检测器（支持YOLO和旧的YOLO检测器）
        self._integrated_detector = None
        self._yolo_detector = None
        
        if yolo_detector:
            # [2026-03-11] 添加详细的类型检查（保存到文件）
            from .logger import get_logger
            debug_logger = get_logger()
            # [2026-03-12] 优化日志：移除CMD控制台的ProfileReader技术信息
            # [2026-03-11] 优化日志：移除控制台输出
            # [2026-03-11] 优化日志：移除控制台输出
            
            # 检查是否是YOLO检测器（PageDetectorIntegrated 或 PageDetectorDL）
            if hasattr(yolo_detector, 'detect_page'):
                # 进一步检查是否是PageDetectorIntegrated
                if yolo_detector.__class__.__name__ == 'PageDetectorIntegrated':
                    self._integrated_detector = yolo_detector
                    debug_logger.info(f"[ProfileReader] ✓ 使用PageDetectorIntegrated检测器")
                    # [2026-03-11] 优化日志：移除控制台输出
                else:
                    debug_logger.warning(f"[ProfileReader] ⚠️ 检测器有detect_page方法但不是PageDetectorIntegrated: {yolo_detector.__class__.__name__}")
                    debug_logger.info(f"[ProfileReader] ✓ 使用传入的检测器（可能不支持元素检测）")
                    # [2026-03-11] 优化日志：移除控制台输出
                    self._integrated_detector = yolo_detector
            # 检查是否是PageDetector对象，提取其中的_yolo_detector
            elif hasattr(yolo_detector, '_yolo_detector'):
                self._yolo_detector = yolo_detector._yolo_detector
                debug_logger.info(f"[ProfileReader] ✓ YOLO检测器已初始化（从PageDetector提取）")
            else:
                self._yolo_detector = yolo_detector
                debug_logger.info(f"[ProfileReader] ✓ YOLO检测器已初始化")
        else:
            pass  # [2026-03-11] 优化日志：检测器为None
        
        # 初始化静默日志记录器
        self._silent_log = get_silent_logger()
        
        # [2026-02-21] 删除学习器：移除 OCRRegionLearner
    
    async def get_profile_info(self, device_id: str) -> Dict[str, any]:
        """获取个人信息（积分、抵扣券、总抽奖次数等）
        
        Args:
            device_id: 设备ID
            
        Returns:
            dict: 个人信息
                - points: int, 积分
                - vouchers: int, 抵扣券数量
                - total_draw_times: int, 总抽奖次数
        """
        result = {
            'points': None,
            'vouchers': None,
            'total_draw_times': None
        }
        
        if not HAS_PIL or not self._ocr_pool:
            return result
        
        try:
            # 截图
            screenshot_data = await self.adb.screencap(device_id)
            if not screenshot_data:
                return result
            
            image = Image.open(BytesIO(screenshot_data))
            
            # 使用OCR图像预处理模块增强图像（灰度图 + 对比度增强2倍）
            enhanced_image = enhance_for_ocr(image)
            
            # 使用 OCR 线程池识别（异步，带超时）
            ocr_result = await self._ocr_pool.recognize(enhanced_image, timeout=10.0)
            
            # [2026-03-05] 修复数组比较错误：检查 texts 是否为 None 或长度为 0
            if not ocr_result or ocr_result.texts is None or len(ocr_result.texts) == 0:
                return result
            
            texts = ocr_result.texts
            
            # 解析积分
            points = self._parse_points(texts)
            if points is not None:
                result['points'] = points
            
            # 解析抵扣券
            vouchers = self._parse_vouchers(texts)
            if vouchers is not None:
                result['vouchers'] = vouchers
            
            # [2026-03-01] 删除优惠券解析：个人页已经没有优惠券了
            
            # 解析总抽奖次数
            draw_times = self._parse_draw_times(texts)
            if draw_times is not None:
                result['total_draw_times'] = draw_times
            
            return result
            
        except Exception as e:
            # [2026-03-11] 优化日志：删除CMD输出
            pass
            return result
    
    async def _get_dynamic_data_only(self, device_id: str) -> Dict[str, any]:
        """只获取动态数据（余额、积分、抵扣券、优惠券），跳过昵称和用户ID
        
        # [2026-03-01] 删除优惠券：个人页已经没有优惠券了
        用于缓存登录时，已经有昵称和用户ID，只需要获取动态数据
        
        Args:
            device_id: 设备ID
            
        Returns:
            dict: 动态数据
                - balance: float, 余额
                - points: int, 积分
                - vouchers: float, 抵扣券
        """
        result = {
            'balance': None,
            'points': None,
            'vouchers': None,
        }
        
        if not HAS_PIL or not self._ocr_pool:
            self._silent_log.log("  ! PIL 或 OCR 库未安装")
            return result
        
        try:
            self._silent_log.log(f"  [_get_dynamic_data_only] ========== 开始执行 ==========")
            
            # 截图
            screenshot_data = await self.adb.screencap(device_id)
            if not screenshot_data:
                self._silent_log.log("  ! 截图失败")
                return result
            
            image = Image.open(BytesIO(screenshot_data))
            
            # 优先使用YOLO（与get_balance相同的策略）
            use_yolo_fallback = True
            
            # [2026-02-22] 删除调试日志
            
            # [2026-02-21] 删除学习器：移除 OCRRegionLearner
            
            # [2026-03-11] 修复原因：添加详细调试日志，追踪检测器类型（保存到文件）
            from .logger import get_logger
            debug_logger = get_logger()
            debug_logger.info(f"  [DEBUG-_get_dynamic_data_only] self._integrated_detector类型: {type(self._integrated_detector)}")
            debug_logger.info(f"  [DEBUG-_get_dynamic_data_only] 检测器类名: {self._integrated_detector.__class__.__name__ if self._integrated_detector else 'None'}")
            # [2026-03-11] 优化日志：移除控制台输出
            # [2026-03-11] 优化日志：移除控制台输出
            if self._integrated_detector:
                debug_logger.info(f"  [DEBUG-_get_dynamic_data_only] 是否有find_button_yolo: {hasattr(self._integrated_detector, 'find_button_yolo')}")
                debug_logger.info(f"  [DEBUG-_get_dynamic_data_only] 是否有detect_page: {hasattr(self._integrated_detector, 'detect_page')}")
                # [2026-03-11] 优化日志：移除控制台输出
                # [2026-03-11] 优化日志：移除控制台输出
            
            if self._integrated_detector:
                detection_result = await self._integrated_detector.detect_page(
                    device_id, 
                    use_cache=False, 
                    detect_elements=True
                )
                
                # [2026-03-01] 修复：检查 elements 属性是否存在（PageDetectorDL 不支持元素检测）
                # [2026-03-05] 修复数组比较错误：使用 len() 检查而不是直接布尔判断
                has_elements = hasattr(detection_result, 'elements') and detection_result.elements is not None and len(detection_result.elements) > 0
                self._silent_log.log(f"  [YOLO] 检测到 {len(detection_result.elements) if has_elements else 0} 个元素")
                if has_elements:
                    for elem in detection_result.elements:
                        self._silent_log.log(f"  [YOLO] 元素: {elem.class_name}, 置信度: {elem.confidence:.2f}, 位置: {elem.bbox}")
                else:
                    self._silent_log.log(f"  [YOLO] ⚠️ 未检测到任何元素，将使用降级方案")
                
                if has_elements:
                    # 全屏OCR识别（只调用一次）
                    enhanced_image = enhance_for_ocr(image)
                    full_ocr_result = await self._ocr_pool.recognize(enhanced_image)
                    
                    self._silent_log.log(f"  [YOLO] OCR识别到 {len(full_ocr_result.texts) if full_ocr_result and full_ocr_result.texts else 0} 个文本")
                    
                    # 记录检测到的元素类型，用于后续判断
                    detected_elements = set()
                    if has_elements:
                        for element in detection_result.elements:
                            if '余额' in element.class_name:
                                detected_elements.add('balance')
                            elif '积分' in element.class_name:
                                detected_elements.add('points')
                            elif '抵扣' in element.class_name:
                                detected_elements.add('vouchers')
                            # [2026-03-01] 删除优惠券检测
                    
                    self._silent_log.log(f"  [YOLO] 检测到的元素类型: {detected_elements}")
                    
                    # [2026-03-05] 修复数组比较错误：检查 texts 和 boxes 是否为 None 并且长度大于 0
                    if (full_ocr_result and 
                        full_ocr_result.texts is not None and len(full_ocr_result.texts) > 0 and 
                        full_ocr_result.boxes is not None and len(full_ocr_result.boxes) > 0 and 
                        has_elements):
                        # 根据YOLO检测到的元素位置，从全屏OCR结果中匹配文本
                        for element in detection_result.elements:
                            x1, y1, x2, y2 = element.bbox
                            
                            self._silent_log.log(f"  [YOLO] 处理元素: {element.class_name}, 位置: ({x1}, {y1}, {x2}, {y2})")
                            
                            # 查找与元素位置重叠的OCR文本
                            matched_texts = []
                            matched_boxes = []  # 保存匹配的OCR文本框
                            for i, (text, box) in enumerate(zip(full_ocr_result.texts, full_ocr_result.boxes)):
                                # 计算OCR文本框的边界（使用所有点的最小/最大值）
                                x_coords = [p[0] for p in box]
                                y_coords = [p[1] for p in box]
                                ocr_x1 = min(x_coords)
                                ocr_y1 = min(y_coords)
                                ocr_x2 = max(x_coords)
                                ocr_y2 = max(y_coords)
                                ocr_center_x = (ocr_x1 + ocr_x2) / 2
                                ocr_center_y = (ocr_y1 + ocr_y2) / 2
                                
                                # 检查OCR文本框是否在YOLO元素框内
                                if x1 <= ocr_center_x <= x2 and y1 <= ocr_center_y <= y2:
                                    matched_texts.append(text)
                                    matched_boxes.append((ocr_x1, ocr_y1, ocr_x2, ocr_y2))
                                    self._silent_log.log(f"  [YOLO]   匹配到文本: '{text}' (中心点: {ocr_center_x:.0f}, {ocr_center_y:.0f})")
                            
                            self._silent_log.log(f"  [YOLO] 元素 {element.class_name} 匹配到 {len(matched_texts)} 个文本: {matched_texts}")
                            
                            if matched_texts:
                                # 合并所有匹配的文本
                                combined_text = ' '.join(matched_texts)
                                
                                # 查找所有数字（包括小数）
                                all_numbers = re.findall(r'(\d+\.?\d*)', combined_text)
                                
                                self._silent_log.log(f"  [YOLO] 从文本中提取到的所有数字: {all_numbers}")
                                
                                if all_numbers:
                                    # 转换为浮点数，选择第一个合理值（不使用max，避免误选其他区域的数字）
                                    valid_numbers = []
                                    valid_number_indices = []  # 保存有效数字在matched_texts中的索引
                                    for idx, num_str in enumerate(all_numbers):
                                        try:
                                            num = float(num_str)
                                            # 根据字段类型设置合理范围
                                            if '余额' in element.class_name and 0 <= num <= 100000:
                                                valid_numbers.append(num)
                                                # 找到这个数字在哪个matched_text中
                                                for text_idx, text in enumerate(matched_texts):
                                                    if num_str in text:
                                                        valid_number_indices.append(text_idx)
                                                        break
                                            elif '积分' in element.class_name and 0 <= num <= 100000:
                                                valid_numbers.append(num)
                                                for text_idx, text in enumerate(matched_texts):
                                                    if num_str in text:
                                                        valid_number_indices.append(text_idx)
                                                        break
                                            elif '抵扣' in element.class_name and 0 <= num <= 10000:
                                                valid_numbers.append(num)
                                                for text_idx, text in enumerate(matched_texts):
                                                    if num_str in text:
                                                        valid_number_indices.append(text_idx)
                                                        break
                                            elif '优惠' in element.class_name and 0 <= num <= 10000:
                                                valid_numbers.append(num)
                                                for text_idx, text in enumerate(matched_texts):
                                                    if num_str in text:
                                                        valid_number_indices.append(text_idx)
                                                        break
                                        except ValueError:
                                            continue
                                    
                                    self._silent_log.log(f"  [YOLO] 合理的候选值: {valid_numbers}")
                                    
                                    if valid_numbers:
                                        # 使用第一个合理值，而不是最大值
                                        value = valid_numbers[0]
                                        
                                        # 获取包含这个数字的OCR文本框位置
                                        ocr_box = None
                                        if valid_number_indices:
                                            text_idx = valid_number_indices[0]
                                            if text_idx < len(matched_boxes):
                                                ocr_x1, ocr_y1, ocr_x2, ocr_y2 = matched_boxes[text_idx]
                                                ocr_box = (int(ocr_x1), int(ocr_y1), int(ocr_x2 - ocr_x1), int(ocr_y2 - ocr_y1))
                                        
                                        # 添加详细调试日志
                                        self._silent_log.log(f"  [数据映射调试] 元素: {element.class_name}, 值: {value}, 匹配文本: {combined_text}, 所有候选值: {valid_numbers}")
                                        self._silent_log.log(f"  [数据映射调试] 当前状态 - balance: {result['balance']}, points: {result['points']}, vouchers: {result['vouchers']}")
                                        if ocr_box:
                                            self._silent_log.log(f"  [数据映射调试] OCR文本框位置: {ocr_box}")
                                        
                                        # 根据类别名称分配到对应字段
                                        if '余额' in element.class_name and result['balance'] is None:
                                            result['balance'] = value
                                            use_yolo_fallback = False
                                            self._silent_log.log(f"  ✓ 余额: {result['balance']:.2f} 元")
                                            # 不再记录OCR区域学习数据
                                            # if ocr_box:
                                            #     learner.record_success("profile_balance", ocr_box, element.confidence)
                                        elif '积分' in element.class_name and result['points'] is None:
                                            result['points'] = int(value)
                                            self._silent_log.log(f"  ✓ 积分: {result['points']}")
                                            # 不再记录OCR区域学习数据
                                            # if ocr_box:
                                            #     learner.record_success("profile_points", ocr_box, element.confidence)
                                        elif '抵扣' in element.class_name and result['vouchers'] is None:
                                            result['vouchers'] = value
                                            self._silent_log.log(f"  ✓ 抵扣券: {result['vouchers']}")
                                            # 不再记录OCR区域学习数据
                                            # if ocr_box:
                                            #     learner.record_success("profile_vouchers", ocr_box, element.confidence)
                                        # [2026-03-01] 删除优惠券处理
                                        else:
                                            self._silent_log.log(f"  ⚠️ 未匹配到任何字段！元素类别: {element.class_name}, 当前字段状态: balance={result['balance']}, points={result['points']}, vouchers={result['vouchers']}")
                                    else:
                                        self._silent_log.log(f"  ⚠️ 元素 {element.class_name} 没有合理的候选值")
                                else:
                                    self._silent_log.log(f"  ⚠️ 元素 {element.class_name} 的文本中没有提取到数字")
                            else:
                                self._silent_log.log(f"  ⚠️ 元素 {element.class_name} 没有匹配到任何OCR文本")
                                # 检测到元素但没有匹配到文本，设置为0避免进入区域OCR降级
                                if '积分' in element.class_name and result['points'] is None:
                                    result['points'] = 0
                                    self._silent_log.log(f"  ⚠️ 积分区域未识别到文本，设置为0")
                                # [2026-03-01] 删除优惠券处理
                    else:
                        self._silent_log.log(f"  ⚠️ OCR结果为空或没有位置信息")
                    
                    # 如果检测到了某个元素但OCR完全失败，也设置为0
                    if 'points' in detected_elements and result['points'] is None:
                        result['points'] = 0
                        self._silent_log.log(f"  ⚠️ 检测到积分元素但OCR失败，设置为0")
                    # [2026-03-01] 删除优惠券检测
            
            # 降级：使用旧的YOLO检测器
            if use_yolo_fallback and self._yolo_detector:
                detections = await self._yolo_detector.detect(
                    device_id, 
                    'balance',
                    conf_threshold=0.3
                )
                
                if detections:
                    for det in detections:
                        # 裁剪区域并OCR识别数字
                        x1, y1, x2, y2 = det.bbox
                        region = image.crop((x1, y1, x2, y2))
                        region_enhanced = enhance_for_ocr(region)
                        region_ocr = await self._ocr_pool.recognize(region_enhanced, timeout=3.0)
                        
                        # [2026-03-05] 修复数组比较错误：检查 texts 是否为 None 并且长度大于 0
                        if region_ocr and region_ocr.texts is not None and len(region_ocr.texts) > 0:
                            # 提取数字
                            for text in region_ocr.texts:
                                match = re.search(r'(\d+\.?\d*)', text.strip())
                                if match:
                                    try:
                                        value = float(match.group(1))
                                        
                                        # 根据类别名称分配到对应字段
                                        if '余额' in det.class_name and result['balance'] is None:
                                            result['balance'] = value
                                            self._silent_log.log(f"  ✓ 余额: {result['balance']:.2f} 元")
                                        elif '积分' in det.class_name and result['points'] is None:
                                            result['points'] = int(value)
                                            self._silent_log.log(f"  ✓ 积分: {result['points']}")
                                        elif '抵扣' in det.class_name and result['vouchers'] is None:
                                            result['vouchers'] = value
                                            self._silent_log.log(f"  ✓ 抵扣券: {result['vouchers']}")
                                        # [2026-03-01] 删除优惠券处理
                                        
                                        break
                                    except ValueError:
                                        pass
            
            # 最后降级：使用区域OCR
            # [2026-03-01] 删除优惠券检测
            if result['balance'] is None or result['points'] is None or result['vouchers'] is None:
                # [2026-02-22] 删除调试日志
                
                region_results = await self._recognize_regions(device_id, image)
                
                # [2026-02-22] 删除调试日志
                
                if result['balance'] is None and region_results.get('balance') is not None:
                    result['balance'] = region_results['balance']
                    self._silent_log.log(f"  [区域OCR] 余额: {result['balance']}")
                
                if result['points'] is None and region_results.get('points') is not None:
                    result['points'] = int(region_results['points'])
                    self._silent_log.log(f"  [区域OCR] 积分: {result['points']}")
                
                if result['vouchers'] is None and region_results.get('vouchers') is not None:
                    result['vouchers'] = region_results['vouchers']
                    self._silent_log.log(f"  [区域OCR] 抵扣券: {result['vouchers']}")
                
                # [2026-03-01] 删除优惠券区域OCR
            
            # [2026-02-22] 删除调试日志
            self._silent_log.log(f"  [_get_dynamic_data_only] ========== 执行完成 ==========")
            
            return result
            
        except Exception as e:
            self._silent_log.log(f"  ! 获取动态数据失败: {e}")
            return result
    
    async def get_identity_only(self, device_id: str, account: Optional[str] = None) -> Dict[str, any]:
        """只获取身份信息（昵称、用户ID），不读取余额等动态数据
        
        用于登录状态检查时，只需要确认身份，不需要读取余额
        
        注意：个人页上没有显示手机号，只能通过用户ID进行匹配
        
        Args:
            device_id: 设备ID
            account: 登录账号（可选），保留参数以兼容旧代码，但不使用
            
        Returns:
            dict: 身份信息
                - nickname: str, 昵称
                - user_id: str, 用户ID
        """
        result = {
            'nickname': None,
            'user_id': None
        }
        
        if not HAS_PIL or not self._ocr_pool:
            # [2026-03-11] 优化日志：删除CMD输出
            return result
        
        try:
            # 截图
            screenshot_data = await self.adb.screencap(device_id)
            if not screenshot_data:
                # [2026-03-11] 优化日志：删除CMD输出
                return result
            
            image = Image.open(BytesIO(screenshot_data))
            
            # 优先使用YOLO检测昵称和用户ID
            if self._yolo_detector:
                # 使用profile_logged模型检测
                detections = await self._yolo_detector.detect(
                    device_id, 
                    'profile_logged',
                    conf_threshold=0.3
                )
                
                # [2026-03-11] 优化日志：移除控制台输出
                
                if detections:
                    # 并行OCR识别
                    ocr_tasks = []
                    detection_info = []  # 保存检测信息
                    
                    for det in detections:
                        x1, y1, x2, y2 = det.bbox
                        region = image.crop((x1, y1, x2, y2))
                        region_enhanced = enhance_for_ocr(region)
                        
                        if '昵称' in det.class_name and result['nickname'] is None:
                            # 添加调试日志
                            # [2026-03-11] 优化日志：移除控制台输出
                            # [2026-03-11] 优化日志：移除控制台输出
                            # [2026-03-11] 优化日志：移除控制台输出
                            # [2026-03-11] 优化日志：移除控制台输出
                            
                            ocr_tasks.append(('nickname', det.class_name, self._ocr_pool.recognize(region_enhanced, timeout=5.0)))
                            detection_info.append(('nickname', (x1, y1, x2, y2)))
                        elif 'ID' in det.class_name and result['user_id'] is None:
                            ocr_tasks.append(('user_id', det.class_name, self._ocr_pool.recognize(region_enhanced, timeout=5.0)))
                            detection_info.append(('user_id', None))
                    
                    # 并行执行OCR识别
                    if ocr_tasks:
                        ocr_results = await asyncio.gather(*[task[2] for task in ocr_tasks])
                        
                        # 处理OCR结果
                        for i, (field_type, class_name, _) in enumerate(ocr_tasks):
                            ocr_result = ocr_results[i]
                            
                            # [2026-03-05] 修复数组比较错误：检查 texts 是否为 None 或长度为 0
                            if not ocr_result or ocr_result.texts is None or len(ocr_result.texts) == 0:
                                continue
                            
                            # 添加OCR调试日志
                            # [2026-03-11] 优化日志：移除控制台输出
                            for j, text in enumerate(ocr_result.texts):
                                pass  # [2026-03-11] 优化日志：移除控制台输出
                            
                            # 处理昵称
                            if field_type == 'nickname':
                                # 获取检测区域坐标
                                detection_bbox = detection_info[i][1] if i < len(detection_info) else None
                                
                                nickname = self._extract_nickname_from_texts(
                                    ocr_result.texts,
                                    ocr_result,
                                    detection_bbox
                                )
                                if nickname:
                                    result['nickname'] = nickname
                                    # [2026-03-11] 优化日志：移除控制台输出
                            
                            # 处理用户ID
                            elif field_type == 'user_id':
                                for text in ocr_result.texts:
                                    text = text.strip()
                                    # 修复：不要求文本中必须包含"ID"，直接提取6位以上的数字
                                    match = re.search(r'(\d{6,})', text)
                                    if match:
                                        result['user_id'] = match.group(1)
                                        # [2026-03-11] 优化日志：移除控制台输出
                                        break
            
            # 如果YOLO检测失败，降级到区域OCR
            if result['nickname'] is None or result['user_id'] is None:
                # [2026-03-06] 优先使用区域OCR识别昵称
                if result['nickname'] is None:
                    # [2026-03-11] 优化日志：移除控制台输出
                    result['nickname'] = await self._extract_nickname_from_region(device_id, image)
                    if result['nickname']:
                        pass  # [2026-03-11] 优化日志：移除控制台输出
                
                # 如果区域OCR也失败，或者需要识别用户ID，使用全屏OCR
                if result['nickname'] is None or result['user_id'] is None:
                    # [2026-03-11] 优化日志：移除控制台输出
                    enhanced_image = enhance_for_ocr(image)
                    ocr_result = await self._ocr_pool.recognize(enhanced_image, timeout=10.0)
                    
                    # [2026-03-05] 修复数组比较错误：检查 texts 是否为 None 并且长度大于 0
                    if ocr_result and ocr_result.texts is not None and len(ocr_result.texts) > 0:
                        texts = ocr_result.texts
                        
                        # 保存OCR结果以便提取昵称时使用位置信息
                        self._last_ocr_result = ocr_result
                        
                        if result['nickname'] is None:
                            result['nickname'] = self._extract_nickname(texts)
                            if result['nickname']:
                                pass  # [2026-03-11] 优化日志：移除控制台输出
                        
                        if result['user_id'] is None:
                            result['user_id'] = self._extract_user_id(texts)
                        if result['user_id']:
                            pass  # [2026-03-11] 优化日志：移除控制台输出
            
            return result
            
        except Exception as e:
            # [2026-03-11] 优化日志：移除控制台输出
            return result
    
    async def get_full_profile(self, device_id: str, account: Optional[str] = None, step_number: int = 3, gui_logger = None) -> Dict[str, any]:
        """获取完整的个人资料信息（并行优化版）
        
        Args:
            device_id: 设备ID
            account: 登录账号（可选），用于提取手机号
            step_number: 步骤编号（用于简洁日志）
            gui_logger: GUI日志记录器（可选）
            
        Returns:
            dict: 完整个人资料
                - nickname: str, 昵称
                - user_id: str, 用户ID
                - phone: str, 手机号
                - balance: float, 余额
                - points: int, 积分
                - vouchers: int, 抵扣券数量
        """
        import time
        from .concise_logger import ConciseLogger
        
        # [2026-03-11] 优化日志：不输出到GUI，避免CMD显示过多日志
        concise_logger = ConciseLogger("profile_reader", None, None)
        
        # 记录步骤开始
        concise_logger.step(step_number, "获取资料")
        
        start_time = time.time()
        
        result = {
            'nickname': None,
            'user_id': None,
            'phone': None,
            'balance': None,
            'points': None,
            'vouchers': None,
            'instance_closed': False,  # [2026-03-14] 添加：标记实例是否关闭
        }
        
        if not HAS_PIL or not self._ocr_pool:
            # [2026-03-11] 优化日志：删除CMD输出
            concise_logger.error("PIL 或 OCR 库未安装")
            return result
        
        try:
            # [2026-03-14] 修复原因：删除页面验证和导航逻辑，调用方负责确保在个人页
            # 只测试截图，如果截图失败说明设备已断开
            test_screenshot = await self.adb.screencap(device_id)
            if not test_screenshot:
                concise_logger.error("设备连接已断开")
                result['instance_closed'] = True
                return result
            
            # 记录操作：进入个人页
            concise_logger.action("进入个人页")
            
            # 截图
            screenshot_start = time.time()
            screenshot_data = await self.adb.screencap(device_id)
            if not screenshot_data:
                # [2026-03-11] 优化日志：删除CMD输出
                concise_logger.error("截图失败")
                return result
            
            image = Image.open(BytesIO(screenshot_data))
            screenshot_time = time.time() - screenshot_start
            # [2026-03-11] 优化日志：移除控制台输出
            
            # [2026-03-02] 统一术语：优先使用YOLO识别器检测页面类型
            use_yolo_fallback = True  # 标记是否需要降级到YOLO检测器
            
            # [2026-03-02] 修改原因：PageDetectorIntegrated 不再负责页面类型检测，只负责元素检测
            # 直接尝试检测关闭按钮元素，不依赖页面类型判断
            if self._integrated_detector:
                detect_start = time.time()
                # [2026-03-11] 优化日志：移除控制台输出
                
                # 检测页面元素（查找关闭按钮）
                from .page_detector import PageState
                element_result = await self._integrated_detector.detect_page(
                    device_id, 
                    use_cache=False, 
                    detect_elements=True
                )
                
                detect_time = time.time() - detect_start
                # [2026-03-11] 优化日志：移除控制台输出
                
                # 检查是否检测到关闭按钮（可能有弹窗）
                # [2026-03-05] 修复数组比较错误：使用 is not None 和 len() 检查
                has_close_button = False
                if hasattr(element_result, 'elements') and element_result.elements is not None and len(element_result.elements) > 0:
                    for element in element_result.elements:
                        if "关闭" in element.class_name or "确认" in element.class_name or "确定" in element.class_name:
                            has_close_button = True
                            break
                
                # 如果检测到关闭按钮，说明可能有弹窗，需要处理
                if has_close_button:
                    # [2026-03-11] 优化日志：移除控制台输出
                    
                    # 记录操作：关闭提示弹窗
                    concise_logger.action("关闭提示弹窗")
                    
                    # 弹窗关闭逻辑：每5秒重试一次，最多4次（总共15秒超时）
                    max_attempts = 4
                    retry_interval = 5.0  # 每5秒重试一次
                    close_start_time = time.time()
                    
                    for attempt in range(1, max_attempts + 1):
                        # [2026-03-11] 优化日志：移除控制台输出
                        
                        # 步骤1：YOLO检测关闭按钮并点击
                        clicked = False
                        try:
                            element_result = await self._integrated_detector.detect_page(
                                device_id, 
                                use_cache=False, 
                                detect_elements=True
                            )
                            
                            # [2026-03-01] 修复：检查 elements 属性是否存在
                            # [2026-03-05] 修复数组比较错误：使用 is not None 和 len() 检查
                            if hasattr(element_result, 'elements') and element_result.elements is not None and len(element_result.elements) > 0:
                                # 查找关闭按钮
                                for element in element_result.elements:
                                    if "关闭" in element.class_name or "确认" in element.class_name or "确定" in element.class_name:
                                        # [2026-03-11] 优化日志：移除控制台输出
                                        
                                        # 点击关闭按钮
                                        x1, y1, x2, y2 = element.bbox
                                        center_x = (x1 + x2) // 2
                                        center_y = (y1 + y2) // 2
                                        await self.adb.tap(device_id, center_x, center_y)
                                        # [2026-03-11] 优化日志：移除控制台输出
                                        clicked = True
                                        await asyncio.sleep(0.5)
                                        break
                        except Exception as e:
                            pass  # [2026-03-11] 优化日志：移除控制台输出
                        
                        # 步骤2：OCR确认是否关闭成功
                        # [2026-03-11] 优化日志：移除控制台输出
                        screenshot_data = await self.adb.screencap(device_id)
                        if screenshot_data:
                            image = Image.open(BytesIO(screenshot_data))
                            enhanced_image = enhance_for_ocr(image)
                            ocr_result = await self._ocr_pool.recognize(enhanced_image, timeout=3.0)
                            
                            # [2026-03-05] 修复数组比较错误：检查 texts 是否为 None 并且长度大于 0
                            if ocr_result and ocr_result.texts is not None and len(ocr_result.texts) > 0:
                                texts = ' '.join(ocr_result.texts)
                                
                                # 检查是否有个人页关键词
                                profile_keywords = ['昵称', 'ID', '余额', '积分', '抵扣券', '优惠券']
                                popup_keywords = ['友情提示', '确认', '取消', '关闭', '广告']
                                
                                has_profile = any(keyword in texts for keyword in profile_keywords)
                                has_popup = any(keyword in texts for keyword in popup_keywords)
                                
                                if has_profile and not has_popup:
                                    # [2026-03-11] 优化日志：移除控制台输出
                                    break
                                elif has_popup:
                                    # [2026-03-11] 优化日志：移除控制台输出
                                    await self.adb.press_back(device_id)
                                    await asyncio.sleep(0.5)
                                    
                                    # 再次OCR确认
                                    screenshot_data = await self.adb.screencap(device_id)
                                    if screenshot_data:
                                        image = Image.open(BytesIO(screenshot_data))
                                        enhanced_image = enhance_for_ocr(image)
                                        ocr_result = await self._ocr_pool.recognize(enhanced_image, timeout=3.0)
                                        
                                        # [2026-03-05] 修复数组比较错误：检查 texts 是否为 None 并且长度大于 0
                                        if ocr_result and ocr_result.texts is not None and len(ocr_result.texts) > 0:
                                            texts = ' '.join(ocr_result.texts)
                                            has_profile = any(keyword in texts for keyword in profile_keywords)
                                            has_popup = any(keyword in texts for keyword in popup_keywords)
                                            
                                            if has_profile and not has_popup:
                                                # [2026-03-11] 优化日志：移除控制台输出
                                                break
                        
                        # 检查是否超时（累计计时，不清零）
                        elapsed = time.time() - close_start_time
                        if elapsed >= 15.0:
                            # [2026-03-11] 优化日志：移除控制台输出
                            break
                        
                        # 如果不是最后一次尝试，等待5秒后重试
                        if attempt < max_attempts:
                            remaining = retry_interval - (time.time() - close_start_time - (attempt - 1) * retry_interval)
                            if remaining > 0:
                                # [2026-03-11] 优化日志：移除控制台输出
                                await asyncio.sleep(remaining)

                
                # 现在开始检测页面元素（昵称、余额等）
                # [2026-03-12] 优化日志：移除获取详细资料的技术日志
                
                yolo_start = time.time()
                # [2026-03-11] 优化日志：移除控制台输出
                # [2026-02-22] 删除调试日志
                
                # 使用YOLO识别器的detect_page方法，启用元素检测
                detection_result = await self._integrated_detector.detect_page(
                    device_id, 
                    use_cache=False, 
                    detect_elements=True
                )
                
                yolo_time = time.time() - yolo_start
                # [2026-03-11] 优化日志：移除控制台输出
                
                # [2026-03-01] 修复：检查 elements 属性是否存在（PageDetectorDL 不支持元素检测）
                # [2026-03-05] 修复数组比较错误：使用 is not None 和 len() 检查
                has_elements = hasattr(detection_result, 'elements') and detection_result.elements is not None and len(detection_result.elements) > 0
                
                # [2026-03-11] 修复原因：兼容PageDetectionResult和IntegratedDetectionResult两种类型
                # 检查是否是IntegratedDetectionResult（有elements属性）
                if hasattr(detection_result, 'elements') and detection_result.elements is not None:
                    has_elements = len(detection_result.elements) > 0
                    if has_elements:
                        # [2026-03-11] 优化日志：移除控制台输出
                        # 打印检测到的元素详情
                        for elem in detection_result.elements:
                            pass  # [2026-02-22] 删除调试日志
                    else:
                        pass  # [2026-03-11] 优化日志：移除控制台输出
                else:
                    # PageDetectionResult类型，没有elements属性，跳过YOLO元素检测
                    pass  # [2026-03-11] 优化日志：移除控制台输出
                    has_elements = False
                
                # ===== 优化：全屏OCR一次，然后根据YOLO位置匹配文本 =====
                # [2026-03-01] 修复：检查 elements 属性是否存在
                if has_elements:
                    ocr_start = time.time()
                    
                    # 全屏OCR识别（只调用一次）
                    # [2026-03-11] 优化日志：移除控制台输出
                    enhanced_image = enhance_for_ocr(image)
                    full_ocr_result = await self._ocr_pool.recognize(enhanced_image)
                    
                    ocr_time = time.time() - ocr_start
                    # [2026-03-11] 优化日志：移除控制台输出
                    
                    # [2026-03-05] 修复数组比较错误：检查 texts 和 boxes 是否为 None 并且长度大于 0
                    if (full_ocr_result and 
                        full_ocr_result.texts is not None and len(full_ocr_result.texts) > 0 and 
                        full_ocr_result.boxes is not None and len(full_ocr_result.boxes) > 0):
                        # [2026-03-11] 优化日志：移除控制台输出
                        
                        # 根据YOLO检测到的元素位置，从全屏OCR结果中匹配文本
                        for element in detection_result.elements:
                            x1, y1, x2, y2 = element.bbox
                            element_center_x = (x1 + x2) / 2
                            element_center_y = (y1 + y2) / 2
                            
                            # 查找与元素位置重叠的OCR文本
                            matched_texts = []
                            for i, (text, box) in enumerate(zip(full_ocr_result.texts, full_ocr_result.boxes)):
                                # 计算OCR文本框的中心点
                                box_flat = box.flatten().tolist() if hasattr(box, 'flatten') else box
                                ocr_x1, ocr_y1 = box_flat[0], box_flat[1]
                                ocr_x2, ocr_y2 = box_flat[4], box_flat[5]
                                ocr_center_x = (ocr_x1 + ocr_x2) / 2
                                ocr_center_y = (ocr_y1 + ocr_y2) / 2
                                
                                # 检查OCR文本框是否在YOLO元素框内
                                if x1 <= ocr_center_x <= x2 and y1 <= ocr_center_y <= y2:
                                    matched_texts.append(text)
                            
                            # [2026-03-11] 优化日志：移除控制台输出
                            
                            if not matched_texts:
                                # 检测到元素但没有匹配到OCR文本
                                # [2026-03-11] 优化日志：移除控制台输出
                                # 对于积分和优惠券，如果检测到区域但没有文本，设置为0
                                if '积分' in element.class_name and result['points'] is None:
                                    result['points'] = 0
                                    # [2026-03-11] 优化日志：移除控制台输出
                                    # [2026-03-11] 优化日志：移除控制台输出
                                continue
                            
                            # 处理昵称
                            if '昵称' in element.class_name and result['nickname'] is None:
                                nickname = self._extract_nickname_from_texts(
                                    matched_texts,
                                    ocr_result=full_ocr_result,
                                    detection_bbox=element.bbox
                                )
                                if nickname:
                                    result['nickname'] = nickname
                                    # [2026-03-11] 优化日志：移除控制台输出
                            
                            # 处理用户ID
                            elif 'ID' in element.class_name and result['user_id'] is None:
                                for text in matched_texts:
                                    text = text.strip()
                                    # 修复：不要求文本中必须包含"ID"，因为OCR可能只识别到数字部分
                                    # 直接尝试提取6位以上的数字
                                    match = re.search(r'(\d{6,})', text)
                                    if match:
                                        result['user_id'] = match.group(1)
                                        # [2026-03-11] 优化日志：移除控制台输出
                                        break
                            
                            # 处理余额、积分、抵扣券、优惠券
                            else:
                                # 合并所有匹配的文本
                                combined_text = ' '.join(matched_texts)
                                
                                # 查找所有数字（包括小数）
                                all_numbers = re.findall(r'(\d+\.?\d*)', combined_text)
                                
                                if all_numbers:
                                    # 尝试合并连续的数字（处理"1 0.24"这种情况）
                                    if len(all_numbers) > 1:
                                        try:
                                            first = all_numbers[0]
                                            second = all_numbers[1]
                                            
                                            if '.' in second or '.' not in first:
                                                combined = first + second if '.' in second else first + '.' + second
                                                try:
                                                    combined_value = float(combined)
                                                    if combined_value > float(first):
                                                        all_numbers[0] = str(combined_value)
                                                        # [2026-03-11] 优化日志：移除控制台输出
                                                except ValueError:
                                                    pass
                                        except (IndexError, ValueError):
                                            pass
                                    
                                    # 转换为浮点数并选择最大的合理值
                                    valid_numbers = []
                                    for num_str in all_numbers:
                                        try:
                                            num = float(num_str)
                                            # 根据元素类别进行合理性检查
                                            if '余额' in element.class_name and 0.01 <= num <= 100000:
                                                valid_numbers.append(num)
                                            elif '积分' in element.class_name and 0 <= num <= 1000000:
                                                valid_numbers.append(num)
                                            elif '抵扣' in element.class_name and 0 <= num <= 10000:
                                                valid_numbers.append(num)
                                            elif '优惠' in element.class_name and 0 <= num <= 1000:
                                                valid_numbers.append(num)
                                        except ValueError:
                                            continue
                                    
                                    if valid_numbers:
                                        value = max(valid_numbers)
                                        
                                        # 添加详细调试日志
                                        # [2026-03-11] 优化日志：移除控制台输出
                                        
                                        if '余额' in element.class_name and result['balance'] is None:
                                            result['balance'] = value
                                            # [2026-03-11] 优化日志：移除控制台输出
                                        elif '积分' in element.class_name and result['points'] is None:
                                            result['points'] = int(value)
                                            # [2026-03-11] 优化日志：移除控制台输出
                                        elif '抵扣' in element.class_name:
                                            if result['vouchers'] is None or value > result['vouchers']:
                                                result['vouchers'] = value
                                                # [2026-03-11] 优化日志：移除控制台输出
                                        else:
                                            pass  # [2026-03-11] 优化日志：移除控制台输出
                                else:
                                    # 匹配到文本但没有数字
                                    pass  # [2026-03-11] 优化日志：移除控制台输出
                                    # 对于积分和优惠券，如果匹配到区域但没有数字，设置为0
                                    if '积分' in element.class_name and result['points'] is None:
                                        result['points'] = 0
                                        # [2026-03-11] 优化日志：移除控制台输出
                                        # [2026-03-11] 优化日志：移除控制台输出
                
                # [2026-03-02] 统一术语：如果YOLO识别器成功检测到元素，则不需要降级到YOLO
                # [2026-03-01] 修复：检查 elements 属性是否存在
                if has_elements:
                    use_yolo_fallback = False
            
            # ===== 降级：使用旧的YOLO检测器 =====
            # [2026-03-02] 统一术语：当YOLO识别器未检测到元素时，也应该尝试YOLO检测器
            if use_yolo_fallback and self._yolo_detector:
                # 创建并行YOLO检测任务（降低置信度阈值以提高检测成功率）
                yolo_start = time.time()
                yolo_tasks = [
                    self._yolo_detector.detect(device_id, 'profile_logged', conf_threshold=0.25),
                    self._yolo_detector.detect(device_id, 'balance', conf_threshold=0.25)
                ]
                
                # 并行执行YOLO检测
                profile_detections, balance_detections = await asyncio.gather(*yolo_tasks)
                yolo_time = time.time() - yolo_start
                # [2026-03-11] 优化日志：移除控制台输出
                
                # [2026-03-11] 优化日志：移除控制台输出
                # [2026-03-11] 优化日志：移除控制台输出
                
                # ===== 并行优化：同时进行OCR识别 =====
                ocr_start = time.time()
                ocr_tasks = []
                
                # 处理profile_logged检测结果（昵称和用户ID）
                for det in profile_detections:
                    x1, y1, x2, y2 = det.bbox
                    region = image.crop((x1, y1, x2, y2))
                    region_enhanced = enhance_for_ocr(region)
                    
                    if '昵称' in det.class_name and result['nickname'] is None:
                        # 添加调试日志
                        # [2026-03-11] 优化日志：移除控制台输出
                        # [2026-03-11] 优化日志：移除控制台输出
                        # [2026-03-11] 优化日志：移除控制台输出
                        # [2026-03-11] 优化日志：移除控制台输出
                        
                        ocr_tasks.append(('nickname', det.class_name, det.bbox, self._ocr_pool.recognize(region_enhanced, timeout=3.0)))
                    elif 'ID' in det.class_name and result['user_id'] is None:
                        ocr_tasks.append(('user_id', det.class_name, det.bbox, self._ocr_pool.recognize(region_enhanced, timeout=3.0)))
                
                # 处理balance检测结果（余额、积分、抵扣券、优惠券）
                for det in balance_detections:
                    x1, y1, x2, y2 = det.bbox
                    region = image.crop((x1, y1, x2, y2))
                    region_enhanced = enhance_for_ocr(region)
                    ocr_tasks.append((det.class_name, det.class_name, det.bbox, self._ocr_pool.recognize(region_enhanced, timeout=2.0)))
                
                # 并行执行所有OCR识别
                if ocr_tasks:
                    ocr_results = await asyncio.gather(*[task[3] for task in ocr_tasks])
                    ocr_time = time.time() - ocr_start
                    # [2026-03-11] 优化日志：移除控制台输出
                    
                    # 处理OCR结果
                    for i, (field_type, class_name, bbox, _) in enumerate(ocr_tasks):
                        ocr_result = ocr_results[i]
                        
                        # [2026-03-05] 修复数组比较错误：检查 texts 是否为 None 或长度为 0
                        if not ocr_result or ocr_result.texts is None or len(ocr_result.texts) == 0:
                            continue
                        
                        # 添加OCR调试日志
                        # [2026-03-11] 优化日志：移除控制台输出
                        for j, text in enumerate(ocr_result.texts):
                            pass  # [2026-03-11] 优化日志：移除控制台输出
                        
                        # 处理昵称
                        if field_type == 'nickname':
                            nickname = self._extract_nickname_from_texts(
                                ocr_result.texts,
                                ocr_result=ocr_result,
                                detection_bbox=bbox
                            )
                            if nickname:
                                result['nickname'] = nickname
                                # [2026-03-11] 优化日志：移除控制台输出
                        
                        # 处理用户ID
                        elif field_type == 'user_id':
                            for text in ocr_result.texts:
                                text = text.strip()
                                # 修复：不要求文本中必须包含"ID"，直接提取6位以上的数字
                                match = re.search(r'(\d{6,})', text)
                                if match:
                                    result['user_id'] = match.group(1)
                                    # [2026-03-11] 优化日志：移除控制台输出
                                    break
                        
                        # 处理余额、积分、抵扣券、优惠券
                        else:
                            # 合并所有文本，处理分散识别的情况（如"1 0.24"）
                            combined_text = ' '.join(ocr_result.texts)
                            
                            # 查找所有数字（包括小数）
                            all_numbers = re.findall(r'(\d+\.?\d*)', combined_text)
                            
                            if all_numbers:
                                # 尝试合并连续的数字（处理"1 0.24"这种情况）
                                if len(all_numbers) > 1:
                                    try:
                                        first = all_numbers[0]
                                        second = all_numbers[1]
                                        
                                        # 如果第二个数字以小数点开头或第一个数字是整数
                                        if '.' in second or '.' not in first:
                                            combined = first + second if '.' in second else first + '.' + second
                                            try:
                                                combined_value = float(combined)
                                                if combined_value > float(first):
                                                    all_numbers[0] = str(combined_value)
                                                    # [2026-03-11] 优化日志：移除控制台输出
                                            except ValueError:
                                                pass
                                    except (IndexError, ValueError):
                                        pass
                                
                                # 转换为浮点数并排序（选择最大的合理值）
                                valid_numbers = []
                                for num_str in all_numbers:
                                    try:
                                        num = float(num_str)
                                        # 根据字段类型进行合理性检查
                                        if '余额' in class_name and 0.01 <= num <= 100000:
                                            valid_numbers.append(num)
                                        elif '积分' in class_name and 0 <= num <= 1000000:
                                            valid_numbers.append(num)
                                        elif '抵扣' in class_name and 0 <= num <= 10000:
                                            valid_numbers.append(num)
                                        elif '优惠' in class_name and 0 <= num <= 1000:
                                            valid_numbers.append(num)
                                    except ValueError:
                                        continue
                                
                                if valid_numbers:
                                    # 选择最大的合理值
                                    value = max(valid_numbers)
                                    
                                    if '余额' in class_name and result['balance'] is None:
                                        result['balance'] = value
                                        # [2026-03-11] 优化日志：移除控制台输出
                                    elif '积分' in class_name and result['points'] is None:
                                        result['points'] = int(value)
                                        # [2026-03-11] 优化日志：移除控制台输出
                                    elif '抵扣' in class_name and result['vouchers'] is None:
                                        result['vouchers'] = value
                                        # [2026-03-11] 优化日志：移除控制台输出
            
            # 如果YOLO检测失败，降级到串行OCR方法
            if result['nickname'] is None or result['user_id'] is None:
                fallback_start = time.time()
                # [2026-03-11] 优化日志：移除控制台输出
                # 使用OCR图像预处理模块增强图像
                enhanced_image = enhance_for_ocr(image)
                ocr_result = await self._ocr_pool.recognize(enhanced_image, timeout=5.0)
                
                # [2026-03-05] 修复数组比较错误：检查 texts 是否为 None 并且长度大于 0
                if ocr_result and ocr_result.texts is not None and len(ocr_result.texts) > 0:
                    texts = ocr_result.texts
                    
                    # 保存OCR结果以便提取昵称时使用位置信息
                    self._last_ocr_result = ocr_result
                    
                    if result['nickname'] is None:
                        result['nickname'] = self._extract_nickname(texts)
                    
                    if result['user_id'] is None:
                        result['user_id'] = self._extract_user_id(texts)
                
                fallback_time = time.time() - fallback_start
                # [2026-03-11] 优化日志：移除控制台输出
            
            # 手机号只能从登录账号中提取
            if account:
                result['phone'] = self._extract_phone_from_account(account)
            
            # 如果YOLO检测失败，降级到区域OCR识别
            # 只在关键字段（余额）缺失时才降级，其他字段可以为None
            if result['balance'] is None:
                region_start = time.time()
                # [2026-03-11] 优化日志：移除控制台输出
                region_results = await self._recognize_regions(device_id, image)
                
                if result['balance'] is None:
                    result['balance'] = region_results.get('balance')
                if result['points'] is None:
                    result['points'] = region_results.get('points')
                if result['vouchers'] is None:
                    result['vouchers'] = region_results.get('vouchers')
                
                region_time = time.time() - region_start
                # [2026-03-11] 优化日志：移除控制台输出
            
            total_time = time.time() - start_time
            # [2026-03-11] 优化日志：移除控制台输出
            
            # 记录成功日志
            # 记录成功日志
            # [2026-03-12] 优化日志：移除获取详细资料的技术日志
            concise_logger.success("资料获取完成")
            # 添加分隔线
            if concise_logger.gui_logger:
                concise_logger.gui_logger.info("=" * 60)
            
            # 显示详细资料，每个字段单独一行
            if result.get('nickname'):
                concise_logger.action(f"昵称: {result['nickname']}")
            if result.get('user_id'):
                concise_logger.action(f"用户ID: {result['user_id']}")
            if result.get('balance') is not None:
                concise_logger.action(f"余额: {result['balance']:.2f}元")
            if result.get('points') is not None:
                concise_logger.action(f"积分: {result['points']}")
            if result.get('vouchers') is not None:
                concise_logger.action(f"抵扣券: {result['vouchers']}")
            
            return result
            
        except Exception as e:
            # [2026-03-11] 优化日志：移除控制台输出
            concise_logger.error("获取资料失败", e)
            return result
    
    def _is_chinese_char(self, char: str) -> bool:
        """检查单个字符是否为中文
        
        Args:
            char: 单个字符
            
        Returns:
            bool: 是否为中文字符
        """
        return '\u4e00' <= char <= '\u9fff'
    
    def _is_chinese_text(self, text: str) -> bool:
        """检查文本是否包含中文字符
        
        Args:
            text: 文本字符串
            
        Returns:
            bool: 是否包含中文字符
        """
        chinese_count = sum(1 for c in text if self._is_chinese_char(c))
        return chinese_count > 0
    
    def _is_pure_number(self, text: str) -> bool:
        """检查文本是否为纯数字
        
        Args:
            text: 文本字符串
            
        Returns:
            bool: 是否为纯数字
        """
        return text.isdigit()
    
    def _is_pure_symbol(self, text: str) -> bool:
        """检查文本是否为纯特殊符号
        
        Args:
            text: 文本字符串
            
        Returns:
            bool: 是否为纯特殊符号
        """
        return all(not c.isalnum() for c in text)
    
    def _calculate_nickname_confidence(
        self, 
        text: str, 
        position_info: Optional[Dict] = None
    ) -> float:
        """计算昵称候选的置信度分数
        
        Args:
            text: 候选文本
            position_info: 位置信息字典,包含:
                - center_x: 文本中心x坐标
                - center_y: 文本中心y坐标
                - region_center_x: 检测区域中心x坐标
                - region_center_y: 检测区域中心y坐标
        
        Returns:
            float: 置信度分数 (0.0 - 1.0)
        """
        # [2026-03-05] 修复原因：降低过滤条件，提高昵称识别成功率
        # 排除关键字列表（减少排除关键字，避免误杀）
        # [2026-03-05] 添加品牌名称过滤：排除"溪盟山泉"、"溪盟"等品牌相关文字
        # [2026-03-06] 添加"福利"关键字：排除页面上的"福利"文字
        # [2026-03-06] 添加"溪"、"西"单字：OCR可能把"溪盟山泉"识别成单字
        exclude_keywords = [
            "溪盟山泉", "溪盟", "山泉", "溪", "西",  # 品牌名称及其单字
            "福利",  # 页面固定文字
            "手机", "余额", "积分", 
            "抵扣券", "优惠券", "我的", "设置", "首页", "分类",
            "商城", "订单", "查看", "待付款", "待发货", "待收货", "待评价",
            "元", "张", "次"
        ]
        
        # 检查排除关键字(返回0分)
        for kw in exclude_keywords:
            if kw in text:
                return 0.0
        
        # [2026-03-05] 修复原因：提高基础分数，降低过滤门槛
        # 1. 基础分数（从0.3提高到0.4）
        score = 0.4
        
        # 2. 中文字符加分 (+0.3)
        if self._is_chinese_text(text):
            score += 0.3
        
        # 3. 长度评分（放宽长度限制）
        text_len = len(text)
        if 2 <= text_len <= 15:  # 从10放宽到15
            score += 0.2  # 理想长度
        elif 1 <= text_len <= 30:  # 从20放宽到30
            score += 0.1  # 可接受长度
        
        # 4. 纯数字惩罚 (-0.2，从-0.3降低到-0.2)
        if self._is_pure_number(text) and text_len <= 3:
            score -= 0.2
        
        # 5. 特殊符号惩罚 (-0.05 per symbol, max -0.15，从-0.1降低到-0.05)
        symbol_count = sum(1 for c in text if not c.isalnum() and not self._is_chinese_char(c))
        if symbol_count > 0:
            score -= 0.05 * min(symbol_count, 3)
        
        # 6. 位置加分 (+0.2)
        if position_info:
            try:
                text_center_x = position_info.get('center_x')
                text_center_y = position_info.get('center_y')
                region_center_x = position_info.get('region_center_x')
                region_center_y = position_info.get('region_center_y')
                
                if all([text_center_x is not None, text_center_y is not None,
                       region_center_x is not None, region_center_y is not None]):
                    # 计算距离
                    distance = ((text_center_x - region_center_x) ** 2 + 
                               (text_center_y - region_center_y) ** 2) ** 0.5
                    
                    # 如果距离小于50像素,认为靠近中心
                    if distance < 50:
                        score += 0.2
            except Exception:
                pass  # 位置信息处理失败,跳过位置加分
        
        # 确保分数在0.0-1.0范围内
        return max(0.0, min(1.0, score))
    
    def _extract_nickname_from_texts(
        self, 
        texts: List[str],
        ocr_result: Optional[any] = None,
        detection_bbox: Optional[tuple] = None
    ) -> Optional[str]:
        """从OCR文本列表中提取昵称(改进版)
        
        Args:
            texts: OCR识别的文本列表
            ocr_result: OCR结果对象(包含boxes信息)
            detection_bbox: YOLO检测的区域坐标 (x1, y1, x2, y2)
        
        Returns:
            str: 提取的昵称,如果没有找到则返回None
        """
        if not texts:
            # [2026-03-11] 优化日志：移除控制台输出
            return None
        
        # [2026-03-11] 优化日志：移除控制台输出
        # [2026-03-11] 优化日志：移除控制台输出
        
        # 会员等级标识关键字
        member_keywords = [
            "钻石会员", "黄金会员", "白金会员", "铂金会员",
            "普通会员", "初级会员", "银牌会员",
            "VIP会员", "SVIP", "VIP",
            "vip会员", "vip", "Vip",
            "会员"
        ]
        
        # 准备候选列表
        candidates = []
        
        # 计算检测区域中心(如果提供)
        region_center = None
        if detection_bbox:
            x1, y1, x2, y2 = detection_bbox
            region_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            # [2026-03-11] 优化日志：移除控制台输出
        
        # 遍历所有文本
        for i, text in enumerate(texts):
            text = text.strip()
            if not text:
                continue
            
            # 处理会员等级标识
            nickname_candidate = text
            for member_kw in member_keywords:
                if member_kw in text:
                    nickname_candidate = text.split(member_kw)[0].strip()
                    # [2026-03-11] 优化日志：移除控制台输出
                    break
            
            if not nickname_candidate:
                continue
            
            # 准备位置信息
            position_info = None
            if ocr_result and hasattr(ocr_result, 'boxes') and ocr_result.boxes is not None:
                try:
                    if i < len(ocr_result.boxes):
                        box = ocr_result.boxes[i]
                        # 计算文本框中心
                        if hasattr(box, 'flatten'):
                            box_flat = box.flatten().tolist()
                        else:
                            box_flat = box
                        
                        # box格式: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                        text_center_x = (box_flat[0] + box_flat[4]) / 2
                        text_center_y = (box_flat[1] + box_flat[5]) / 2
                        
                        position_info = {
                            'center_x': text_center_x,
                            'center_y': text_center_y
                        }
                        
                        if region_center:
                            position_info['region_center_x'] = region_center[0]
                            position_info['region_center_y'] = region_center[1]
                except Exception as e:
                    pass  # [2026-03-11] 优化日志：移除控制台输出
            
            # 计算置信度
            confidence = self._calculate_nickname_confidence(
                nickname_candidate, 
                position_info
            )
            
            # 记录调试信息
            # [2026-03-11] 优化日志：移除控制台输出
            
            if confidence > 0:
                candidates.append((nickname_candidate, confidence))
        
        # 按置信度排序
        if not candidates:
            # [2026-03-11] 优化日志：移除控制台输出
            return None
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 输出最终选择
        best_candidate = candidates[0]
        # [2026-03-11] 优化日志：移除控制台输出
        
        return best_candidate[0]
    
    async def get_full_profile_with_retry(self, device_id: str, max_retries: int = 3, account: Optional[str] = None, 
                                          gui_logger=None, step_number: int = 2) -> Dict[str, any]:
        """获取完整个人资料，支持重试机制和缓存
        
        优化策略：
        1. 登录时就有手机号，直接从缓存查询昵称和ID
        2. 如果缓存有完整数据（昵称+用户ID），则只获取余额等动态数据
        3. 如果缓存不完整，才进行完整OCR识别
        4. 识别成功后，自动保存到缓存
        
        Args:
            device_id: 设备ID
            max_retries: 最大重试次数，默认3次
            account: 登录账号（可选），用于提取手机号和使用缓存
            gui_logger: GUI日志记录器（可选）
            step_number: 步骤编号（用于简洁日志）
            
        Returns:
            dict: 完整个人资料（累积最佳结果）
        """
        # [2026-03-11] 优化日志：移除控制台输出
        
        # 提取手机号（登录时就有）
        phone = None
        if account:
            phone = self._extract_phone_from_account(account)
            if phone:
                self._silent_log.log(f"[账号] 手机号: {phone}")
        
        best_result = {}
        collected_fields = []
        
        # ===== 优化：从缓存获取昵称和用户ID作为降级方案 =====
        # 注意：不再跳过身份识别，而是始终尝试识别，以检测用户改名
        cached_nickname = None
        cached_user_id = None
        if phone:
            cached_nickname = self._cache.get_nickname(phone)
            cached_user_id = self._cache.get_user_id(phone)
            
            if cached_nickname and cached_user_id:
                self._silent_log.log(f"[缓存] 找到缓存信息（作为降级方案）")
                self._silent_log.log(f"[缓存] - 昵称: {cached_nickname}")
                self._silent_log.log(f"[缓存] - 用户ID: {cached_user_id}")
            elif cached_nickname or cached_user_id:
                self._silent_log.log(f"[缓存] 找到部分缓存信息")
                if cached_nickname:
                    self._silent_log.log(f"[缓存] - 昵称: {cached_nickname}")
                if cached_user_id:
                    self._silent_log.log(f"[缓存] - 用户ID: {cached_user_id}")
            else:
                self._silent_log.log(f"[缓存] 未找到缓存")
        
        # 手机号可以直接从账号中提取
        if phone:
            best_result['phone'] = phone
            collected_fields.append('phone')
        
        for attempt in range(max_retries):
            try:
                # 始终执行完整识别，以检测用户改名
                # [2026-03-11] 优化日志：移除控制台输出
                self._silent_log.log(f"[尝试 {attempt + 1}/{max_retries}] 开始完整OCR识别...")
                profile = await self.get_full_profile(device_id, account=account, gui_logger=gui_logger, step_number=step_number)
                
                # 静默记录OCR识别到的原始数据
                # [2026-02-22] 删除调试日志
                self._silent_log.log(f"  - nickname: {profile.get('nickname')}")
                self._silent_log.log(f"  - user_id: {profile.get('user_id')}")
                self._silent_log.log(f"  - phone: {profile.get('phone')}")
                self._silent_log.log(f"  - balance: {profile.get('balance')}")
                self._silent_log.log(f"  - points: {profile.get('points')}")
                self._silent_log.log(f"  - vouchers: {profile.get('vouchers')}")
                
                # 合并结果（保留非空值）
                newly_collected = []
                for key, value in profile.items():
                    if value is not None and best_result.get(key) is None:
                        best_result[key] = value
                        newly_collected.append(key)
                        if key not in collected_fields:
                            collected_fields.append(key)
                
                # 显示本次新获取的字段
                if newly_collected:
                    field_names = {
                        'nickname': '昵称',
                        'user_id': '用户ID',
                        'phone': '手机号',
                        'balance': '余额',
                        'points': '积分',
                        'vouchers': '抵扣券',
                    }
                    new_field_names = [field_names.get(f, f) for f in newly_collected]
                    self._silent_log.log(f"[尝试 {attempt + 1}/{max_retries}] 新获取: {', '.join(new_field_names)}")
                
                # ===== 优化：更新缓存（如果获取到新的昵称或用户ID）=====
                if phone:
                    new_nickname = profile.get('nickname')
                    new_user_id = profile.get('user_id')
                    
                    if new_nickname or new_user_id:
                        # 检查是否是新数据
                        cached_nickname = self._cache.get_nickname(phone)
                        cached_user_id = self._cache.get_user_id(phone)
                        
                        if new_nickname and new_nickname != cached_nickname:
                            self._cache.set(phone, nickname=new_nickname)
                            self._silent_log.log(f"[缓存] 已保存昵称: {new_nickname}")
                        
                        if new_user_id and new_user_id != cached_user_id:
                            self._cache.set(phone, user_id=new_user_id)
                            self._silent_log.log(f"[缓存] 已保存用户ID: {new_user_id}")
                
                # 检查是否所有字段都已获取
                # [2026-03-01] 删除优惠券字段
                all_fields = ['nickname', 'user_id', 'phone', 'balance', 'points', 'vouchers']
                missing_fields = [f for f in all_fields if best_result.get(f) is None]
                
                if not missing_fields:
                    # [2026-03-11] 优化日志：移除控制台输出
                    self._log_collection_summary(collected_fields, [])
                    
                    # 使用 ConciseLogger 显示详细资料（如果提供了 gui_logger）
                    if gui_logger:
                        from .concise_logger import ConciseLogger
                        import logging
                        file_logger = logging.getLogger(__name__)
                        # [2026-03-11] 优化日志：不输出到GUI，避免CMD显示过多日志
                        concise = ConciseLogger("profile_reader", None, file_logger)
                        
                        # 显示成功消息
                        concise.success("资料获取完成")
                        
                        # 添加分隔线
                        if gui_logger:
                            gui_logger.info("=" * 60)
                        
                        # 显示详细资料，每个字段单独一行，使用 → 前缀
                        if best_result.get('nickname'):
                            gui_logger.info(f"  → 昵称: {best_result['nickname']}")
                        if best_result.get('user_id'):
                            gui_logger.info(f"  → 用户ID: {best_result['user_id']}")
                        if best_result.get('balance') is not None:
                            gui_logger.info(f"  → 余额: {best_result['balance']:.2f}元")
                        if best_result.get('points') is not None:
                            gui_logger.info(f"  → 积分: {best_result['points']}")
                        if best_result.get('vouchers') is not None:
                            gui_logger.info(f"  → 抵扣券: {best_result['vouchers']}")
                    
                    return best_result
                
                if attempt < max_retries - 1:
                    field_names = {
                        'nickname': '昵称',
                        'user_id': '用户ID',
                        'phone': '手机号',
                        'balance': '余额',
                        'points': '积分',
                        'vouchers': '抵扣券',
                    }
                    missing_field_names = [field_names.get(f, f) for f in missing_fields]
                    self._silent_log.log(f"[尝试 {attempt + 1}/{max_retries}] 仍缺少: {', '.join(missing_field_names)}")
                    self._silent_log.log(f"等待2秒后重试...")
                    await asyncio.sleep(2)  # 等待2秒后重试
                    
            except Exception as e:
                self._silent_log.log(f"[尝试 {attempt + 1}/{max_retries}] OCR识别出错: {str(e)}")
                if attempt < max_retries - 1:
                    self._silent_log.log(f"等待2秒后重试...")
                    await asyncio.sleep(2)
        
        # 所有重试后，尝试备选方案
        # [2026-03-01] 删除优惠券字段
        all_fields = ['nickname', 'user_id', 'phone', 'balance', 'points', 'vouchers']
        missing_fields = [f for f in all_fields if best_result.get(f) is None]
        
        if missing_fields:
            pass  # [2026-03-11] 优化日志：移除控制台输出
            
            # 优先使用缓存作为降级方案
            if phone and (best_result.get('nickname') is None or best_result.get('user_id') is None):
                pass  # [2026-03-11] 优化日志：移除控制台输出
                
                if best_result.get('nickname') is None and cached_nickname:
                    best_result['nickname'] = cached_nickname
                    collected_fields.append('nickname')
                    # [2026-03-11] 优化日志：移除控制台输出
                
                if best_result.get('user_id') is None and cached_user_id:
                    best_result['user_id'] = cached_user_id
                    collected_fields.append('user_id')
                    # [2026-03-11] 优化日志：移除控制台输出
            
            # 重新检查缺失字段
            missing_fields = [f for f in all_fields if best_result.get(f) is None]
            
            if missing_fields:
                pass  # [2026-03-11] 优化日志：移除控制台输出
            
            fallback_success = []
            fallback_failed = []
            
            # 尝试备选方案获取缺失字段
            if best_result.get('balance') is None:
                try:
                    # [2026-03-11] 优化日志：移除控制台输出
                    balance = await self.get_balance_fallback(device_id)
                    if balance is not None:
                        best_result['balance'] = balance
                        fallback_success.append('余额')
                        collected_fields.append('balance')
                        # [2026-03-11] 优化日志：移除控制台输出
                    else:
                        fallback_failed.append('余额')
                        # [2026-03-11] 优化日志：移除控制台输出
                except Exception as e:
                    fallback_failed.append('余额')
                    # [2026-03-11] 优化日志：移除控制台输出
            
            if best_result.get('user_id') is None:
                try:
                    # [2026-03-11] 优化日志：移除控制台输出
                    user_id = await self.get_user_id_fallback(device_id)
                    if user_id is not None:
                        best_result['user_id'] = user_id
                        fallback_success.append('用户ID')
                        collected_fields.append('user_id')
                        # [2026-03-11] 优化日志：移除控制台输出
                    else:
                        fallback_failed.append('用户ID')
                        # [2026-03-11] 优化日志：移除控制台输出
                except Exception as e:
                    fallback_failed.append('用户ID')
                    # [2026-03-11] 优化日志：移除控制台输出
            
            if best_result.get('nickname') is None:
                try:
                    # [2026-03-11] 优化日志：移除控制台输出
                    nickname = await self.get_nickname_fallback(device_id)
                    if nickname is not None:
                        best_result['nickname'] = nickname
                        fallback_success.append('昵称')
                        collected_fields.append('nickname')
                        # [2026-03-11] 优化日志：移除控制台输出
                    else:
                        fallback_failed.append('昵称')
                        # [2026-03-11] 优化日志：移除控制台输出
                except Exception as e:
                    fallback_failed.append('昵称')
                    # [2026-03-11] 优化日志：移除控制台输出
            
            if best_result.get('phone') is None:
                try:
                    # [2026-03-11] 优化日志：移除控制台输出
                    phone = await self.get_phone_fallback(device_id)
                    if phone is not None:
                        best_result['phone'] = phone
                        fallback_success.append('手机号')
                        collected_fields.append('phone')
                        # [2026-03-11] 优化日志：移除控制台输出
                    else:
                        fallback_failed.append('手机号')
                        # [2026-03-11] 优化日志：移除控制台输出
                except Exception as e:
                    fallback_failed.append('手机号')
                    # [2026-03-11] 优化日志：移除控制台输出
            
            if best_result.get('points') is None:
                try:
                    # [2026-03-11] 优化日志：移除控制台输出
                    points = await self.get_points_fallback(device_id)
                    if points is not None:
                        best_result['points'] = points
                        fallback_success.append('积分')
                        collected_fields.append('points')
                        # [2026-03-11] 优化日志：移除控制台输出
                    else:
                        fallback_failed.append('积分')
                        # [2026-03-11] 优化日志：移除控制台输出
                except Exception as e:
                    fallback_failed.append('积分')
                    # [2026-03-11] 优化日志：移除控制台输出
            
            if best_result.get('vouchers') is None:
                try:
                    # [2026-03-11] 优化日志：移除控制台输出
                    vouchers = await self.get_vouchers_fallback(device_id)
                    if vouchers is not None:
                        best_result['vouchers'] = vouchers
                        fallback_success.append('抵扣券')
                        collected_fields.append('vouchers')
                        # [2026-03-11] 优化日志：移除控制台输出
                    else:
                        fallback_failed.append('抵扣券')
                        # [2026-03-11] 优化日志：移除控制台输出
                except Exception as e:
                    fallback_failed.append('抵扣券')
                    # [2026-03-11] 优化日志：移除控制台输出
            
            # 显示备选方案总结
            if fallback_success or fallback_failed:
                pass  # [2026-03-11] 优化日志：移除控制台输出
                if fallback_success:
                    pass  # [2026-03-11] 优化日志：移除控制台输出
                if fallback_failed:
                    pass  # [2026-03-11] 优化日志：移除控制台输出
        
        # 显示最终收集结果
        final_missing = [f for f in all_fields if best_result.get(f) is None]
        field_names = {
            'nickname': '昵称',
            'user_id': '用户ID',
            'phone': '手机号',
            'balance': '余额',
            'points': '积分',
            'vouchers': '抵扣券',
        }
        collected_field_names = [field_names.get(f, f) for f in collected_fields]
        failed_field_names = [field_names.get(f, f) for f in final_missing]
        
        self._log_collection_summary(collected_field_names, failed_field_names)
        
        # 使用 ConciseLogger 显示详细资料（如果提供了 gui_logger）
        # 注意：这里也需要显示，因为可能经过备选方案后才获取到完整数据
        if gui_logger:
            from .concise_logger import ConciseLogger
            import logging
            file_logger = logging.getLogger(__name__)
            # [2026-03-11] 优化日志：不输出到GUI，避免CMD显示过多日志
            concise = ConciseLogger("profile_reader", None, file_logger)
            
            # 显示成功消息
            if not final_missing:
                concise.success("资料获取完成")
            else:
                concise.success("资料获取完成（部分字段缺失）")
            
            # 添加分隔线
            if gui_logger:
                gui_logger.info("=" * 60)
            
            # 显示详细资料，每个字段单独一行，使用 → 前缀
            if best_result.get('nickname'):
                gui_logger.info(f"  → 昵称: {best_result['nickname']}")
            if best_result.get('user_id'):
                gui_logger.info(f"  → 用户ID: {best_result['user_id']}")
            if best_result.get('balance') is not None:
                gui_logger.info(f"  → 余额: {best_result['balance']:.2f}元")
            if best_result.get('points') is not None:
                gui_logger.info(f"  → 积分: {best_result['points']}")
            if best_result.get('vouchers') is not None:
                gui_logger.info(f"  → 抵扣券: {best_result['vouchers']}")
        
        return best_result
    
    def _log_collection_summary(self, collected_fields: List[str], failed_fields: List[str]):
        """记录数据收集总结
        
        Args:
            collected_fields: 成功收集的字段列表
            failed_fields: 收集失败的字段列表
        """
        # [2026-03-11] 优化日志：移除控制台输出
        if collected_fields:
            pass  # [2026-03-11] 优化日志：移除控制台输出
        
        if failed_fields:
            pass  # [2026-03-11] 优化日志：移除控制台输出
            pass  # [2026-03-11] 优化日志：移除控制台输出
    
    async def _recognize_regions(self, device_id: str, full_image: 'Image.Image') -> Dict[str, any]:
        """使用全屏OCR + 关键字定位识别余额、积分、抵扣券、优惠券
        
        不使用固定坐标，而是：
        1. 全屏OCR识别所有文本
        2. 查找"余额"、"积分"、"抵扣券"、"优惠券"等关键字
        3. 在关键字附近提取数字
        
        Args:
            device_id: 设备ID
            full_image: 完整截图的PIL Image对象
            
        Returns:
            dict: 识别结果
                - balance: float, 余额
                - points: int, 积分
                - vouchers: float, 抵扣券
        """
        result = {
            'balance': None,
            'points': None,
            'vouchers': None,
        }
        
        if not HAS_PIL or not self._ocr_pool:
            return result
        
        try:
            # [2026-03-11] 优化日志：移除控制台输出
            
            # 全屏OCR识别
            enhanced_image = enhance_for_ocr(full_image)
            ocr_result = await self._ocr_pool.recognize(enhanced_image, timeout=5.0)
            
            # [2026-03-05] 修复数组比较错误：检查 texts 是否为 None 或长度为 0
            if not ocr_result or ocr_result.texts is None or len(ocr_result.texts) == 0:
                # [2026-03-11] 优化日志：移除控制台输出
                return result
            
            texts = ocr_result.texts
            boxes = ocr_result.boxes if hasattr(ocr_result, 'boxes') and ocr_result.boxes is not None else None
            
            # [2026-03-11] 优化日志：移除控制台输出
            # [2026-03-11] 优化日志：移除控制台输出
            
            # 如果有位置信息，使用位置辅助提取
            if boxes is not None and len(boxes) == len(texts):
                # [2026-03-11] 优化日志：移除控制台输出
                result = self._extract_values_with_positions(texts, boxes)
            else:
                # 没有位置信息，使用文本顺序提取
                # [2026-03-11] 优化日志：移除控制台输出
                result = self._extract_values_from_texts(texts)
            
            # 打印结果
            # [2026-03-11] 优化日志：移除控制台输出
            # [2026-03-11] 优化日志：移除控制台输出
            # [2026-03-11] 优化日志：移除控制台输出
            # [2026-03-11] 优化日志：移除控制台输出
            
            if result['balance'] is not None:
                pass  # [2026-03-11] 优化日志：移除控制台输出
            if result['points'] is not None:
                pass  # [2026-03-11] 优化日志：移除控制台输出
            if result['vouchers'] is not None:
                pass  # [2026-03-11] 优化日志：移除控制台输出
            
            return result
            
        except Exception as e:
            # [2026-03-11] 优化日志：移除控制台输出
            return result
    
    def _extract_values_with_positions(self, texts: List[str], boxes: List) -> Dict[str, any]:
        """使用位置信息提取数值
        
        策略：
        1. 找到"余额"、"积分"等关键字的位置
        2. 在关键字右侧或下方查找数字
        
        Args:
            texts: OCR文本列表
            boxes: OCR文本框位置列表
            
        Returns:
            dict: 提取的数值
        """
        result = {
            'balance': None,
            'points': None,
            'vouchers': None,
        }
        
        # [2026-03-11] 优化日志：移除控制台输出
        
        # 构建文本-位置映射
        text_positions = []
        for i, (text, box) in enumerate(zip(texts, boxes)):
            # 计算文本框中心点
            box_flat = box.flatten().tolist() if hasattr(box, 'flatten') else box
            x1, y1 = box_flat[0], box_flat[1]
            x2, y2 = box_flat[4], box_flat[5]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            text_positions.append({
                'index': i,
                'text': text,
                'center_x': center_x,
                'center_y': center_y,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2
            })
        
        # 先尝试检测网格布局（4个关键字在同一行，4个数值在同一行）
        keywords_info = {
            'balance': ['余额', '账户余额'],
            'points': ['积分', '我的积分'],
            'vouchers': ['抵扣券', '抵扣', '代金券'],
        }
        
        # 查找所有关键字的位置
        found_keywords = []
        for field, keyword_list in keywords_info.items():
            for keyword in keyword_list:
                for pos in text_positions:
                    if keyword in pos['text']:
                        found_keywords.append({
                            'field': field,
                            'keyword': keyword,
                            'pos': pos
                        })
                        break
                if found_keywords and found_keywords[-1]['field'] == field:
                    break
        
        # 检查是否是网格布局：所有关键字的y坐标相近（±10px）
        if len(found_keywords) >= 3:
            keyword_y_values = [kw['pos']['center_y'] for kw in found_keywords]
            y_min, y_max = min(keyword_y_values), max(keyword_y_values)
            
            if y_max - y_min < 10:
                pass  # [2026-03-11] 优化日志：移除控制台输出
                
                # 网格布局：按列匹配
                # 找出所有数值，按y坐标分组
                number_positions = []
                for pos in text_positions:
                    match = re.search(r'(\d+\.?\d*)', pos['text'].strip())
                    if match:
                        try:
                            value = float(match.group(1))
                            number_positions.append({
                                'value': value,
                                'text': pos['text'],
                                'x': pos['center_x'],
                                'y': pos['center_y']
                            })
                        except ValueError:
                            pass
                
                # 找出在关键字上方的数值行（y < keyword_y - 10）
                keyword_y = keyword_y_values[0]
                above_numbers = [n for n in number_positions if n['y'] < keyword_y - 10]
                
                # [2026-03-11] 优化日志：移除控制台输出
                
                if above_numbers:
                    # 找出最接近关键字的那一行数值
                    # 按y坐标分组（±5px为同一行）
                    y_groups = {}
                    for num in above_numbers:
                        # 找到最接近的组
                        found_group = False
                        for group_y in y_groups:
                            if abs(num['y'] - group_y) < 5:
                                y_groups[group_y].append(num)
                                found_group = True
                                break
                        if not found_group:
                            y_groups[num['y']] = [num]
                    
                    # 选择最接近关键字的组（y坐标最大的组）
                    closest_y = max(y_groups.keys())
                    closest_numbers = y_groups[closest_y]
                    
                    # [2026-03-11] 优化日志：移除控制台输出
                    
                    if len(closest_numbers) >= 3:
                        pass  # [2026-03-11] 优化日志：移除控制台输出
                        
                        # 按列匹配：为每个关键字找到x坐标最接近的数值
                        for kw in found_keywords:
                            field = kw['field']
                            kw_x = kw['pos']['center_x']
                            
                            # 找到x坐标最接近的数值
                            best_match = None
                            best_distance = float('inf')
                            
                            for num in closest_numbers:
                                distance = abs(num['x'] - kw_x)
                                if distance < best_distance:
                                    best_distance = distance
                                    best_match = num
                            
                            if best_match:
                                result[field] = best_match['value']
                                # [2026-03-11] 优化日志：移除控制台输出
                        
                        # [2026-03-11] 优化日志：移除控制台输出
                        # [2026-03-11] 优化日志：移除控制台输出
                        # [2026-03-11] 优化日志：移除控制台输出
                        # [2026-03-11] 优化日志：移除控制台输出
                        
                        return result
        
        # 如果不是网格布局，使用原来的逻辑
        # [2026-03-11] 优化日志：移除控制台输出
        
        # 查找关键字并提取附近的数字
        keywords = {
            'balance': ['余额', '账户余额'],
            'points': ['积分', '我的积分'],
            'vouchers': ['抵扣券', '抵扣', '代金券'],
        }
        
        for field, keyword_list in keywords.items():
            # [2026-03-11] 优化日志：移除控制台输出
            
            for keyword in keyword_list:
                # 查找关键字
                keyword_pos = None
                for pos in text_positions:
                    if keyword in pos['text']:
                        keyword_pos = pos
                        # [2026-03-11] 优化日志：移除控制台输出
                        break
                
                if keyword_pos:
                    # 在关键字右侧或上方查找数字
                    # 右侧：x > keyword_x, y 相近（±20px）
                    # 上方：y < keyword_y, x 相近（±40px）
                    candidates = []
                    
                    for pos in text_positions:
                        if pos['index'] == keyword_pos['index']:
                            continue
                        
                        # 提取数字
                        match = re.search(r'(\d+\.?\d*)', pos['text'].strip())
                        if not match:
                            continue
                        
                        try:
                            value = float(match.group(1))
                        except ValueError:
                            continue
                        
                        # 检查位置关系 - 使用更严格的垂直对齐要求
                        # 右侧：x > keyword_x + 20, y 几乎相同（±20px，更严格）
                        y_diff = abs(pos['center_y'] - keyword_pos['center_y'])
                        is_right = (pos['center_x'] > keyword_pos['center_x'] + 20 and y_diff < 20)
                        
                        # 上方：y < keyword_y - 10, x 几乎相同（±40px，稍微放宽）
                        x_diff = abs(pos['center_x'] - keyword_pos['center_x'])
                        is_above = (pos['center_y'] < keyword_pos['center_y'] - 10 and x_diff < 40)
                        
                        if is_right or is_above:
                            # 计算综合得分：对齐程度 + 距离
                            # 对齐程度越好，得分越低（越优先）
                            if is_right:
                                # 右侧：优先考虑垂直对齐程度，然后是水平距离
                                alignment_score = y_diff * 10  # 垂直偏差的权重
                                distance_score = abs(pos['center_x'] - keyword_pos['center_x'])
                                score = alignment_score + distance_score
                                direction = "右侧"
                            else:
                                # 上方：优先考虑水平对齐程度，然后是垂直距离
                                alignment_score = x_diff * 10  # 水平偏差的权重
                                distance_score = abs(pos['center_y'] - keyword_pos['center_y'])
                                score = alignment_score + distance_score
                                direction = "上方"
                            
                            # [2026-03-11] 优化日志：移除控制台输出
                            
                            candidates.append({
                                'value': value,
                                'score': score,
                                'text': pos['text'],
                                'direction': direction
                            })
                    
                    # 选择得分最低的候选（对齐最好且距离最近）
                    if candidates:
                        candidates.sort(key=lambda x: x['score'])
                        selected = candidates[0]
                        result[field] = selected['value']
                        # [2026-03-11] 优化日志：移除控制台输出
                        break  # 找到就跳出keyword循环
                    else:
                        pass  # [2026-03-11] 优化日志：移除控制台输出
        
        # [2026-03-11] 优化日志：移除控制台输出
        # [2026-03-11] 优化日志：移除控制台输出
        # [2026-03-11] 优化日志：移除控制台输出
        # [2026-03-11] 优化日志：移除控制台输出
        
        return result
    
    def _extract_values_from_texts(self, texts: List[str]) -> Dict[str, any]:
        """从文本列表中提取数值（无位置信息）
        
        策略：
        1. 找到"余额"、"积分"等关键字
        2. 在关键字后面的几个文本中查找数字
        
        Args:
            texts: OCR文本列表
            
        Returns:
            dict: 提取的数值
        """
        result = {
            'balance': None,
            'points': None,
            'vouchers': None,
        }
        
        keywords = {
            'balance': ['余额', '账户余额'],
            'points': ['积分', '我的积分'],
            'vouchers': ['抵扣券', '抵扣', '代金券'],
        }
        
        for field, keyword_list in keywords.items():
            for keyword in keyword_list:
                # 查找关键字
                for i, text in enumerate(texts):
                    if keyword in text:
                        # 在后面的3个文本中查找数字
                        for j in range(i, min(i + 4, len(texts))):
                            match = re.search(r'(\d+\.?\d*)', texts[j].strip())
                            if match:
                                try:
                                    value = float(match.group(1))
                                    # 合理性检查
                                    if field == 'balance' and 0 <= value <= 10000:
                                        result[field] = value
                                        break
                                    elif field == 'points' and 0 <= value <= 100000:
                                        result[field] = int(value)
                                        break
                                    elif field == 'vouchers' and 0 <= value <= 1000:
                                        result[field] = value
                                        break
                                except ValueError:
                                    pass
                        
                        if result[field] is not None:
                            break  # 找到就跳出keyword循环
                
                if result[field] is not None:
                    break  # 找到就跳出keyword_list循环
        
        return result
    
    async def get_full_profile_parallel(self, device_id: str, account: Optional[str] = None) -> Dict[str, any]:
        """获取完整的个人资料信息(并行版本，实际上是 get_full_profile 的别名)
        
        为了兼容性保留此方法，实际调用 get_full_profile
        
        Args:
            device_id: 设备ID
            account: 登录账号(可选)，用于提取手机号
            
        Returns:
            dict: 完整个人资料
        """
        return await self.get_full_profile(device_id, account)
    
    async def _extract_nickname_from_region(self, device_id: str, image: 'Image.Image') -> Optional[str]:
        """[2026-03-06] 从指定区域提取昵称（使用固定坐标）
        
        直接裁剪昵称区域进行OCR识别，避免全屏OCR的干扰
        
        Args:
            device_id: 设备ID
            image: 截图图像
            
        Returns:
            str: 昵称，未找到返回 None
        """
        if not HAS_PIL or not self._ocr_pool:
            return None
        
        try:
            # 使用定义好的昵称区域坐标
            x1, y1, x2, y2 = self.REGIONS['nickname']
            # [2026-03-11] 优化日志：移除控制台输出
            
            # 裁剪昵称区域
            nickname_region = image.crop((x1, y1, x2, y2))
            
            # 增强图像
            enhanced_region = enhance_for_ocr(nickname_region)
            
            # OCR识别
            ocr_result = await self._ocr_pool.recognize(enhanced_region, timeout=5.0)
            
            if not ocr_result or ocr_result.texts is None or len(ocr_result.texts) == 0:
                # [2026-03-11] 优化日志：移除控制台输出
                return None
            
            # [2026-03-11] 优化日志：移除控制台输出
            
            # 排除关键字
            exclude_keywords = [
                "溪盟山泉", "溪盟", "山泉", "溪", "西",  # 品牌名称及其单字
                "福利",  # 页面固定文字
                "ID", "id", "手机", "余额", "积分", 
                "抵扣券", "优惠券", "我的", "设置", "首页", "分类",
                "商城", "订单", "查看", "待付款", "待发货", "待收货", "待评价",
                "元", "张", "次"
            ]
            
            # 会员标签关键字
            member_keywords = [
                "钻石会员", "黄金会员", "白金会员", "铂金会员",
                "普通会员", "初级会员", "银牌会员",
                "VIP会员", "SVIP", "VIP",
                "vip会员", "vip", "Vip",
                "会员"
            ]
            
            # 遍历识别到的文本，找到最合适的昵称
            for text in ocr_result.texts:
                text = text.strip()
                
                if not text:
                    continue
                
                # 跳过纯数字
                if text.isdigit():
                    continue
                
                # 跳过时间格式
                if re.match(r'\d+:\d+', text):
                    continue
                
                # 跳过包含冒号的文本
                if ':' in text or '：' in text:
                    continue
                
                # 处理会员标签
                nickname_candidate = text
                for member_kw in member_keywords:
                    if member_kw in text:
                        nickname_candidate = text.split(member_kw)[0].strip()
                        # [2026-03-11] 优化日志：移除控制台输出
                        break
                
                if not nickname_candidate:
                    continue
                
                # 检查排除关键字
                has_keyword = False
                for kw in exclude_keywords:
                    if kw in nickname_candidate:
                        has_keyword = True
                        # [2026-03-11] 优化日志：移除控制台输出
                        break
                if has_keyword:
                    continue
                
                # 长度检查
                text_len = len(nickname_candidate)
                if 1 <= text_len <= 20:
                    # 单字检查
                    if text_len == 1:
                        single_char_exclude = ['我', '的', '首', '页', '设', '置']
                        if nickname_candidate in single_char_exclude:
                            # [2026-03-11] 优化日志：移除控制台输出
                            continue
                    
                    # [2026-03-11] 优化日志：移除控制台输出
                    return nickname_candidate
            
            # [2026-03-11] 优化日志：移除控制台输出
            return None
            
        except Exception as e:
            # [2026-03-11] 优化日志：移除控制台输出
            return None
    
    def _extract_nickname(self, texts: List[str]) -> Optional[str]:
        """从OCR文本中提取昵称
        
        改进策略：基于ID的相对位置提取昵称
        - 昵称通常在ID的上方
        - 先找到ID，然后在ID上方的文本中查找昵称
        
        Args:
            texts: OCR识别的文本列表
            
        Returns:
            str: 昵称，未找到返回 None
        """
        # [2026-03-11] 优化日志：移除控制台输出
        # [2026-03-11] 优化日志：移除控制台输出
        
        # 策略1: 基于ID位置提取昵称
        # 先找到ID的位置
        id_index = -1
        for i, text in enumerate(texts):
            text_no_space = text.replace(" ", "")
            if "ID" in text_no_space or "id" in text_no_space.lower():
                # 确认是用户ID（包含数字）
                if re.search(r'(?:用户)?[Ii][Dd][:：]?(\d+)', text_no_space):
                    id_index = i
                    # [2026-03-11] 优化日志：移除控制台输出
                    break
        
        if id_index >= 0:
            # [2026-03-05] 修复昵称识别：扩大检查范围到ID之前的5个文本
            # 在ID之前的文本中查找昵称（通常在ID的前1-5个位置）
            # [2026-03-11] 优化日志：移除控制台输出
            
            # 会员标签关键字
            member_keywords = [
                "钻石会员", "黄金会员", "白金会员", "铂金会员",
                "普通会员", "初级会员", "银牌会员",
                "VIP会员", "SVIP", "VIP",
                "vip会员", "vip", "Vip",
                "会员"
            ]
            
            # [2026-03-05] 修复昵称识别：添加品牌名称过滤
            # 排除关键字
            # [2026-03-06] 添加"福利"关键字：排除页面上的"福利"文字
            # [2026-03-06] 添加"溪"、"西"单字：OCR可能把"溪盟山泉"识别成单字
            exclude_keywords = [
                "溪盟山泉", "溪盟", "山泉", "溪", "西",  # 品牌名称及其单字
                "福利",  # 页面固定文字
                "ID", "id", "手机", "余额", "积分", 
                "抵扣券", "优惠券", "抵扣券", "我的", "设置", "首页", "分类",
                "商城", "订单", "查看", "待付款", "待发货", "待收货", "待评价",
                "元", "张", "次"
            ]
            
            # 获取ID的位置（用于过滤）
            id_box = None
            if hasattr(self, '_last_ocr_result') and self._last_ocr_result is not None:
                if hasattr(self._last_ocr_result, 'boxes') and self._last_ocr_result.boxes is not None:
                    try:
                        id_box = self._last_ocr_result.boxes[id_index]
                    except:
                        pass
            
            # [2026-03-05] 修复昵称识别：扩大检查范围从3个文本到5个文本
            # 检查ID之前的5个文本（覆盖更多可能的昵称位置）
            for i in range(max(0, id_index - 5), id_index):
                text = texts[i].strip()
                
                # 获取当前文本的位置
                text_box = None
                if hasattr(self, '_last_ocr_result') and self._last_ocr_result is not None:
                    if hasattr(self._last_ocr_result, 'boxes') and self._last_ocr_result.boxes is not None:
                        try:
                            text_box = self._last_ocr_result.boxes[i]
                        except:
                            pass
                
                # 位置过滤：排除右上角的文本（x > 400，通常是状态栏图标）
                if text_box is not None:
                    x_min = min(text_box[0][0], text_box[1][0], text_box[2][0], text_box[3][0])
                    if x_min > 400:
                        # [2026-03-11] 优化日志：移除控制台输出
                        continue
                
                # [2026-03-11] 优化日志：移除控制台输出
                
                # 跳过空文本
                if not text:
                    # [2026-03-11] 优化日志：移除控制台输出
                    continue
                
                # 跳过纯数字
                if text.isdigit():
                    # [2026-03-11] 优化日志：移除控制台输出
                    continue
                
                # 跳过数字+空格组合（如"1 0"、"10"等）
                text_no_space = text.replace(" ", "").replace("\t", "")
                if text_no_space.isdigit() and len(text_no_space) <= 3:
                    # [2026-03-11] 优化日志：移除控制台输出
                    continue
                
                # 跳过时间格式
                if re.match(r'\d+:\d+', text):
                    # [2026-03-11] 优化日志：移除控制台输出
                    continue
                
                # 跳过包含冒号的文本
                if ':' in text or '：' in text:
                    # [2026-03-11] 优化日志：移除控制台输出
                    continue
                
                # 处理会员标签
                nickname_candidate = text
                for member_kw in member_keywords:
                    if member_kw in text:
                        nickname_candidate = text.split(member_kw)[0].strip()
                        # [2026-03-11] 优化日志：移除控制台输出
                        break
                
                if not nickname_candidate:
                    # [2026-03-11] 优化日志：移除控制台输出
                    continue
                
                # 检查排除关键字
                has_keyword = False
                for kw in exclude_keywords:
                    if kw in nickname_candidate:
                        has_keyword = True
                        # [2026-03-11] 优化日志：移除控制台输出
                        break
                if has_keyword:
                    continue
                
                # 长度检查
                text_len = len(nickname_candidate)
                if 1 <= text_len <= 20:
                    # 单字检查
                    if text_len == 1:
                        single_char_exclude = ['我', '的', '首', '页', '设', '置']
                        if nickname_candidate in single_char_exclude:
                            # [2026-03-11] 优化日志：移除控制台输出
                            continue
                    
                    # [2026-03-11] 优化日志：移除控制台输出
                    return nickname_candidate
                else:
                    pass  # [2026-03-11] 优化日志：移除控制台输出
        
        # 策略2: 查找"昵称"关键字（备选）
        # [2026-03-11] 优化日志：移除控制台输出
        for text in texts:
            if "昵称" in text:
                match = re.search(r'昵称[:：\s]+(.+)', text)
                if match:
                    nickname = match.group(1).strip()
                    if nickname:
                        # [2026-03-11] 优化日志：移除控制台输出
                        return nickname
                
                if text.startswith("昵称"):
                    nickname = text[2:].strip()
                    if nickname:
                        # [2026-03-11] 优化日志：移除控制台输出
                        return nickname
        
        # 策略3: 在前10个文本中查找（最后的备选）
        # [2026-03-11] 优化日志：移除控制台输出
        
        # [2026-03-05] 修复昵称识别：添加品牌名称过滤
        # [2026-03-06] 添加"溪"、"西"单字：OCR可能把"溪盟山泉"识别成单字
        exclude_keywords = [
            "溪盟山泉", "溪盟", "山泉", "溪", "西",  # 品牌名称及其单字
            "福利",  # 页面固定文字
            "ID", "id", "手机", "余额", "积分", 
            "抵扣券", "优惠券", "抵扣券", "我的", "设置", "首页", "分类",
            "商城", "订单", "查看", "待付款", "待发货", "待收货", "待评价",
            "元", "张", "次"
        ]
        
        member_keywords = [
            "钻石会员", "黄金会员", "白金会员", "铂金会员",
            "普通会员", "初级会员", "银牌会员",
            "VIP会员", "SVIP", "VIP",
            "vip会员", "vip", "Vip",
            "会员"
        ]
        
        for i, text in enumerate(texts[:10]):
            text = text.strip()
            
            if not text or text.isdigit():
                continue
            
            if re.match(r'\d+:\d+', text):
                continue
            
            if ':' in text or '：' in text:
                continue
            
            nickname_candidate = text
            for member_kw in member_keywords:
                if member_kw in text:
                    nickname_candidate = text.split(member_kw)[0].strip()
                    break
            
            if not nickname_candidate:
                continue
            
            has_keyword = False
            for kw in exclude_keywords:
                if kw in nickname_candidate:
                    has_keyword = True
                    break
            if has_keyword:
                continue
            
            text_len = len(nickname_candidate)
            if 1 <= text_len <= 20:
                if text_len == 1:
                    single_char_exclude = ['我', '的', '首', '页', '设', '置']
                    if nickname_candidate in single_char_exclude:
                        continue
                
                # [2026-03-11] 优化日志：移除控制台输出
                return nickname_candidate
        
        # [2026-03-11] 优化日志：移除控制台输出
        return None
    
    def _extract_user_id(self, texts: List[str]) -> Optional[str]:
        """从OCR文本中提取用户ID
        
        常见模式：
        - "ID: 123456"
        - "用户ID: 123456"
        - "ID 123456"
        - "123456" (纯数字，6位以上)
        
        Args:
            texts: OCR识别的文本列表
            
        Returns:
            str: 用户ID，未找到返回 None
        """
        # [2026-03-05] 修复原因：优化用户ID提取逻辑，提高识别成功率
        for text in texts:
            # 移除空格
            text_no_space = text.replace(" ", "")
            
            # 模式1: "ID:数字" 或 "用户ID:数字"
            if "ID" in text_no_space or "id" in text_no_space.lower():
                # 提取数字部分
                match = re.search(r'(?:用户)?[Ii][Dd][:：]?(\d+)', text_no_space)
                if match:
                    return match.group(1)
            
            # [2026-03-05] 新增：模式2: 纯数字（6位以上）
            # 如果文本是纯数字且长度在6-10位之间，可能是用户ID
            if text_no_space.isdigit() and 6 <= len(text_no_space) <= 10:
                return text_no_space
        
        return None
    
    def _extract_phone(self, texts: List[str]) -> Optional[str]:
        """从OCR文本中提取手机号
        
        常见模式：
        - "手机号: 138****1234"
        - "138****1234"
        - "13812341234"（完整手机号）
        
        Args:
            texts: OCR识别的文本列表
            
        Returns:
            str: 手机号，未找到返回 None
        """
        for text in texts:
            # 移除空格
            text_no_space = text.replace(" ", "")
            
            # 模式1: 掩码格式 "138****1234"
            match = re.search(r'(\d{3}\*{4}\d{4})', text_no_space)
            if match:
                return match.group(1)
            
            # 模式2: 完整手机号 "13812341234"
            match = re.search(r'(1[3-9]\d{9})', text_no_space)
            if match:
                return match.group(1)
            
            # 模式3: "手机号:XXX"
            if "手机" in text_no_space:
                match = re.search(r'手机号?[:：]?(\d{3}\*{4}\d{4}|1[3-9]\d{9})', text_no_space)
                if match:
                    return match.group(1)
        
        return None
    
    def _extract_phone_from_account(self, account: str) -> Optional[str]:
        """从登录账号中提取手机号
        
        账号格式通常是: 手机号----密码
        例如: 15766121960----hye19911206
        
        Args:
            account: 登录账号字符串
            
        Returns:
            str: 手机号，未找到返回 None
        """
        if not account:
            return None
        
        # 提取----之前的部分
        if '----' in account:
            phone = account.split('----')[0].strip()
        else:
            phone = account.strip()
        
        # 验证是否是有效的手机号（11位数字，以1开头）
        if phone and len(phone) == 11 and phone.isdigit() and phone.startswith('1'):
            return phone
        
        return None
    
    def _extract_balance(self, texts: List[str]) -> Optional[float]:
        """从OCR文本中提取余额
        
        常见模式：
        - "余额: 16.26"
        - "余额" 和 "16.26" 分开识别（数值通常在标签之前）
        - "16.26元"
        
        策略：余额通常是第一个数值（最远离标签的）
        
        Args:
            texts: OCR识别的文本列表
            
        Returns:
            float: 余额，未找到返回 None
        """
        # [2026-03-11] 优化日志：移除控制台输出
        
        # 策略1: 查找包含"余额"的文本
        for text in texts:
            text_no_space = text.replace(" ", "")
            
            # 模式1: "余额:数字"
            if "余额" in text_no_space:
                match = re.search(r'余额[:：]?(\d+\.?\d*)', text_no_space)
                if match:
                    try:
                        balance = float(match.group(1))
                        # [2026-03-11] 优化日志：移除控制台输出
                        return balance
                    except ValueError:
                        pass
        
        # 策略2: 查找"余额"标签，然后在其前5个文本块中查找数值
        # 余额通常是最远的那个数值（第一个数值）
        for i, text in enumerate(texts):
            if "余额" in text:
                # [2026-03-11] 优化日志：移除控制台输出
                candidates = []
                
                # 检查前面的文本块，扩大到5个，收集所有候选值
                for j in range(i-1, max(0, i-6), -1):
                    # 尝试提取数字（支持小数）
                    match = re.search(r'^(\d+\.?\d*)$', texts[j].strip())
                    if match:
                        try:
                            balance = float(match.group(1))
                            # 合理性检查：余额通常在0-10000之间
                            if 0 <= balance <= 10000:
                                candidates.append((j, balance))  # 保存索引和值
                                # [2026-03-11] 优化日志：移除控制台输出
                        except ValueError:
                            pass
                
                # [2026-03-11] 优化日志：移除控制台输出
                
                # 优先选择非零值，如果有多个非零值，选择最远的（索引最小的）
                # 因为页面布局通常是：余额、积分、抵扣券，余额离标签最远
                non_zero = [(idx, val) for idx, val in candidates if val > 0]
                if non_zero:
                    # 按索引排序，选择最远的（索引最小的）
                    non_zero.sort(key=lambda x: x[0])
                    # [2026-03-11] 优化日志：移除控制台输出
                    return non_zero[0][1]
                elif candidates:
                    # 如果都是0，返回最远的
                    candidates.sort(key=lambda x: x[0])
                    # [2026-03-11] 优化日志：移除控制台输出
                    return candidates[0][1]
        
        # 策略3: 查找带"元"的数字（但不包含"余额"）
        for text in texts:
            text_no_space = text.replace(" ", "")
            if "元" in text_no_space and "余额" not in text_no_space:
                match = re.search(r'(\d+\.?\d*)元', text_no_space)
                if match:
                    try:
                        balance = float(match.group(1))
                        if 0 <= balance <= 10000:
                            # [2026-03-11] 优化日志：移除控制台输出
                            return balance
                    except ValueError:
                        pass
        
        # [2026-03-11] 优化日志：移除控制台输出
        return None
    
    def _parse_points(self, texts: list) -> Optional[int]:
        """从OCR文本中解析积分
        
        常见模式：
        - "积分: 1234"
        - "积分" 和 "1234" 分开识别（数值通常在标签之前）
        - "1234积分"
        - "0.00" 或 "1.00" (可能被识别为小数，需要转换为整数)
        
        布局：余额值 | 积分值 | 抵扣券值 | "余额" | "抵扣券" | "优惠券"
        注意："积分"标签可能没有被OCR识别出来
        
        Args:
            texts: OCR识别的文本列表
            
        Returns:
            int: 积分，未找到返回 None
        """
        # 策略1: 查找包含"积分"的文本
        for text in texts:
            text_no_space = text.replace(" ", "")
            
            if "积分" in text_no_space:
                # 模式1: "积分:数字"
                match = re.search(r'积分[:：]?(\d+\.?\d*)', text_no_space)
                if match:
                    try:
                        # 转换为整数（去掉小数部分）
                        return int(float(match.group(1)))
                    except ValueError:
                        pass
                
                # 模式2: "数字积分"
                match = re.search(r'(\d+\.?\d*)积分', text_no_space)
                if match:
                    try:
                        return int(float(match.group(1)))
                    except ValueError:
                        pass
        
        # 策略2: 查找"积分"标签，然后只在其前面的文本块中查找数值
        for i, text in enumerate(texts):
            if "积分" in text:
                # 只检查前面的文本块（最多2个，避免跨到余额字段）
                for j in range(i-1, max(0, i-3), -1):
                    text_j = texts[j].strip()
                    
                    # 跳过其他标签（避免误识别）
                    if any(keyword in text_j for keyword in ["余额", "抵扣券", "优惠券", "抵扣劵", "优惠劵"]):
                        continue
                    
                    # 尝试提取纯数字或小数（必须是完整的数字，不能包含其他字符）
                    match = re.search(r'^(\d+\.?\d*)$', text_j)
                    if match:
                        try:
                            # 积分可能显示为小数（如 0.00 或 1.00），转换为整数
                            points = int(float(match.group(1)))
                            # 合理性检查：积分通常在0-100000之间
                            if 0 <= points <= 100000:
                                # 找到第一个符合条件的值就返回（最近的）
                                return points
                        except ValueError:
                            pass
        
        # 策略3: 如果没有找到"积分"标签，使用位置推断
        # 布局：余额值 | 积分值 | 抵扣券值 | "余额" | "抵扣券" | "优惠券"
        # 找到"余额"标签，它前面第2个数值就是积分
        for i, text in enumerate(texts):
            if "余额" in text:
                # 收集前面的所有数值
                values = []
                for j in range(i-1, max(0, i-5), -1):
                    text_j = texts[j].strip()
                    # 尝试提取纯数字或小数
                    match = re.search(r'^(\d+\.?\d*)$', text_j)
                    if match:
                        try:
                            value = float(match.group(1))
                            values.append((j, value))
                        except ValueError:
                            pass
                
                # 如果找到至少2个数值，第2个（从后往前数）就是积分
                if len(values) >= 2:
                    # values是从近到远排列的，所以values[1]是第2个数值（积分）
                    points = int(values[1][1])
                    # 合理性检查
                    if 0 <= points <= 100000:
                        return points
        
        return None
    
    def _parse_vouchers(self, texts: list) -> Optional[float]:
        """从OCR文本中解析抵扣券数量/金额
        
        常见模式：
        - "抵扣券: 5"
        - "抵扣券" 和 "5" 分开识别（数值通常在标签之前）
        - "5张抵扣券"
        - "5.97" (可能是金额)
        
        布局：余额值 | 积分值 | 抵扣券值 | "余额" | "抵扣券" | "优惠券"
        
        注意：返回浮点数以保留原始精度
        
        Args:
            texts: OCR识别的文本列表
            
        Returns:
            float: 抵扣券数量/金额，未找到返回 None
        """
        # 策略1: 查找包含"抵扣券"的文本
        for text in texts:
            text_no_space = text.replace(" ", "")
            
            if "抵扣券" in text_no_space:
                # 模式1: "抵扣券:数字"
                match = re.search(r'抵扣券[:：]?(\d+\.?\d*)', text_no_space)
                if match:
                    try:
                        return float(match.group(1))
                    except ValueError:
                        pass
                
                # 模式2: "数字张抵扣券"
                match = re.search(r'(\d+\.?\d*)张?抵扣券', text_no_space)
                if match:
                    try:
                        return float(match.group(1))
                    except ValueError:
                        pass
        
        # 策略2: 查找"抵扣券"标签，然后只在其前面的文本块中查找数值
        for i, text in enumerate(texts):
            if "抵扣券" in text:
                # 只检查前面的文本块（最多2个，避免跨到积分字段）
                for j in range(i-1, max(0, i-3), -1):
                    text_j = texts[j].strip()
                    
                    # 跳过其他标签（避免误识别）
                    if any(keyword in text_j for keyword in ["余额", "积分", "优惠券", "优惠劵"]):
                        continue
                    
                    # 尝试提取纯数字或小数（必须是完整的数字，不能包含其他字符）
                    match = re.search(r'^(\d+\.?\d*)$', text_j)
                    if match:
                        try:
                            value = float(match.group(1))
                            # 合理性检查：抵扣券通常在0-100之间
                            if 0 <= value <= 100:
                                # 找到第一个符合条件的值就返回（最近的）
                                return value
                        except ValueError:
                            pass
        
        # 策略3: 如果没有找到"抵扣券"标签，使用位置推断
        # 布局：余额值 | 积分值 | 抵扣券值 | "余额" | "抵扣券" | "优惠券"
        # 找到"余额"标签，它前面第3个数值就是抵扣券
        for i, text in enumerate(texts):
            if "余额" in text:
                # 收集前面的所有数值
                values = []
                for j in range(i-1, max(0, i-5), -1):
                    text_j = texts[j].strip()
                    # 尝试提取纯数字或小数
                    match = re.search(r'^(\d+\.?\d*)$', text_j)
                    if match:
                        try:
                            value = float(match.group(1))
                            values.append((j, value))
                        except ValueError:
                            pass
                
                # 如果找到至少3个数值，第3个（从后往前数）就是抵扣券
                if len(values) >= 3:
                    # values是从近到远排列的，所以values[2]是第3个数值（抵扣券）
                    vouchers = values[2][1]
                    # 合理性检查
                    if 0 <= vouchers <= 100:
                        return vouchers
        
        return None
    
    
    # [2026-03-01] 删除_parse_coupons方法：个人页已经没有优惠券了
    
    def _parse_draw_times(self, texts: list) -> Optional[int]:
        """从OCR文本中解析总抽奖次数
        
        常见模式：
        - "抽奖次数: 10"
        - "剩余次数: 10"
        - "可抽奖: 10次"
        
        Args:
            texts: OCR识别的文本列表
            
        Returns:
            int: 总抽奖次数，未找到返回 None
        """
        for text in texts:
            # 移除空格
            text = text.replace(" ", "")
            
            # 模式1: "抽奖次数:数字" 或 "剩余次数:数字"
            if "抽奖" in text or "次数" in text:
                # 提取数字部分
                match = re.search(r'(抽奖次数|剩余次数|可抽奖)[:：]?(\d+)', text)
                if match:
                    try:
                        return int(match.group(2))
                    except ValueError:
                        pass
        
        return None
    
    # ==================== 余额获取方法 ====================
    
    async def get_balance(self, device_id: str) -> Optional[float]:
        """获取余额（优化版：使用全屏OCR一次+位置匹配）
        
        Args:
            device_id: 设备ID
            
        Returns:
            float: 余额，失败返回 None
        """
        if not HAS_PIL or not self._ocr_pool:
            return None
        
        try:
            # 截图
            screenshot_data = await self.adb.screencap(device_id)
            if not screenshot_data:
                return None
            
            image = Image.open(BytesIO(screenshot_data))
            
            # [2026-03-02] 统一术语：优先使用YOLO识别器（全屏OCR优化）
            use_yolo_fallback = True  # 标记是否需要降级到YOLO检测器
            
            if self._integrated_detector:
                detection_result = await self._integrated_detector.detect_page(
                    device_id, 
                    use_cache=False, 
                    detect_elements=True
                )
                
                # [2026-03-01] 修复：检查 elements 属性是否存在
                # [2026-03-05] 修复数组比较错误：使用 is not None 和 len() 检查
                if hasattr(detection_result, 'elements') and detection_result.elements is not None and len(detection_result.elements) > 0:
                    # 全屏OCR识别（只调用一次）
                    enhanced_image = enhance_for_ocr(image)
                    full_ocr_result = await self._ocr_pool.recognize(enhanced_image)
                    
                    if full_ocr_result and full_ocr_result.texts is not None and len(full_ocr_result.texts) > 0 and full_ocr_result.boxes is not None and len(full_ocr_result.boxes) > 0:
                        # 根据YOLO检测到的余额元素位置，从全屏OCR结果中匹配文本
                        for element in detection_result.elements:
                            if '余额' in element.class_name:
                                x1, y1, x2, y2 = element.bbox
                                
                                # 查找与元素位置重叠的OCR文本
                                matched_texts = []
                                for i, (text, box) in enumerate(zip(full_ocr_result.texts, full_ocr_result.boxes)):
                                    # 计算OCR文本框的中心点
                                    box_flat = box.flatten().tolist() if hasattr(box, 'flatten') else box
                                    ocr_x1, ocr_y1 = box_flat[0], box_flat[1]
                                    ocr_x2, ocr_y2 = box_flat[4], box_flat[5]
                                    ocr_center_x = (ocr_x1 + ocr_x2) / 2
                                    ocr_center_y = (ocr_y1 + ocr_y2) / 2
                                    
                                    # 检查OCR文本框是否在YOLO元素框内
                                    if x1 <= ocr_center_x <= x2 and y1 <= ocr_center_y <= y2:
                                        matched_texts.append(text)
                                
                                if matched_texts:
                                    # 合并所有匹配的文本
                                    combined_text = ' '.join(matched_texts)
                                    
                                    # 查找所有数字（包括小数）
                                    all_numbers = re.findall(r'(\d+\.?\d*)', combined_text)
                                    
                                    if all_numbers:
                                        # 转换为浮点数并选择最大的合理值
                                        valid_numbers = []
                                        for num_str in all_numbers:
                                            try:
                                                num = float(num_str)
                                                if 0.01 <= num <= 100000:
                                                    valid_numbers.append(num)
                                            except ValueError:
                                                continue
                                        
                                        if valid_numbers:
                                            use_yolo_fallback = False  # 成功获取余额，不需要降级
                                            return max(valid_numbers)
            
            # 降级：使用旧的YOLO检测器
            if use_yolo_fallback and self._yolo_detector:
                detections = await self._yolo_detector.detect(
                    device_id, 
                    'balance',
                    conf_threshold=0.3
                )
                
                if detections:
                    for det in detections:
                        if '余额' in det.class_name:
                            x1, y1, x2, y2 = det.bbox
                            region = image.crop((x1, y1, x2, y2))
                            region_enhanced = enhance_for_ocr(region)
                            ocr_result = await self._ocr_pool.recognize(region_enhanced, timeout=2.0)
                            
                            # [2026-03-05] 修复数组比较错误：检查 texts 是否为 None 并且长度大于 0
                            if ocr_result and ocr_result.texts is not None and len(ocr_result.texts) > 0:
                                for text in ocr_result.texts:
                                    match = re.search(r'(\d+\.?\d*)', text.strip())
                                    if match:
                                        try:
                                            balance = float(match.group(1))
                                            if 0 <= balance <= 10000:
                                                return balance
                                        except ValueError:
                                            pass
            
            # 最后降级：使用区域OCR
            region_results = await self._recognize_regions(device_id, image)
            return region_results.get('balance')
            
        except Exception as e:
            # [2026-03-11] 优化日志：移除控制台输出
            return None
    
    # ==================== 备选方案方法 ====================
    
    async def get_balance_fallback(self, device_id: str) -> Optional[float]:
        """备选方案：使用区域OCR获取余额
        
        策略：
        1. 对整个屏幕进行OCR
        2. 查找"余额"关键字
        3. 提取其附近的数字
        
        Args:
            device_id: 设备ID
            
        Returns:
            float: 余额，失败返回 None
        """
        if not HAS_PIL or not HAS_OCR:
            return None
        
        try:
            # 截图
            screenshot_data = await self.adb.screencap(device_id)
            if not screenshot_data:
                return None
            
            image = Image.open(BytesIO(screenshot_data))
            
            # OCR识别，超时10秒
            try:
                ocr_result = await asyncio.wait_for(
                    self._ocr_pool.recognize(image, timeout=10.0),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                return None
            
            # [2026-03-05] 修复数组比较错误：检查 texts 是否为 None 或长度为 0
            if not ocr_result or ocr_result.texts is None or len(ocr_result.texts) == 0:
                return None
            
            texts = list(ocr_result.texts)
            
            # 策略1: 查找"余额"关键字附近的数字
            for i, text in enumerate(texts):
                if "余额" in text:
                    # 检查前后的文本
                    for j in range(max(0, i-3), min(len(texts), i+4)):
                        if j != i:
                            # 尝试提取数字
                            match = re.search(r'(\d+\.?\d*)', texts[j])
                            if match:
                                try:
                                    balance = float(match.group(1))
                                    # 合理性检查：余额通常在0-10000之间
                                    if 0 <= balance <= 10000:
                                        return balance
                                except ValueError:
                                    pass
            
            # 策略2: 查找带"元"的数字
            for text in texts:
                if "元" in text and "余额" not in text:
                    match = re.search(r'(\d+\.?\d*)元', text)
                    if match:
                        try:
                            balance = float(match.group(1))
                            if 0 <= balance <= 10000:
                                return balance
                        except ValueError:
                            pass
            
            return None
            
        except Exception as e:
            # [2026-03-11] 优化日志：删除CMD输出
            pass
            return None
    
    async def get_user_id_fallback(self, device_id: str) -> Optional[str]:
        """备选方案：从固定位置提取用户ID
        
        策略：
        1. 对屏幕顶部30%区域进行OCR
        2. 查找"ID"关键字
        3. 提取其后的数字
        
        Args:
            device_id: 设备ID
            
        Returns:
            str: 用户ID，失败返回 None
        """
        if not HAS_PIL or not HAS_OCR:
            return None
        
        try:
            # 截图
            screenshot_data = await self.adb.screencap(device_id)
            if not screenshot_data:
                return None
            
            image = Image.open(BytesIO(screenshot_data))
            
            # 裁剪顶部30%区域
            width, height = image.size
            top_region = image.crop((0, 0, width, int(height * 0.3)))
            
            # OCR识别，超时10秒
            try:
                ocr_result = await asyncio.wait_for(
                    self._ocr_pool.recognize(top_region, timeout=10.0),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                return None
            
            # [2026-03-05] 修复数组比较错误：检查 texts 是否为 None 或长度为 0
            if not ocr_result or ocr_result.texts is None or len(ocr_result.texts) == 0:
                return None
            
            texts = list(ocr_result.texts)
            
            # 查找ID模式
            for text in texts:
                text_no_space = text.replace(" ", "")
                # 匹配 "ID:数字" 或 "用户ID:数字"
                match = re.search(r'(?:用户)?[Ii][Dd][:：]?(\d+)', text_no_space)
                if match:
                    user_id = match.group(1)
                    # 合理性检查：ID通常是6-12位数字
                    if 6 <= len(user_id) <= 12:
                        return user_id
            
            return None
            
        except Exception as e:
            # [2026-03-11] 优化日志：删除CMD输出
            pass
            return None
    
    async def get_nickname_fallback(self, device_id: str) -> Optional[str]:
        """备选方案：从顶部区域提取昵称
        
        策略：
        1. 对屏幕顶部20%区域进行OCR
        2. 查找"昵称"关键字
        3. 提取其后的文本
        4. 如果没有"昵称"关键字，返回顶部区域最长的非数字文本
        
        Args:
            device_id: 设备ID
            
        Returns:
            str: 昵称，失败返回 None
        """
        if not HAS_PIL or not HAS_OCR:
            return None
        
        try:
            # 截图
            screenshot_data = await self.adb.screencap(device_id)
            if not screenshot_data:
                return None
            
            image = Image.open(BytesIO(screenshot_data))
            
            # 裁剪顶部20%区域
            width, height = image.size
            top_region = image.crop((0, 0, width, int(height * 0.2)))
            
            # OCR识别，超时10秒
            try:
                ocr_result = await asyncio.wait_for(
                    self._ocr_pool.recognize(top_region, timeout=10.0),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                return None
            
            # [2026-03-05] 修复数组比较错误：检查 texts 是否为 None 或长度为 0
            if not ocr_result or ocr_result.texts is None or len(ocr_result.texts) == 0:
                return None
            
            texts = list(ocr_result.texts)
            
            # 策略1: 查找"昵称"关键字
            for text in texts:
                if "昵称" in text:
                    match = re.search(r'昵称[:：\s]+(.+)', text)
                    if match:
                        nickname = match.group(1).strip()
                        if nickname and len(nickname) <= 20:  # 昵称通常不超过20个字符
                            return nickname
            
            # 策略2: 返回最长的非数字、非关键字文本
            candidates = []
            # [2026-03-06] 添加"福利"关键字：排除页面上的"福利"文字
            # [2026-03-06] 添加"溪"、"西"单字：OCR可能把"溪盟山泉"识别成单字
            keywords = ["溪盟山泉", "溪盟", "山泉", "溪", "西", "福利", "ID", "id", "手机", "余额", "积分", "抵扣券", "优惠券", "我的", "设置"]
            for text in texts:
                # 过滤掉纯数字、包含关键字的文本
                if not text.isdigit() and not any(kw in text for kw in keywords):
                    # 过滤掉太短或太长的文本
                    if 2 <= len(text) <= 20:
                        candidates.append(text)
            
            if candidates:
                # 返回最长的候选
                return max(candidates, key=len)
            
            return None
            
        except Exception as e:
            # [2026-03-11] 优化日志：删除CMD输出
            pass
            return None
    
    async def get_phone_fallback(self, device_id: str) -> Optional[str]:
        """备选方案：从手机号区域提取
        
        策略：
        1. 对整个屏幕进行OCR
        2. 查找手机号模式(掩码或完整)
        3. 优先返回掩码格式
        
        Args:
            device_id: 设备ID
            
        Returns:
            str: 手机号，失败返回 None
        """
        if not HAS_PIL or not HAS_OCR:
            return None
        
        try:
            # 截图
            screenshot_data = await self.adb.screencap(device_id)
            if not screenshot_data:
                return None
            
            image = Image.open(BytesIO(screenshot_data))
            
            # OCR识别，超时10秒
            try:
                ocr_result = await asyncio.wait_for(
                    self._ocr_pool.recognize(image, timeout=10.0),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                return None
            
            # [2026-03-05] 修复数组比较错误：检查 texts 是否为 None 或长度为 0
            if not ocr_result or ocr_result.texts is None or len(ocr_result.texts) == 0:
                return None
            
            texts = list(ocr_result.texts)
            
            # 策略1: 查找掩码格式 "138****1234"
            for text in texts:
                text_no_space = text.replace(" ", "")
                match = re.search(r'(\d{3}\*{4}\d{4})', text_no_space)
                if match:
                    return match.group(1)
            
            # 策略2: 查找完整手机号 "13812341234"
            for text in texts:
                text_no_space = text.replace(" ", "")
                match = re.search(r'(1[3-9]\d{9})', text_no_space)
                if match:
                    return match.group(1)
            
            return None
            
        except Exception as e:
            # [2026-03-11] 优化日志：删除CMD输出
            pass
            return None
    
    async def get_points_fallback(self, device_id: str) -> Optional[int]:
        """备选方案：从积分区域提取
        
        策略：
        1. 对整个屏幕进行OCR
        2. 查找"积分"关键字
        3. 提取其附近的数字
        
        Args:
            device_id: 设备ID
            
        Returns:
            int: 积分，失败返回 None
        """
        if not HAS_PIL or not HAS_OCR:
            return None
        
        try:
            # 截图
            screenshot_data = await self.adb.screencap(device_id)
            if not screenshot_data:
                return None
            
            image = Image.open(BytesIO(screenshot_data))
            
            # OCR识别，超时10秒
            try:
                ocr_result = await asyncio.wait_for(
                    self._ocr_pool.recognize(image, timeout=10.0),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                return None
            
            # [2026-03-05] 修复数组比较错误：检查 texts 是否为 None 或长度为 0
            if not ocr_result or ocr_result.texts is None or len(ocr_result.texts) == 0:
                return None
            
            texts = list(ocr_result.texts)
            
            # 策略1: 查找"积分"关键字附近的数字
            for i, text in enumerate(texts):
                if "积分" in text:
                    # 检查同一文本中的数字
                    match = re.search(r'(\d+)积分', text)
                    if match:
                        try:
                            points = int(match.group(1))
                            # 合理性检查：积分通常在0-100000之间
                            if 0 <= points <= 100000:
                                return points
                        except ValueError:
                            pass
                    
                    match = re.search(r'积分[:：]?(\d+)', text)
                    if match:
                        try:
                            points = int(match.group(1))
                            if 0 <= points <= 100000:
                                return points
                        except ValueError:
                            pass
                    
                    # 检查前后的文本
                    for j in range(max(0, i-3), min(len(texts), i+4)):
                        if j != i and texts[j].isdigit():
                            try:
                                points = int(texts[j])
                                if 0 <= points <= 100000:
                                    return points
                            except ValueError:
                                pass
            
            return None
            
        except Exception as e:
            # [2026-03-11] 优化日志：删除CMD输出
            pass
            return None
    
    async def get_vouchers_fallback(self, device_id: str) -> Optional[float]:
        """备选方案：从抵扣券区域提取
        
        策略：
        1. 对整个屏幕进行OCR
        2. 查找"抵扣券"或"优惠券"关键字
        3. 提取其附近的数字（支持小数）
        
        Args:
            device_id: 设备ID
            
        Returns:
            float: 抵扣券数量/金额，失败返回 None
        """
        if not HAS_PIL or not HAS_OCR:
            return None
        
        try:
            # 截图
            screenshot_data = await self.adb.screencap(device_id)
            if not screenshot_data:
                return None
            
            image = Image.open(BytesIO(screenshot_data))
            
            # OCR识别，超时10秒
            try:
                ocr_result = await asyncio.wait_for(
                    self._ocr_pool.recognize(image, timeout=10.0),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                return None
            
            # [2026-03-05] 修复数组比较错误：检查 texts 是否为 None 或长度为 0
            if not ocr_result or ocr_result.texts is None or len(ocr_result.texts) == 0:
                return None
            
            texts = list(ocr_result.texts)
            
            # 策略1: 查找"抵扣券"或"优惠券"关键字附近的数字
            for i, text in enumerate(texts):
                if "抵扣券" in text or "优惠券" in text:
                    # 检查同一文本中的数字（支持小数）
                    match = re.search(r'(\d+\.?\d*)张?(抵扣券|优惠券)', text)
                    if match:
                        try:
                            vouchers = float(match.group(1))
                            # 合理性检查：抵扣券通常在0-100之间
                            if 0 <= vouchers <= 100:
                                return vouchers
                        except ValueError:
                            pass
                    
                    match = re.search(r'(抵扣券|优惠券)[:：]?(\d+\.?\d*)', text)
                    if match:
                        try:
                            vouchers = float(match.group(2))
                            if 0 <= vouchers <= 100:
                                return vouchers
                        except ValueError:
                            pass
                    
                    # 检查前后的文本（支持小数）
                    for j in range(max(0, i-3), min(len(texts), i+4)):
                        if j != i:
                            # 尝试匹配数字（包括小数）
                            match = re.search(r'^(\d+\.?\d*)$', texts[j].strip())
                            if match:
                                try:
                                    vouchers = float(match.group(1))
                                    if 0 <= vouchers <= 100:
                                        return vouchers
                                except ValueError:
                                    pass
            
            return None
            
        except Exception as e:
            # [2026-03-11] 优化日志：删除CMD输出
            pass
            return None
