"""
每日签到模块 - 处理每日签到功能
Daily Check-in Module - Handle daily check-in functionality
"""

import asyncio
import re
from typing import Optional, Dict, Tuple, Union
from datetime import datetime
from pathlib import Path
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
from .page_detector import PageState
from .navigator import Navigator
from .checkin_page_reader import CheckinPageReader
from .ocr_thread_pool import get_ocr_pool
from .performance.smart_waiter import wait_for_page  # 智能等待器
from .models.error_types import ErrorType
from .timeouts_config import TimeoutsConfig


class DailyCheckin:
    """每日签到处理器"""
    
    # 签到按钮坐标 (540x960) - 首页的签到入口
    CHECKIN_BUTTON = (477, 598)
    
    # 首页签到按钮的合理坐标范围 (x_min, x_max, y_min, y_max)
    # 基于实际观察：按钮通常在屏幕右侧中下部
    CHECKIN_BUTTON_VALID_RANGE = (400, 540, 500, 650)
    
    def __init__(self, adb: ADBBridge, detector: 'PageDetectorIntegrated', navigator: Navigator):
        """初始化签到处理器
        
        Args:
            adb: ADB桥接对象
            detector: 页面检测器（整合检测器或混合检测器，应该是从ModelManager获取的共享实例）
            navigator: 导航器
        """
        self.adb = adb
        self.detector = detector
        self.navigator = navigator
        self.reader = CheckinPageReader(adb)
        
        # 缓存首次成功的签到按钮坐标（用于后续点击）
        self._cached_home_checkin_button = None
        
        # 初始化智能按钮点击器
        from .smart_button_clicker import SmartButtonClicker
        from .model_manager import ModelManager
        model_manager = ModelManager.get_instance()
        ocr_pool = model_manager.get_ocr_thread_pool() if HAS_OCR else None
        self._smart_clicker = SmartButtonClicker(adb, detector, ocr_pool)
        
        # 初始化页面状态守卫
        from .page_state_guard import PageStateGuard
        self.guard = PageStateGuard(adb, detector)
        
        # 初始化页面检测缓存管理器
        from .page_detector_cache import PageDetectorCache
        self._page_cache = PageDetectorCache(
            default_ttl=0.5,  # 默认缓存0.5秒（签到流程中页面变化较快）
            max_cache_size=50
        )
        
        # 从ModelManager获取OCR线程池
        from .model_manager import ModelManager
        model_manager = ModelManager.get_instance()
        self._ocr_pool = model_manager.get_ocr_thread_pool() if HAS_OCR else None
        
        # 初始化OCR增强器（不需要传递参数，它会自己获取OCR线程池）
        if HAS_OCR:
            from .ocr_enhancer import OCREnhancer
            self._ocr_enhancer = OCREnhancer()
        else:
            self._ocr_enhancer = None
        
        # 创建截图保存目录
        self.screenshot_dir = Path("checkin_screenshots")
        self.screenshot_dir.mkdir(exist_ok=True)
        
        # 创建未签到截图目录
        self.no_checkin_dir = Path("no_checkin_screenshots")
        self.no_checkin_dir.mkdir(exist_ok=True)
        
        # 截图计数器（用于生成连续序号）
        self._screenshot_counter = self._get_next_counter()
    
    def _get_next_counter(self) -> int:
        """获取下一个截图序号（每天从1开始）
        
        Returns:
            int: 下一个可用的序号（从1开始）
        """
        # 获取当前日期
        date_str = datetime.now().strftime("%Y%m%d")
        
        # 扫描当天的截图，找到最大序号
        max_num = 0
        
        # 扫描签到截图目录的当天子目录
        date_dir = self.screenshot_dir / date_str
        if date_dir.exists():
            for file in date_dir.glob("*.png"):
                # 文件名格式：1.png, 2.png, ...
                match = re.match(r'^(\d+)\.png$', file.name)
                if match:
                    num = int(match.group(1))
                    max_num = max(max_num, num)
        
        # 扫描未签到截图目录的当天子目录
        no_checkin_date_dir = self.no_checkin_dir / date_str
        if no_checkin_date_dir.exists():
            for file in no_checkin_date_dir.glob("*.png"):
                match = re.match(r'^(\d+)\.png$', file.name)
                if match:
                    num = int(match.group(1))
                    max_num = max(max_num, num)
        
        return max_num + 1
    
    async def _detect_page_cached(self, device_id: str, use_cache: bool = True, 
                                  detect_elements: bool = False, 
                                  cache_key: str = "default",
                                  ttl: Optional[float] = None) -> Optional[any]:
        """使用缓存的页面检测
        
        这是一个便捷方法，封装了页面检测缓存的使用逻辑
        
        Args:
            device_id: 设备ID
            use_cache: 是否使用缓存
            detect_elements: 是否检测元素
            cache_key: 缓存键（用于区分不同类型的检测）
            ttl: 缓存生存时间（秒），None表示使用默认值
            
        Returns:
            页面检测结果
        """
        # 如果不使用缓存，直接检测并失效旧缓存
        if not use_cache:
            self._page_cache.invalidate(device_id, cache_key)
            result = await self.detector.detect_page(device_id, use_cache=False, detect_elements=detect_elements)
            return result
        
        # 尝试从缓存获取
        cached_result = self._page_cache.get(device_id, cache_key)
        if cached_result is not None:
            return cached_result
        
        # 缓存未命中，执行检测
        result = await self.detector.detect_page(device_id, use_cache=False, detect_elements=detect_elements)
        
        # 更新缓存
        if result is not None:
            self._page_cache.set(device_id, result, cache_key, ttl)
        
        return result
    
    async def _save_screenshot(self, device_id: str, phone: str, stage: str, attempt: int = None) -> Optional[str]:
        """保存截图（按日期文件夹组织）
        
        Args:
            device_id: 设备ID
            phone: 手机号（用于文件命名）
            stage: 截图阶段（page_enter/before_checkin/after_checkin/popup/after_close）
            attempt: 第几次签到（可选）
            
        Returns:
            str: 截图保存路径，失败返回None
        """
        if not HAS_PIL:
            return None
        
        # 导入日志记录器
        from .logger import get_logger
        logger = get_logger()
        
        try:
            screenshot_data = await self.adb.screencap(device_id)
            if not screenshot_data:
                return None
            
            image = Image.open(BytesIO(screenshot_data))
            
            # 获取当前日期，创建日期子目录
            date_str = datetime.now().strftime("%Y%m%d")
            date_dir = self.screenshot_dir / date_str
            date_dir.mkdir(parents=True, exist_ok=True)
            
            # 纯数字命名：1.png, 2.png, ...（不补0）
            current_num = self._screenshot_counter
            self._screenshot_counter += 1
            
            filename = f"{current_num}.png"
            screenshot_path = date_dir / filename
            image.save(screenshot_path)
            
            # 打印截图信息，方便查看
            logger.info(f"  [截图 {current_num}] {stage}" + (f" 第{attempt}次" if attempt else ""))
            
            return str(screenshot_path)
            
        except Exception as e:
            logger.warning(f"  ⚠️ 保存截图失败 ({stage}): {e}")
            return None
    
    async def _extract_reward_from_popup(self, device_id: str, phone: str = "unknown") -> Dict[str, any]:
        """从签到弹窗中提取奖励信息并保存截图（按日期文件夹组织）
        
        ⚠️ 已废弃：此方法已不再使用，改用余额对比计算总奖励
        保留此方法仅供参考或未来可能的需求
        
        Args:
            device_id: 设备ID
            phone: 手机号（用于文件命名）
            
        Returns:
            dict: 奖励信息
                - amount: float, 奖励金额
                - screenshot_path: str, 截图保存路径
                - ocr_texts: list, OCR识别的所有文本
        """
        result = {
            'amount': 0.0,
            'screenshot_path': None,
            'ocr_texts': []
        }
        
        if not HAS_PIL or not HAS_OCR:
            return result
        
        # 导入日志记录器
        from .logger import get_logger
        logger = get_logger()
        
        try:
            # 1. 截图
            screenshot_data = await self.adb.screencap(device_id)
            if not screenshot_data:
                return result
            
            image = Image.open(BytesIO(screenshot_data))
            
            # 2. 获取当前日期，创建日期子目录
            date_str = datetime.now().strftime("%Y%m%d")
            date_dir = self.screenshot_dir / date_str
            date_dir.mkdir(parents=True, exist_ok=True)
            
            # 3. 保存截图（纯数字命名）
            current_num = self._screenshot_counter
            self._screenshot_counter += 1
            
            filename = f"{current_num}.png"
            screenshot_path = date_dir / filename
            image.save(screenshot_path)
            result['screenshot_path'] = str(screenshot_path)
            
            logger.info(f"  [截图 {current_num}] 奖励弹窗")
            
            # 4. 使用OCR增强器识别金额（多策略）
            if self._ocr_enhancer:
                # 方法1: 裁剪金额区域后识别（更准确）
                amount_region = (120, 300, 300, 200)  # 弹窗中央偏上
                from .image_processor import ImageProcessor
                cropped = ImageProcessor.crop_region(image, *amount_region)
                
                # 使用增强器识别金额（减少超时时间）
                amount = await self._ocr_enhancer.recognize_amount(
                    cropped,
                    min_value=0.01,
                    max_value=100.0
                )
                
                if amount and amount > 0:
                    result['amount'] = amount
                    logger.info(f"  [OCR增强] ✓ 识别到金额: {amount:.2f} 元")
                else:
                    # 方法2: 如果区域识别失败，尝试全屏识别
                    logger.info(f"  [OCR增强] 区域识别失败，尝试全屏识别...")
                    amount = await self._ocr_enhancer.recognize_amount(
                        image,
                        min_value=0.01,
                        max_value=100.0
                    )
                    
                    if amount and amount > 0:
                        result['amount'] = amount
                        logger.info(f"  [OCR增强] ✓ 全屏识别到金额: {amount:.2f} 元")
                    else:
                        # 方法3: 使用传统方法作为后备（减少超时）
                        logger.info(f"  [OCR增强] 增强识别失败，使用传统方法...")
                        ocr_result = await self._ocr_pool.recognize(image, timeout=2.0)  # 优化：减少超时 10秒→2秒
                        if ocr_result and ocr_result.texts:
                            result['ocr_texts'] = ocr_result.texts
                            amount = self._parse_reward_amount(result['ocr_texts'])
                            result['amount'] = amount
            elif self._ocr_pool:
                # 如果没有增强器，使用传统方法（减少超时）
                ocr_result = await self._ocr_pool.recognize(image, timeout=2.0)  # 优化：减少超时 10秒→2秒
                if ocr_result and ocr_result.texts:
                    result['ocr_texts'] = ocr_result.texts
                    
                    # 5. 提取金额
                    # 查找包含"元"、"￥"、"¥"的文本，或者纯数字
                    amount = self._parse_reward_amount(result['ocr_texts'])
                    result['amount'] = amount
            
            return result
            
        except Exception as e:
            print(f"提取奖励信息失败: {e}")
            return result
    
    async def _extract_reward_amount_async(self, screenshot_path: str) -> float:
        """异步识别奖励金额（不阻塞主流程）
        
        ⚠️ 已废弃：此方法已不再使用，改用余额对比计算总奖励
        保留此方法仅供参考或未来可能的需求
        
        使用裁剪区域识别，避免误识别其他数字
        
        Args:
            screenshot_path: 已保存的截图路径
            
        Returns:
            float: 识别到的金额
        """
        from .logger import get_logger
        logger = get_logger()
        
        try:
            if not HAS_PIL:
                return 0.0
            
            # 从已保存的截图中识别金额
            image = Image.open(screenshot_path)
            
            # 使用OCR增强器识别金额（裁剪金额区域）
            if self._ocr_enhancer:
                # 裁剪金额区域（弹窗中央偏上）
                amount_region = (120, 300, 300, 200)
                from .image_processor import ImageProcessor
                cropped = ImageProcessor.crop_region(image, *amount_region)
                
                # 识别金额
                amount = await self._ocr_enhancer.recognize_amount(
                    cropped,
                    min_value=0.01,
                    max_value=100.0
                )
                if amount and amount > 0:
                    return amount
            
            # 降级到传统OCR（也使用裁剪区域）
            if self._ocr_pool:
                amount_region = (120, 300, 300, 200)
                from .image_processor import ImageProcessor
                cropped = ImageProcessor.crop_region(image, *amount_region)
                
                ocr_result = await self._ocr_pool.recognize(cropped, timeout=2.0)
                if ocr_result and ocr_result.texts:
                    return self._parse_reward_amount(ocr_result.texts)
            
            return 0.0
        except Exception as e:
            logger.warning(f"  ⚠️ 异步识别金额失败: {e}")
            return 0.0
    
    def _parse_reward_amount(self, texts: list) -> float:
        """从OCR文本中解析奖励金额
        
        Args:
            texts: OCR识别的文本列表
            
        Returns:
            float: 奖励金额
        """
        # 常见的奖励文本模式：
        # "恭喜获得 0.5 元"
        # "获得 ￥0.5"
        # "0.5元"
        # "奖励: 0.5"
        
        for text in texts:
            # 移除空格
            text = text.replace(" ", "")
            
            # 模式1: 包含"元"的文本
            if "元" in text:
                # 提取数字部分
                match = re.search(r'(\d+\.?\d*)', text)
                if match:
                    try:
                        return float(match.group(1))
                    except ValueError:
                        pass
            
            # 模式2: 包含货币符号的文本
            if "￥" in text or "¥" in text:
                # 提取数字部分
                match = re.search(r'[￥¥](\d+\.?\d*)', text)
                if match:
                    try:
                        return float(match.group(1))
                    except ValueError:
                        pass
            
            # 模式3: 纯数字（小数）
            if re.match(r'^\d+\.\d+$', text):
                try:
                    amount = float(text)
                    # 合理的金额范围：0.01 - 100
                    if 0.01 <= amount <= 100:
                        return amount
                except ValueError:
                    pass
        
        return 0.0
    
    async def _save_no_checkin_screenshot(self, device_id: str, phone: str = "unknown") -> Optional[str]:
        """保存未签到截图（无次数或已签到的情况，按日期文件夹组织）
        
        Args:
            device_id: 设备ID
            phone: 手机号（用于文件命名）
            
        Returns:
            str: 截图保存路径，失败返回None
        """
        if not HAS_PIL:
            return None
        
        # 导入日志记录器
        from .logger import get_logger
        logger = get_logger()
        
        try:
            # 截图
            screenshot_data = await self.adb.screencap(device_id)
            if not screenshot_data:
                return None
            
            image = Image.open(BytesIO(screenshot_data))
            
            # 获取当前日期，创建日期子目录
            date_str = datetime.now().strftime("%Y%m%d")
            date_dir = self.no_checkin_dir / date_str
            date_dir.mkdir(parents=True, exist_ok=True)
            
            # 纯数字命名
            current_num = self._screenshot_counter
            self._screenshot_counter += 1
            
            filename = f"{current_num}.png"
            screenshot_path = date_dir / filename
            image.save(screenshot_path)
            
            logger.info(f"  [截图 {current_num}] 无法签到")
            
            return str(screenshot_path)
            
        except Exception as e:
            logger.warning(f"保存未签到截图失败: {e}")
            return None
    
    async def do_checkin(self, device_id: str, phone: str = "unknown", password: str = None, 
                        login_callback=None, log_callback=None, profile_data: Optional[Dict] = None,
                        step_number: int = 1, gui_logger=None, allow_skip_profile: bool = False) -> Dict[str, any]:
        """执行每日签到（循环签到直到次数用完）
        
        Args:
            device_id: 设备ID
            phone: 手机号（用于截图文件命名）
            password: 密码（如果需要重新登录）
            login_callback: 登录回调函数（用于重新登录）
            log_callback: 日志回调函数（可选）
            profile_data: 个人信息数据（可选，如果提供则跳过获取个人信息步骤）
                - balance: float, 余额
                - points: int, 积分
                - vouchers: float, 抵扣券
            step_number: 步骤编号（用于简洁日志）
            gui_logger: GUI日志记录器（用于简洁日志输出）
            allow_skip_profile: 是否允许跳过个人信息获取（快速签到模式使用）
            
        Returns:
            dict: 签到结果
                - success: bool, 是否成功
                - message: str, 结果消息
                - already_checked: bool, 是否已签到
                - total_times: int, 总次数
                - remaining_times: int, 剩余次数
                - reward_amount: float, 总奖励金额
                - checkin_count: int, 本次签到次数
                - rewards: list, 每次签到的奖励列表
                - screenshots: list, 所有截图路径
                - need_relogin: bool, 是否需要重新登录
        """
        # 导入日志记录器
        from .logger import get_logger
        logger = get_logger()
        
        # 创建简洁日志记录器
        from .concise_logger import ConciseLogger
        concise = ConciseLogger("daily_checkin", gui_logger, logger)
        
        # 记录步骤开始
        concise.step(step_number, "签到")
        
        # 定义日志函数（同时输出到控制台和日志文件）
        def log(msg):
            logger.info(msg)
            if log_callback:
                log_callback(msg)
            else:
                print(msg)
        
        try:
            result = {
                'success': False,
                'message': '',
                'already_checked': False,
                'total_times': None,
                'remaining_times': None,
                'reward_amount': 0.0,
                'checkin_count': 0,
                'rewards': [],
                'screenshots': [],
                'need_relogin': False,
                'error_type': None  # 错误类型（ErrorType枚举）
            }
            
            # 初始化余额变量，避免后续引用未定义
            balance = None
            
            # 登录后直接开始签到流程，跳过登录状态检查（因为是顺序执行）
            log(f"  [签到] 开始签到流程（登录后直接执行）...")
            
            # 获取个人信息（余额、积分、抵扣券）- 如果已提供则跳过
            if profile_data:
                # 使用已提供的个人信息
                log(f"  [签到] 使用已获取的个人信息（跳过重复读取）")
                profile_success = True
                balance = profile_data.get('balance')
                points = profile_data.get('points')
                vouchers = profile_data.get('vouchers')
                
                log(f"  [签到] ✓ 个人信息:")
                if balance is not None:
                    log(f"    - 余额: {balance:.2f} 元")
                else:
                    log(f"    - 余额: None（⚠️ 无法计算签到奖励）")
                if points is not None:
                    log(f"    - 积分: {points} 积分")
                if vouchers is not None:
                    log(f"    - 抵扣券: {vouchers} 张")
            elif allow_skip_profile:
                # 快速签到模式：只获取余额（用于计算签到奖励），跳过积分和抵扣券
                log(f"  [签到] 快速签到模式：只获取余额...")
                profile_success = True  # 标记为成功，允许继续
                
                try:
                    from .profile_reader import ProfileReader
                    profile_reader = ProfileReader(self.adb, self.detector)
                    balance = await profile_reader.get_balance(device_id)
                    
                    if balance is not None:
                        log(f"  [签到] ✓ 签到前余额: {balance:.2f} 元")
                    else:
                        log(f"  [签到] ⚠️ 无法获取余额，将无法计算签到奖励")
                except Exception as e:
                    log(f"  [签到] ⚠️ 获取余额失败: {e}")
                    balance = None
                
                points = None
                vouchers = None
            else:
                # 需要获取个人信息
                log(f"  [签到] 获取个人信息...")
                profile_success = False
                
                try:
                    from .profile_reader import ProfileReader
                    profile_reader = ProfileReader(self.adb)
                    profile_data = await profile_reader.get_full_profile(device_id)
                    
                    if profile_data:
                        balance = profile_data.get('balance')
                        points = profile_data.get('points')
                        vouchers = profile_data.get('vouchers')
                        
                        # 至少要能获取到余额，才认为成功
                        if balance is not None:
                            profile_success = True
                            log(f"  [签到] ✓ 个人信息:")
                            log(f"    - 余额: {balance:.2f} 元")
                            
                            if points is not None:
                                log(f"    - 积分: {points} 积分")
                            else:
                                log(f"    - 积分: 无法获取")
                            
                            if vouchers is not None:
                                log(f"    - 抵扣券: {vouchers} 张")
                            else:
                                log(f"    - 抵扣券: 无法获取")
                        else:
                            log(f"  [签到] ❌ 无法获取余额信息")
                    else:
                        log(f"  [签到] ❌ 无法获取个人信息")
                        
                except Exception as e:
                    log(f"  [签到] ❌ 获取个人信息出错: {e}")
            
            # 如果获取个人信息失败，终止签到流程（快速签到模式除外）
            if not profile_success and not allow_skip_profile:
                result['message'] = "无法获取个人信息，可能不在应用内或登录状态异常"
                result['error_type'] = ErrorType.CANNOT_REACH_CHECKIN  # 无法到达签到页（前置条件失败）
                result['error_message'] = result['message']
                result['need_relogin'] = True
                log(f"  [签到] ❌ {result['message']}")
                return result
            
            # 优化：删除不必要的1秒等待
            
            # 4. 导航到首页准备签到
            concise.action("导航到首页")
            log(f"  [签到] 导航到首页...")
            success = await self.guard.ensure_page_state(
                device_id,
                PageState.HOME,
                self.navigator.navigate_to_home,
                "签到前导航首页"
            )
            
            if not success:
                result['message'] = "无法导航到首页"
                result['error_type'] = ErrorType.CANNOT_REACH_CHECKIN  # 无法到达签到页
                result['error_message'] = result['message']
                log(f"  [签到] ❌ 无法导航到首页")
                return result
            
            log(f"  [签到] ✓ 已到达首页")
            # 优化：删除不必要的1秒等待
            
            # 5. 使用YOLO或OCR识别签到按钮位置
            log(f"  [签到] 查找签到按钮...")
            checkin_button_pos = await self._find_checkin_button(device_id)
            if not checkin_button_pos:
                checkin_button_pos = self.CHECKIN_BUTTON
                log(f"  [签到] 使用默认签到按钮坐标: {checkin_button_pos}")
            else:
                log(f"  [签到] YOLO检测到按钮: {checkin_button_pos}")
            
            # 6. 点击签到按钮进入签到页面
            concise.action("点击每日签到")
            log(f"  [签到] 点击签到按钮 ({checkin_button_pos[0]}, {checkin_button_pos[1]})...")
            
            # 点击前先检测当前页面状态
            pre_click_result = await self._detect_page_cached(device_id, use_cache=False, cache_key="pre_click")
            if pre_click_result:
                log(f"  [签到] 点击前页面状态: {pre_click_result.state.value} (置信度: {pre_click_result.confidence:.2%})")
            
            # 使用智能按钮点击器（自动学习坐标）
            success, used_pos = await self._smart_clicker.click_button(
                device_id=device_id,
                button_name="home_checkin_button",
                valid_range=self.CHECKIN_BUTTON_VALID_RANGE,
                default_position=checkin_button_pos,
                cached_position=checkin_button_pos,
                log_callback=lambda msg: log(msg)
            )
            if not success:
                log(f"  [签到] ⚠️ 智能点击器失败，使用直接点击")
                await self.adb.tap(device_id, checkin_button_pos[0], checkin_button_pos[1])
            
            # 点击后短暂等待，让页面开始变化
            await asyncio.sleep(0.3)
            
            # 点击后立即检测页面状态
            post_click_result = await self._detect_page_cached(device_id, use_cache=False, cache_key="post_click")
            if post_click_result:
                log(f"  [签到] 点击后页面状态: {post_click_result.state.value} (置信度: {post_click_result.confidence:.2%})")
            
            # 等待页面加载（优化：使用智能等待器代替固定等待）
            log(f"  [签到] 等待签到页面加载...")
            from .performance.smart_waiter import wait_for_page
            page_result = await wait_for_page(
                device_id,
                self.detector,
                [PageState.CHECKIN, PageState.CHECKIN_POPUP, PageState.WARMTIP],  # 可能到达签到页、签到弹窗或温馨提示
                log_callback=lambda msg: log(f"  [SmartWaiter] {msg}")  # 输出SmartWaiter的日志
            )
            
            if not page_result:
                # 如果智能等待超时，给一个短暂的固定等待作为降级
                log(f"  [签到] 智能等待超时，使用降级等待...")
                await asyncio.sleep(1.0)  # 降级等待1秒
                # 重新检测页面状态
                page_result = await self._detect_page_cached(device_id, use_cache=False, cache_key="page_enter")
            else:
                # 智能等待成功，直接使用返回的页面结果
                log(f"  [签到] ✓ 页面已加载: {page_result.state.value} (置信度: {page_result.confidence:.2%})")
            
            # 6.1 进入签到页面后先截图
            log(f"  [签到] 保存进入页面截图...")
            page_enter_screenshot = await self._save_screenshot(device_id, phone, "page_enter")
            if page_enter_screenshot:
                result['screenshots'].append(page_enter_screenshot)
            
            # 6.2 智能等待识别到页面后，立即用深度学习检测页面类型
            
            # 检查是否误点到其他页面（文章页、搜索页、分类页）
            pages_need_return_home = [
                PageState.ARTICLE,   # 文章页
                PageState.SEARCH,    # 搜索页
                PageState.CATEGORY,  # 分类页
            ]
            
            if page_result and page_result.state in pages_need_return_home:
                log(f"  [签到] ⚠️ 误点到{page_result.state.value}，返回首页重新点击...")
                
                # 点击首页按钮或返回按钮
                if page_result.state == PageState.CATEGORY:
                    # 分类页：使用智能按钮点击器点击首页按钮
                    log(f"  [签到] 使用智能按钮点击器点击首页按钮...")
                    success, home_pos = await self._smart_clicker.click_button(
                        device_id=device_id,
                        button_name="nav_home_button",
                        valid_range=(50, 150, 850, 950),
                        default_position=(90, 920),
                        log_callback=lambda msg: log(msg)
                    )
                    if not success:
                        log(f"  [签到] ⚠️ 智能点击器失败，使用默认坐标")
                        await self.adb.tap(device_id, 90, 920)
                else:
                    # 其他页面：使用智能按钮点击器点击返回按钮
                    log(f"  [签到] 使用智能按钮点击器点击返回按钮...")
                    success, back_pos = await self._smart_clicker.click_button(
                        device_id=device_id,
                        button_name="back_button",
                        valid_range=(10, 80, 10, 80),
                        default_position=(40, 40),
                        log_callback=lambda msg: log(msg)
                    )
                    if not success:
                        log(f"  [签到] ⚠️ 智能点击器失败，按返回键")
                        await self.adb.press_back(device_id)
                
                await asyncio.sleep(1)
                self.detector.clear_cache(device_id)
                
                # 验证是否回到首页
                page_result = await self.detector.detect_page(device_id, use_cache=False, detect_elements=False)
                if page_result and page_result.state == PageState.HOME:
                    log(f"  [签到] ✓ 已返回首页，重新点击签到按钮...")
                    # 使用智能按钮点击器重新点击签到按钮
                    success, used_pos = await self._smart_clicker.click_button(
                        device_id=device_id,
                        button_name="home_checkin_button",
                        valid_range=self.CHECKIN_BUTTON_VALID_RANGE,
                        default_position=checkin_button_pos,
                        cached_position=checkin_button_pos,
                        log_callback=lambda msg: log(msg)
                    )
                    if not success:
                        log(f"  [签到] ⚠️ 智能点击器失败，使用直接点击")
                        await self.adb.tap(device_id, checkin_button_pos[0], checkin_button_pos[1])
                    await asyncio.sleep(1)
                    # 重新检测页面
                    page_result = await self.detector.detect_page(device_id, use_cache=False, detect_elements=False)
                else:
                    log(f"  [签到] ❌ 返回首页失败，当前页面: {page_result.state.value if page_result else 'unknown'}")
                    result['message'] = "返回首页失败"
                    result['error_type'] = ErrorType.CANNOT_REACH_CHECKIN
                    result['error_message'] = result['message']
                    return result
            
            if page_result and (page_result.state == PageState.CHECKIN or page_result.state == PageState.CHECKIN_POPUP):
                log(f"  [签到] ✓ 已进入签到页面（深度学习检测，置信度: {page_result.confidence:.2%}）")
                
                # 添加简洁日志：验证页面
                concise.action("验证当前页面")
                concise.action("签到页")
                
                # 6.3 立即进行 OCR 次数识别（必须的）- 增加重试机制
                log(f"  [签到] 读取签到次数信息...")
                concise.action("获取签到次数")
                
                # 重试机制：最多尝试3次
                initial_info = None
                for ocr_attempt in range(3):
                    if ocr_attempt > 0:
                        log(f"  [签到] OCR识别失败，第{ocr_attempt + 1}次重试...")
                        # 等待页面稳定
                        await asyncio.sleep(0.5)
                    
                    initial_info = await self.reader.get_checkin_info(device_id)
                    
                    # 检查是否识别成功
                    if initial_info and (initial_info['total_times'] is not None or initial_info['daily_remaining_times'] is not None):
                        if ocr_attempt > 0:
                            log(f"  [签到] ✓ 重试成功！")
                        break
                
                if initial_info and (initial_info['total_times'] is not None or initial_info['daily_remaining_times'] is not None):
                    log(f"  [签到] ✓ OCR 识别成功")
                    log(f"    - 总次数: {initial_info['total_times']}")
                    log(f"    - 剩余次数: {initial_info['daily_remaining_times']}")
                    
                    result['total_times'] = initial_info['total_times']
                    result['remaining_times'] = initial_info['daily_remaining_times']
                    
                    # 添加简洁日志
                    if initial_info['total_times'] is not None:
                        concise.action(f"总次数: {initial_info['total_times']}")
                    if initial_info['daily_remaining_times'] is not None:
                        concise.action(f"当日剩余: {initial_info['daily_remaining_times']}")
                else:
                    log(f"  [签到] ⚠️ OCR 识别失败（已重试3次），继续执行签到（无法显示次数信息）")
                    if initial_info:
                        log(f"    - 原始文本: {initial_info.get('raw_text', 'N/A')}")
                    
                    # 保存失败截图用于调试
                    fail_screenshot = await self._save_screenshot(device_id, phone, "ocr_failed")
                    if fail_screenshot:
                        result['screenshots'].append(fail_screenshot)
                        log(f"    - 已保存失败截图: {fail_screenshot}")
            else:
                # 无法确认是否进入签到页面
                log(f"  [签到] ❌ 无法确认是否进入签到页面")
                log(f"  [签到] 当前页面状态: {page_result.state.value if page_result else 'UNKNOWN'}")
                if page_result and hasattr(page_result, 'details'):
                    log(f"  [签到] 页面检测详情: {page_result.details}")
                
                # 保存当前页面截图用于调试
                debug_screenshot = await self._save_screenshot(device_id, phone, "checkin_page_failed")
                if debug_screenshot:
                    result['screenshots'].append(debug_screenshot)
                    log(f"  [签到] 已保存调试截图: {debug_screenshot}")
                
                result['message'] = "进入签到页面失败"
                result['error_type'] = ErrorType.CANNOT_REACH_CHECKIN  # 无法到达签到页
                result['error_message'] = result['message']
                return result
            
            # 8. 循环签到直到次数用完
            log(f"\n  [签到] 开始循环签到...")
            # 最多尝试20次，防止无限循环
            # 正常情况下会在以下条件退出：
            # 1. 检测到温馨提示弹窗（次数用完）
            # 2. 推算剩余次数为0
            # 3. 页面状态异常无法恢复
            max_attempts = 20
            
            # 优化：缓存变量
            cached_home_checkin_button = None  # 缓存首页的"每日签到"入口按钮位置
            cached_checkin_button = None  # 缓存签到页的"立即签到"按钮位置
            total_times = result['total_times']  # 缓存总次数
            initial_remaining_times = result['remaining_times']  # 缓存初始的当日剩余次数
            checkin_count = 0  # 签到计数器
            skip_page_verification = False  # 是否跳过页面验证（快速签到模式使用）
            
            # 优化：第一次循环时，使用已知的页面状态（从 wait_for_page 返回）
            current_state = page_result.state if page_result else PageState.UNKNOWN
            
            for attempt in range(max_attempts):
                # 5.0 每次循环前验证仍在签到页面
                # 优化：第一次循环跳过验证（已经知道页面状态）
                # 优化：快速签到模式下跳过验证（刚从首页重新进入签到页）
                if attempt > 0 and not skip_page_verification:
                    log(f"  [签到循环 {attempt+1}/{max_attempts}] 验证页面状态...")
                    
                    # 使用缓存的页面检测（循环中页面状态变化频繁，使用较短的TTL）
                    page_result_loop = await self._detect_page_cached(
                        device_id, 
                        use_cache=True, 
                        cache_key=f"loop_{attempt}",
                        ttl=0.3  # 循环中使用更短的缓存时间
                    )
                    current_state = page_result_loop.state if page_result_loop else PageState.UNKNOWN
                else:
                    # 第一次循环：使用已知的页面状态
                    log(f"  [签到循环 {attempt+1}/{max_attempts}] 使用已知页面状态: {current_state.value}")
                    # 重置跳过验证标志
                    skip_page_verification = False
                
                if current_state not in [PageState.CHECKIN, PageState.CHECKIN_POPUP]:
                    log(f"  [签到] ⚠️ 不在签到页面: {current_state.value}")
                    
                    # 特殊处理：如果是签到弹窗或温馨提示，尝试关闭弹窗
                    if current_state == PageState.CHECKIN_POPUP or current_state == PageState.WARMTIP:
                        log(f"  [签到] 检测到弹窗（{current_state.value}），尝试关闭...")
                        
                        # 先用OCR判断是温馨提示还是签到奖励弹窗
                        screenshot_data = await self.adb.screencap(device_id)
                        is_warmtip = False
                        if screenshot_data and HAS_PIL and self._ocr_pool:
                            image = Image.open(BytesIO(screenshot_data))
                            try:
                                ocr_result = await self._ocr_pool.recognize(image, timeout=2.0)
                                if ocr_result and ocr_result.texts:
                                    text_str = ''.join(ocr_result.texts)
                                    if "温馨提示" in text_str or ("提示" in text_str and "次数" in text_str):
                                        is_warmtip = True
                                        log(f"  [签到] OCR确认：温馨提示弹窗（次数用完）")
                            except Exception as e:
                                log(f"  [签到] OCR判断弹窗类型失败: {e}")
                        
                        # 如果是温馨提示，直接返回首页
                        if is_warmtip:
                            log(f"  [签到] 关闭温馨提示弹窗并返回首页...")
                            await self.adb.press_back(device_id)
                            await wait_for_page(
                                device_id,
                                self.detector,
                                [PageState.HOME],
                                log_callback=lambda msg: log(f"    [智能等待] {msg}")
                            )
                            log(f"  [签到] ✓ 已返回首页")
                            result['already_checked'] = True
                            result['remaining_times'] = 0
                            result['message'] = "今日已签到完成（签到次数已用完）"
                            result['success'] = True
                            break
                        
                        # 如果是签到奖励弹窗，关闭并继续
                        close_success = await self.detector.close_popup(device_id)
                        if close_success:
                            log(f"  [签到] ✓ 弹窗已关闭，等待页面刷新...")
                            await asyncio.sleep(0.5)  # 优化：减少等待时间从2秒到0.5秒
                            # 重新验证页面状态（失效缓存）
                            self._page_cache.invalidate(device_id, f"loop_{attempt}")
                            page_result_loop = await self._detect_page_cached(device_id, use_cache=False, cache_key=f"after_close_{attempt}")
                            current_state = page_result_loop.state if page_result_loop else PageState.UNKNOWN
                            if current_state == PageState.CHECKIN:
                                log(f"  [签到] ✓ 已返回签到页面，继续下一轮循环")
                                continue  # 直接进入下一轮循环
                            else:
                                log(f"  [签到] ⚠️ 关闭弹窗后仍不在签到页面: {current_state.value}")
                                # 如果仍然是弹窗状态，可能是关闭失败，再尝试一次
                                if current_state == PageState.CHECKIN_POPUP or current_state == PageState.WARMTIP:
                                    log(f"  [签到] 再次尝试关闭弹窗...")
                                    await self.adb.press_back(device_id)
                                    await asyncio.sleep(0.5)  # 优化：减少等待时间
                                    self._page_cache.invalidate(device_id, f"after_close_{attempt}")
                                    page_result_loop = await self._detect_page_cached(device_id, use_cache=False, cache_key=f"retry_close_{attempt}")
                                    current_state = page_result_loop.state if page_result_loop else PageState.UNKNOWN
                                    if current_state == PageState.CHECKIN:
                                        log(f"  [签到] ✓ 已返回签到页面，继续下一轮循环")
                                        continue
                        else:
                            log(f"  [签到] ⚠️ 关闭弹窗失败，尝试按返回键...")
                            await self.adb.press_back(device_id)
                            await asyncio.sleep(0.5)  # 优化：减少等待时间
                            self._page_cache.invalidate(device_id, f"loop_{attempt}")
                            page_result_loop = await self._detect_page_cached(device_id, use_cache=False, cache_key=f"fallback_{attempt}")
                            current_state = page_result_loop.state if page_result_loop else PageState.UNKNOWN
                            if current_state == PageState.CHECKIN:
                                log(f"  [签到] ✓ 已返回签到页面，继续下一轮循环")
                                continue
                    
                    # 如果不是弹窗，或者关闭弹窗后仍不在签到页面，尝试其他处理
                    if current_state not in [PageState.CHECKIN, PageState.CHECKIN_POPUP]:
                        # 尝试处理异常页面
                        log(f"  [签到] 尝试处理异常页面...")
                        handled = await self.guard._handle_unexpected_page(device_id, current_state, PageState.CHECKIN, "签到循环中", retry_count=attempt)
                        if not handled:
                            result['message'] = f"签到循环中页面异常: {current_state.value}"
                            result['error_type'] = ErrorType.CHECKIN_FAILED  # 签到失败
                            result['error_message'] = result['message']
                            log(f"  [签到] ❌ 无法处理异常页面")
                            break
                        # 重新验证（使用签到循环场景优先级模板）
                        log(f"  [签到] 重新验证页面状态...")
                        current_state = await self.guard.get_current_page_state(device_id, "处理异常后", scenario='checkin_loop')
                        if current_state not in [PageState.CHECKIN, PageState.CHECKIN_POPUP]:
                            result['message'] = "无法返回签到页面"
                            result['error_type'] = ErrorType.CHECKIN_FAILED  # 签到失败
                            result['error_message'] = result['message']
                            log(f"  [签到] ❌ 无法返回签到页面")
                            break
                        else:
                            log(f"  [签到] ✓ 已返回签到页面，继续循环")
                            continue  # 添加continue，避免执行后续的签到逻辑
                
                # 5.1 读取签到页面信息
                if attempt == 0:
                    # 第一次循环：直接使用进入页面时已读取的次数信息
                    total_times = result.get('total_times')
                    remaining_times = result.get('remaining_times')
                    
                    if total_times is not None:
                        log(f"  [签到 1] 使用已读取的次数信息 - 总次数: {total_times}, 当日剩余: {remaining_times}")
                    else:
                        log(f"  [签到 1] ⚠️ 未获取到次数信息，继续执行签到")
                else:
                    # 后续循环：通过初始剩余次数减去已签到次数来推算当前剩余次数
                    remaining_times = initial_remaining_times - checkin_count if initial_remaining_times else None
                    # 同步更新 result 中的剩余次数
                    result['remaining_times'] = remaining_times
                    log(f"  [签到 {attempt + 1}] 推算当日剩余次数: {remaining_times} (初始剩余: {initial_remaining_times}, 已签到: {checkin_count})")
                
                # 5.2 检查是否可以签到
                # 修复：第一次循环时，即使remaining_times为0，也要尝试点击一次
                # 因为OCR可能识别错误，或者页面数据是旧的
                # 只有在后续循环中，如果remaining_times为0才跳出
                if attempt > 0 and remaining_times is not None and remaining_times <= 0:
                    log(f"  [签到 {attempt + 1}] 剩余次数为0，今日已签到完成")
                    # 设置已签到标志
                    result['already_checked'] = True
                    result['remaining_times'] = 0
                    # 跳出循环
                    break
                else:
                    if remaining_times is not None and remaining_times <= 0 and attempt == 0:
                        log(f"  [签到 {attempt + 1}] OCR识别剩余次数为0，但仍尝试点击一次（可能是识别错误）")
                    else:
                        log(f"  [签到 {attempt + 1}] 剩余次数: {remaining_times if remaining_times is not None else '未知'}，继续签到...")
                
                # 5.4 执行签到（使用缓存的按钮位置）
                if cached_checkin_button is None:
                    # 第一次循环：检测签到按钮位置
                    log(f"  [签到 {attempt + 1}] 使用整合检测器检测签到按钮...")
                    try:
                        detection_result = await self._detect_page_cached(
                            device_id,
                            use_cache=True,  # 使用缓存（按钮位置通常不变）
                            detect_elements=True,
                            cache_key="button_detect",
                            ttl=2.0  # 按钮位置缓存2秒
                        )
                        
                        if detection_result and detection_result.elements:
                            for element in detection_result.elements:
                                if '签到按钮' in element.class_name or '签到' in element.class_name:
                                    cached_checkin_button = element.center
                                    log(f"  [签到 {attempt + 1}] 整合检测器检测到签到按钮: {cached_checkin_button} (置信度: {element.confidence:.2%})")
                                    break
                        
                        if cached_checkin_button is None:
                            log(f"  [签到 {attempt + 1}] 整合检测器未检测到按钮，降级到OCR...")
                    except Exception as e:
                        log(f"  [签到 {attempt + 1}] 整合检测器检测失败: {e}，降级到OCR...")
                    
                    # 如果整合检测器失败，使用默认签到按钮坐标
                    if cached_checkin_button is None:
                        cached_checkin_button = (270, 800)
                        log(f"  [签到 {attempt + 1}] 使用默认坐标: {cached_checkin_button}")
                else:
                    # 后续循环：使用缓存的按钮位置
                    log(f"  [签到 {attempt + 1}] 使用缓存的按钮位置: {cached_checkin_button}")
                
                x, y = cached_checkin_button
                
                # 5.4.1 点击签到按钮前截图（可选，调试用）
                # log(f"  [签到 {attempt + 1}] 保存点击前截图...")
                # before_screenshot = await self._save_screenshot(device_id, phone, "before_click", attempt + 1)
                # if before_screenshot:
                #     result['screenshots'].append(before_screenshot)
                
                # 5.4.2 点击签到按钮
                log(f"  [签到 {attempt + 1}] 点击签到按钮 ({x}, {y})...")
                
                # 添加简洁日志：点击立即签到
                if attempt == 0:
                    concise.action("开始签到")
                concise.action(f"第{attempt + 1}次签到")
                
                # 使用智能按钮点击器（自动学习坐标）
                success, used_pos = await self._smart_clicker.click_button(
                    device_id=device_id,
                    button_name="checkin_page_button",
                    valid_range=(200, 350, 750, 850),
                    default_position=(x, y),
                    cached_position=(x, y),
                    log_callback=lambda msg: log(msg)
                )
                if not success:
                    log(f"  [签到 {attempt + 1}] ⚠️ 智能点击器失败，使用直接点击")
                    await self.adb.tap(device_id, x, y)
                
                # 5.4.3 优化：根据当日剩余次数决定是否等待弹窗
                # 
                # 【优化策略】（识别到当日剩余次数时使用）
                # - 如果当日剩余次数 >= 1：跳过弹窗等待，直接返回首页再进入签到页（快速签到）
                #   注意：即使是最后一次（剩余1次），点击后也是签到成功弹窗，不是温馨提示
                #   只有再次点击签到按钮时才会出现温馨提示
                #
                # 【降级策略】（识别不到当日剩余次数时使用）
                # 使用保守策略：每次都等待弹窗，通过OCR判断是签到成功还是温馨提示
                # 这样可以保证核心流程的完整性，即使OCR识别失败也能正常签到
                #
                remaining = result.get('remaining_times')
                skip_popup_wait = False
                
                if remaining is not None and remaining >= 1:
                    skip_popup_wait = True
                    log(f"  [签到 {attempt + 1}] 【优化策略】当日剩余 {remaining} 次，跳过弹窗等待，直接返回首页...")
                    concise.action("跳过弹窗等待")
                    
                    # 等待0.5秒让签到请求完成
                    await asyncio.sleep(0.5)
                    
                    # 直接返回首页
                    await self.adb.press_back(device_id)
                    await asyncio.sleep(0.3)
                    
                    # 等待返回首页
                    await wait_for_page(
                        device_id,
                        self.detector,
                        [PageState.HOME],
                        log_callback=None
                    )
                    
                    log(f"  [签到 {attempt + 1}] ✓ 已返回首页，准备重新进入签到页...")
                    
                    # 重新点击首页的"每日签到"入口按钮进入签到页
                    # 使用智能按钮点击器（自动学习坐标）
                    if cached_home_checkin_button:
                        log(f"  [签到 {attempt + 1}] 使用缓存的首页入口按钮位置: {cached_home_checkin_button}")
                        success, used_pos = await self._smart_clicker.click_button(
                            device_id=device_id,
                            button_name="home_checkin_button",
                            valid_range=self.CHECKIN_BUTTON_VALID_RANGE,
                            default_position=cached_home_checkin_button,
                            cached_position=cached_home_checkin_button,
                            log_callback=lambda msg: log(msg)
                        )
                        if not success:
                            log(f"  [签到 {attempt + 1}] ⚠️ 智能点击器失败，使用直接点击")
                            await self.adb.tap(device_id, cached_home_checkin_button[0], cached_home_checkin_button[1])
                    else:
                        # 第一次需要查找首页入口按钮位置
                        button_pos = await self._find_checkin_button(device_id)
                        if button_pos:
                            cached_home_checkin_button = button_pos
                            log(f"  [签到 {attempt + 1}] 检测到首页入口按钮: {button_pos}")
                        else:
                            cached_home_checkin_button = self.CHECKIN_BUTTON
                            log(f"  [签到 {attempt + 1}] 使用默认首页入口按钮坐标: {cached_home_checkin_button}")
                        success, used_pos = await self._smart_clicker.click_button(
                            device_id=device_id,
                            button_name="home_checkin_button",
                            valid_range=self.CHECKIN_BUTTON_VALID_RANGE,
                            default_position=cached_home_checkin_button,
                            cached_position=cached_home_checkin_button,
                            log_callback=lambda msg: log(msg)
                        )
                        if not success:
                            log(f"  [签到 {attempt + 1}] ⚠️ 智能点击器失败，使用直接点击")
                            await self.adb.tap(device_id, cached_home_checkin_button[0], cached_home_checkin_button[1])
                    
                    # 使用SmartWaiter等待页面变化
                    log(f"  [签到 {attempt + 1}] 等待签到页面加载...")
                    from .performance.smart_waiter import wait_for_page
                    page_result = await wait_for_page(
                        device_id,
                        self.detector,
                        [PageState.CHECKIN, PageState.CHECKIN_POPUP, PageState.WARMTIP],
                        log_callback=None
                    )
                    
                    page_verified = False
                    if page_result and page_result.state == PageState.CHECKIN:
                        log(f"  [签到 {attempt + 1}] ✓ 已进入签到页面")
                        page_verified = True
                    else:
                        log(f"  [签到 {attempt + 1}] ❌ 未能进入签到页面，终止签到")
                        result['message'] = "快速签到模式：无法重新进入签到页面（超时）"
                        result['error_type'] = ErrorType.CHECKIN_FAILED
                        result['error_message'] = result['message']
                        break
                    
                    # 更新已处理次数
                    checkin_count += 1
                    result['checkin_count'] = checkin_count
                    
                    # 更新剩余次数
                    if result.get('remaining_times') is not None:
                        result['remaining_times'] -= 1
                    
                    log(f"  [签到 {attempt + 1}] ✓ 签到成功，当日剩余 {result.get('remaining_times', '?')} 次")
                    
                    # 设置跳过页面验证标志（下一次循环直接点击签到按钮）
                    skip_page_verification = True
                    
                    # 继续下一次循环（直接进入下一次签到，不需要重新识别次数）
                    continue
                else:
                    # 只有在剩余次数为0或未知时才等待弹窗
                    log(f"  [签到 {attempt + 1}] 【降级策略】当日剩余次数未知或为0，使用保守策略，等待弹窗...")
                
                # 5.4.4 清除页面检测缓存，确保智能等待器检测到最新状态
                if hasattr(self.detector, '_detection_cache'):
                    self.detector._detection_cache.clear(device_id)
                
                # 5.5 使用智能等待器检测弹窗类型
                log(f"  [签到 {attempt + 1}] 等待弹窗出现...")
                popup_detected = False
                is_warmtip = False  # 是否是温馨提示弹窗
                
                # 使用智能等待器等待弹窗出现（签到弹窗或温馨提示）
                wait_result = await wait_for_page(
                    device_id,
                    self.detector,
                    [PageState.CHECKIN_POPUP, PageState.WARMTIP],
                    log_callback=lambda msg: log(f"    [智能等待] {msg}")
                )
                
                # 优先使用智能等待器的结果
                if wait_result:
                    log(f"  [签到] 智能等待器检测到: {wait_result.state.value} (置信度: {wait_result.confidence:.2%})")
                    
                    if wait_result.state == PageState.CHECKIN_POPUP:
                        # 检测到签到弹窗，但需要用OCR验证是否是温馨提示（页面分类器可能误判）
                        log(f"  [签到] 检测到弹窗，使用OCR验证类型...")
                        screenshot_data = await self.adb.screencap(device_id)
                        if screenshot_data and HAS_PIL and self._ocr_pool:
                            image = Image.open(BytesIO(screenshot_data))
                            try:
                                ocr_result = await self._ocr_pool.recognize(image, timeout=TimeoutsConfig.OCR_TIMEOUT_SHORT)
                                if ocr_result and ocr_result.texts:
                                    text_str = ''.join(ocr_result.texts)
                                    
                                    # 先判断是否是温馨提示
                                    if "温馨提示" in text_str:
                                        popup_detected = True
                                        is_warmtip = True
                                        log(f"  [签到] ✓ OCR确认：温馨提示弹窗")
                                    else:
                                        popup_detected = True
                                        log(f"  [签到] ✓ OCR确认：签到奖励弹窗")
                            except Exception as e:
                                log(f"  [签到] OCR验证失败: {e}，假设为签到奖励弹窗")
                                popup_detected = True
                        else:
                            # 无法OCR验证，假设为签到奖励弹窗
                            popup_detected = True
                            log(f"  [签到] 无法OCR验证，假设为签到奖励弹窗")
                    elif wait_result.state == PageState.WARMTIP:
                        popup_detected = True
                        is_warmtip = True
                        log(f"  [签到] ✓ 检测到温馨提示弹窗")
                else:
                    # 智能等待器超时，重新检测当前页面状态
                    log(f"  [签到] ⚠️ 智能等待器超时，重新检测页面状态...")
                    self._page_cache.invalidate(device_id, f"loop_{attempt}")
                    page_result = await self._detect_page_cached(device_id, use_cache=False, cache_key=f"popup_check_{attempt}")
                    
                    if page_result:
                        log(f"  [签到] 当前页面: {page_result.state.value} (置信度: {page_result.confidence:.2%})")
                        
                        if page_result.state == PageState.CHECKIN_POPUP:
                            popup_detected = True
                            log(f"  [签到] ✓ 检测到签到奖励弹窗")
                        elif page_result.state == PageState.WARMTIP:
                            popup_detected = True
                            is_warmtip = True
                            log(f"  [签到] ✓ 检测到温馨提示弹窗")
                        else:
                            log(f"  [签到] ⚠️ 未检测到弹窗")
                    else:
                        log(f"  [签到] ⚠️ 页面检测失败，page_result 为 None")
                
                # 5.5.1 如果检测到温馨提示弹窗，直接处理
                if is_warmtip:
                    log(f"  [签到] ⚠️ 检测到温馨提示弹窗（次数用完）")
                    concise.action("出现温馨提示")
                    log(f"  [签到] 关闭温馨提示弹窗并返回首页...")
                    
                    # 关闭弹窗（按返回键）
                    await self.adb.press_back(device_id)
                    
                    # 智能等待返回首页
                    await wait_for_page(
                        device_id,
                        self.detector,
                        [PageState.HOME],
                        log_callback=lambda msg: log(f"    [智能等待] {msg}")
                    )
                    
                    log(f"  [签到] ✓ 已返回首页")
                    
                    # 设置已签到标志（次数用完 = 今日已签到完成）
                    result['already_checked'] = True
                    result['remaining_times'] = 0
                    
                    # 跳出循环
                    break
                
                # 5.5.2 如果检测到签到奖励弹窗，继续处理
                if popup_detected and not is_warmtip:
                    log(f"  [签到] ✓ 检测到签到奖励弹窗")
                    
                    # 优化：不截图，不识别金额（使用余额对比）
                    
                    # 检测关闭按钮位置
                    log(f"  [签到] 检测关闭按钮位置...")
                    close_button_pos = None
                    try:
                        detection_result = await self._detect_page_cached(
                            device_id,
                            use_cache=False,  # 不使用缓存（按钮位置会变化）
                            detect_elements=True,
                            cache_key=f"close_button_{attempt}",  # 每次尝试使用不同的key
                            ttl=0  # 不缓存
                        )
                        
                        if detection_result and detection_result.elements:
                            for element in detection_result.elements:
                                if '关闭' in element.class_name or '知道了' in element.class_name:
                                    close_button_pos = element.center
                                    log(f"  [签到] 找到关闭按钮: {close_button_pos} (置信度: {element.confidence:.2%})")
                                    break
                        
                        if close_button_pos is None:
                            # 如果YOLO未检测到，使用默认位置（与close_popup方法中的坐标一致）
                            close_button_pos = (270, 812)
                            log(f"  [签到] 未检测到关闭按钮，使用默认位置: {close_button_pos}")
                    except Exception as e:
                        log(f"  [签到] 检测关闭按钮失败: {e}，使用默认位置")
                        close_button_pos = (270, 812)
                    
                    # 使用智能按钮点击器关闭弹窗（自动学习坐标）
                    log(f"  [签到] 关闭弹窗...")
                    log(f"  [签到] 点击关闭按钮: {close_button_pos}")
                    success, used_pos = await self._smart_clicker.click_button(
                        device_id=device_id,
                        button_name="popup_close_button",
                        valid_range=(200, 350, 750, 850),
                        default_position=close_button_pos,
                        cached_position=close_button_pos,
                        log_callback=lambda msg: log(msg)
                    )
                    if not success:
                        log(f"  [签到] ⚠️ 智能点击器失败，使用直接点击")
                        await self.adb.tap(device_id, close_button_pos[0], close_button_pos[1])
                    
                    # 等待0.5秒检查是否关闭成功（给足够时间让弹窗消失）
                    await asyncio.sleep(0.5)
                    quick_check = await self._detect_page_cached(device_id, use_cache=False, cache_key=f"quick_check_{attempt}")
                    if quick_check and quick_check.state == PageState.CHECKIN_POPUP:
                        log(f"  [签到] 单击无效，尝试备用坐标...")
                        # 尝试备用坐标（与close_popup方法中的坐标一致）
                        backup_positions = [(278, 811), (274, 811)]
                        for backup_pos in backup_positions:
                            log(f"  [签到] 尝试备用坐标: {backup_pos}")
                            success, used_pos = await self._smart_clicker.click_button(
                                device_id=device_id,
                                button_name="popup_close_button",
                                valid_range=(200, 350, 750, 850),
                                default_position=backup_pos,
                                cached_position=backup_pos,
                                log_callback=lambda msg: log(msg)
                            )
                            if not success:
                                await self.adb.tap(device_id, backup_pos[0], backup_pos[1])
                            await asyncio.sleep(0.3)
                            check_result = await self._detect_page_cached(device_id, use_cache=False, cache_key=f"backup_check_{attempt}")
                            if check_result and check_result.state != PageState.CHECKIN_POPUP:
                                log(f"  [签到] ✓ 备用坐标成功关闭弹窗")
                                break
                    
                    # 使用智能等待器等待返回签到页
                    log(f"  [签到] 智能等待返回签到页...")
                    wait_success = await wait_for_page(
                        device_id,
                        self.detector,
                        [PageState.CHECKIN],
                        log_callback=lambda msg: log(f"    [智能等待] {msg}")
                    )
                    
                    if wait_success:
                        log(f"  [签到] ✓ 已返回签到页")
                    else:
                        log(f"  [签到] ⚠️ 等待超时，验证当前页面状态...")
                        page_result = await self._detect_page_cached(device_id, use_cache=False, cache_key=f"timeout_check_{attempt}")
                        if page_result:
                            log(f"  [签到] 当前页面: {page_result.state.value} (置信度: {page_result.confidence:.2%})")
                            if page_result.state == PageState.CHECKIN_POPUP:
                                log(f"  [签到] ⚠️ 弹窗未关闭，再次尝试（按返回键）...")
                                await self.adb.press_back(device_id)
                                # 再次智能等待
                                await wait_for_page(
                                    device_id,
                                    self.detector,
                                    [PageState.CHECKIN],
                                    log_callback=lambda msg: log(f"    [智能等待] {msg}")
                                )
                            elif page_result.state != PageState.CHECKIN:
                                log(f"  [签到] ⚠️ 页面状态异常: {page_result.state.value}")
                    
                    # 签到计数+1
                    checkin_count += 1
                    result['checkin_count'] = checkin_count
                    
                    # 更新剩余次数
                    if result.get('remaining_times') is not None:
                        result['remaining_times'] -= 1
                    
                    log(f"  [签到] ✓ 第{checkin_count}次签到完成，当日剩余 {result.get('remaining_times', '?')} 次")
                    concise.action(f"第{checkin_count}次签到完成")
                    
                    # 设置跳过页面验证标志（下一次循环直接点击签到按钮）
                    skip_page_verification = True
                    
                    # 继续下一轮循环
                    continue
                
                # 5.5.3 如果未检测到弹窗，使用OCR验证（后备方案）
                if not popup_detected:
                    log(f"  [签到] ⚠️ 智能等待未检测到弹窗，使用OCR验证...")
                    # 最后尝试使用OCR验证（可能是温馨提示弹窗）
                    screenshot_data = await self.adb.screencap(device_id)
                    if screenshot_data and HAS_PIL and self._ocr_pool:
                        image = Image.open(BytesIO(screenshot_data))
                        try:
                            ocr_result = await self._ocr_pool.recognize(image, timeout=TimeoutsConfig.OCR_TIMEOUT_SHORT)
                            if ocr_result and ocr_result.texts:
                                text_str = ''.join(ocr_result.texts)
                                
                                # 检测"温馨提示"弹窗（次数用完）- 只判断"温馨提示"
                                if "温馨提示" in text_str:
                                    log(f"  [签到] ⚠️ OCR检测到温馨提示弹窗（次数用完）")
                                    concise.action("出现温馨提示")
                                    # 调试模式下打印OCR文本（生产环境应关闭）
                                    # log(f"  [签到] OCR文本: {text_str[:100]}...")
                                    log(f"  [签到] 关闭温馨提示弹窗并返回首页...")
                                    
                                    # 关闭弹窗（按返回键）
                                    await self.adb.press_back(device_id)
                                    
                                    # 智能等待返回首页
                                    await wait_for_page(
                                        device_id,
                                        self.detector,
                                        [PageState.HOME],
                                        log_callback=lambda msg: log(f"    [智能等待] {msg}")
                                    )
                                    
                                    log(f"  [签到] ✓ 已返回首页")
                                    
                                    # 设置已签到标志（次数用完 = 今日已签到完成）
                                    result['already_checked'] = True
                                    result['remaining_times'] = 0
                                    
                                    # 跳出循环
                                    break
                                
                                # 检测签到奖励弹窗
                                has_congrats = "恭喜" in text_str
                                has_success = "成功" in text_str
                                has_know_button = "知道了" in text_str or "知道" in text_str
                                has_amount = "¥" in text_str or "￥" in text_str
                                
                                if has_congrats and has_success and (has_know_button or has_amount):
                                    popup_detected = True
                                    log(f"  [签到] ✓ OCR检测到奖励弹窗")
                                    
                                    # 不需要截图和OCR识别金额（使用余额对比计算总奖励）
                                    
                                    # 关闭弹窗
                                    await self.detector.close_popup(device_id)
                                    await wait_for_page(device_id, self.detector, [PageState.CHECKIN])
                                    
                                    checkin_count += 1
                                    log(f"  [签到] ✓ 第{checkin_count}次签到完成")
                                    concise.action(f"第{checkin_count}次签到完成")
                                    continue
                        except Exception as e:
                            log(f"  [签到] OCR验证失败: {e}")
                    
                    if not popup_detected:
                        log(f"  [签到] ⚠️ 未检测到弹窗，使用深度学习检测当前页面...")
                        # 截图记录当前状态
                        no_popup_screenshot = await self._save_screenshot(device_id, phone, "no_popup", attempt + 1)
                        if no_popup_screenshot:
                            result['screenshots'].append(no_popup_screenshot)
                        
                        # 使用深度学习检测当前页面类型
                        page_result = await self._detect_page_cached(device_id, use_cache=False, cache_key=f"no_popup_{attempt}")
                        
                        if page_result:
                            log(f"  [签到] 当前页面: {page_result.state.value} (置信度: {page_result.confidence:.2%})")
                            
                            # 如果在首页，说明可能已经自动返回（温馨提示弹窗自动关闭）
                            if page_result.state == PageState.HOME:
                                log(f"  [签到] ✓ 已在首页，推测为温馨提示弹窗已自动关闭（次数用完）")
                                result['already_checked'] = True
                                result['remaining_times'] = 0
                                break
                            
                            # 如果还在签到页，可能是弹窗检测失败，继续签到
                            elif page_result.state == PageState.CHECKIN:
                                log(f"  [签到] 仍在签到页，可能弹窗检测失败，继续签到...")
                                continue
                            
                            # 如果在其他页面，记录异常
                            else:
                                log(f"  [签到] ⚠️ 页面状态异常: {page_result.state.value}")
                                result['message'] = f"签到异常，页面跳转到: {page_result.state.value}"
                                result['error_type'] = ErrorType.CHECKIN_FAILED
                                result['error_message'] = result['message']
                                break
                        else:
                            # 无法检测页面状态
                            log(f"  [签到] ❌ 无法检测页面状态")
                            result['message'] = "签到卡住，无法检测页面状态"
                            result['error_type'] = ErrorType.CHECKIN_FAILED
                            result['error_message'] = result['message']
                            break
            
            # 循环结束后，使用余额对比计算总奖励
            log(f"  [签到] 签到循环结束，获取签到后的完整资料...")
            
            # 获取签到后的完整资料（复用正常流程的方式）
            checkin_balance_after = None  # 签到后余额
            final_profile = None  # 签到后的完整资料
            max_retries = 3
            
            for retry in range(max_retries):
                try:
                    log(f"  [签到] 尝试获取签到后资料（第{retry+1}/{max_retries}次）...")
                    
                    # 确保在个人页面
                    await self.navigator.navigate_to_profile(device_id)
                    await asyncio.sleep(TimeoutsConfig.WAIT_MEDIUM)  # 等待页面稳定
                    
                    # 使用正常流程的方式：get_full_profile_with_retry（带重试机制）
                    from .profile_reader import ProfileReader
                    profile_reader = ProfileReader(self.adb, self.detector)
                    account_str = f"{phone}----{password}" if password else phone
                    # 改用带重试的方法，和正常流程完全一样
                    profile_task = profile_reader.get_full_profile_with_retry(device_id, account=account_str, max_retries=3)
                    
                    try:
                        final_profile = await asyncio.wait_for(profile_task, timeout=30.0)  # 增加超时时间，因为内部有重试
                        
                        if final_profile and final_profile.get('balance') is not None:
                            checkin_balance_after = final_profile.get('balance')
                            
                            log(f"  [签到] ✓ 成功获取签到后资料:")
                            log(f"    - 余额: {checkin_balance_after:.2f} 元")
                            if final_profile.get('nickname'):
                                log(f"    - 昵称: {final_profile.get('nickname')}")
                            if final_profile.get('user_id'):
                                log(f"    - 用户ID: {final_profile.get('user_id')}")
                            if final_profile.get('points') is not None:
                                log(f"    - 积分: {final_profile.get('points')}")
                            if final_profile.get('vouchers') is not None:
                                log(f"    - 抵扣券: {final_profile.get('vouchers')}")
                            if final_profile.get('coupons') is not None:
                                log(f"    - 优惠券: {final_profile.get('coupons')}")
                            
                            break
                        else:
                            log(f"  [签到] ⚠️ 第{retry+1}次获取资料失败，余额为None")
                            if retry < max_retries - 1:
                                await asyncio.sleep(TimeoutsConfig.WAIT_MEDIUM)
                    except asyncio.TimeoutError:
                        log(f"  [签到] ⚠️ 第{retry+1}次获取资料超时")
                        if retry < max_retries - 1:
                            await asyncio.sleep(TimeoutsConfig.WAIT_MEDIUM)
                except Exception as e:
                    log(f"  [签到] ⚠️ 第{retry+1}次获取资料出错: {e}")
                    if retry < max_retries - 1:
                        await asyncio.sleep(TimeoutsConfig.WAIT_MEDIUM)
            
            # 计算总奖励
            if checkin_balance_after is not None and balance is not None:
                # 计算总奖励
                total_reward = checkin_balance_after - balance
                result['reward_amount'] = total_reward
                result['checkin_count'] = checkin_count
                result['checkin_balance_after'] = checkin_balance_after  # 返回签到后余额
                
                # 返回签到后的完整资料（供快速签到模式使用）
                if final_profile:
                    result['nickname'] = final_profile.get('nickname')
                    result['user_id'] = final_profile.get('user_id')
                    result['points'] = final_profile.get('points')
                    result['vouchers'] = final_profile.get('vouchers')
                    result['coupons'] = final_profile.get('coupons')
                    result['balance_before'] = balance  # 签到前余额
                
                log(f"  [签到] 签到前余额: {round(balance, 3)} 元")
                log(f"  [签到] 签到后余额: {round(checkin_balance_after, 3)} 元")
                log(f"  [签到] ✓ 总奖励: {round(total_reward, 3)} 元")
            else:
                log(f"  [签到] ⚠️ 无法获取余额，无法计算总奖励")
                log(f"  [签到]    - 签到前余额: {balance}")
                log(f"  [签到]    - 签到后余额: {checkin_balance_after}")
                result['checkin_count'] = checkin_count
                result['checkin_balance_after'] = checkin_balance_after  # 即使为None也返回
                
                # 即使无法计算奖励，也返回获取到的资料
                if final_profile:
                    result['nickname'] = final_profile.get('nickname')
                    result['user_id'] = final_profile.get('user_id')
                    result['points'] = final_profile.get('points')
                    result['vouchers'] = final_profile.get('vouchers')
                    result['coupons'] = final_profile.get('coupons')
                    result['balance_before'] = balance  # 签到前余额
            
            # 6. 设置最终结果
            # 统一处理：签到流程执行完毕即为成功
            result['success'] = True
            result['message'] = f"签到完成，共签到 {result['checkin_count']} 次，总奖励 {round(result['reward_amount'], 3)} 元"
            concise.success("签到完成")
            
            # 优化：获取到最终余额后不需要返回首页，直接返回结果（下一步是退出登录）
            
            return result
            
        except Exception as e:
            result['message'] = f"签到失败: {str(e)}"
            result['error_type'] = ErrorType.CHECKIN_EXCEPTION  # 签到异常
            result['error_message'] = str(e)
            # 优化：异常情况下也不需要返回首页，直接返回错误结果
            return result
        finally:
            pass  # 不需要恢复 print
    
    async def check_checkin_status(self, device_id: str) -> Dict[str, any]:
        """检查签到状态
        
        Args:
            device_id: 设备ID
            
        Returns:
            dict: 签到状态
                - can_checkin: bool, 是否可以签到
                - total_times: int, 总次数
                - remaining_times: int, 剩余次数
        """
        try:
            # 确保在首页
            await self.navigator.navigate_to_home(device_id)
            # 优化：删除不必要的1秒等待
            
            # 使用YOLO或OCR识别"每日签到"按钮位置
            checkin_button_pos = await self._find_checkin_button(device_id)
            if not checkin_button_pos:
                # 如果OCR识别失败，使用预设坐标
                checkin_button_pos = self.CHECKIN_BUTTON
            
            # 使用智能按钮点击器进入签到页面（自动学习坐标）
            success, used_pos = await self._smart_clicker.click_button(
                device_id=device_id,
                button_name="home_checkin_button",
                valid_range=self.CHECKIN_BUTTON_VALID_RANGE,
                default_position=checkin_button_pos,
                cached_position=checkin_button_pos,
                log_callback=lambda msg: print(msg)
            )
            if not success:
                print(f"  [签到] ⚠️ 智能点击器失败，使用直接点击")
                await self.adb.tap(device_id, checkin_button_pos[0], checkin_button_pos[1])
            
            # 优化：使用智能等待器等待进入签到页面
            await wait_for_page(
                device_id,
                self.detector,
                [PageState.CHECKIN],
                log_callback=lambda msg: print(f"    [智能等待] {msg}")
            )
            
            # 读取签到信息
            info = await self.reader.get_checkin_info(device_id)
            
            # 安全返回首页
            await self.navigator.safe_return_to_home(device_id)
            
            return {
                'can_checkin': info['can_checkin'] and (info['daily_remaining_times'] or 0) > 0,
                'total_times': info['total_times'],
                'remaining_times': info['daily_remaining_times']
            }
            
        except Exception:
            return {
                'can_checkin': False,
                'total_times': None,
                'remaining_times': None
            }
    
    async def _find_checkin_button(self, device_id: str) -> Optional[Tuple[int, int]]:
        """使用整合检测器或OCR识别首页的"每日签到"按钮位置
        
        使用智能按钮点击器的学习机制，但只返回坐标不点击
        
        Args:
            device_id: 设备ID
            
        Returns:
            tuple: 按钮中心点坐标 (x, y)，失败返回None
        """
        # 导入日志记录器
        from .logger import get_logger
        logger = get_logger()
        
        # 创建设备专属的学习器
        from .button_position_learner import ButtonPositionLearner
        learner = ButtonPositionLearner(device_id=device_id)
        
        button_name = "home_checkin_button"
        detected_position = None
        detection_confidence = 0.0
        
        # 步骤1: 优先使用整合检测器（YOLO）
        try:
            logger.info(f"  [签到] 使用整合检测器检测'每日签到'按钮...")
            
            # 使用整合检测器的元素检测功能（不使用缓存，避免旧坐标问题）
            detection_result = await self._detect_page_cached(
                device_id,
                use_cache=False,  # 修改：不使用缓存，每次都重新检测
                detect_elements=True,
                cache_key="home_checkin_button",
                ttl=0  # 不缓存
            )
            
            if detection_result and detection_result.elements:
                # 查找签到按钮元素
                for element in detection_result.elements:
                    if '每日签到' in element.class_name or '签到按钮' in element.class_name:
                        detected_position = element.center
                        detection_confidence = element.confidence
                        logger.info(f"  [签到] 整合检测器检测到'{element.class_name}': {detected_position} (置信度: {detection_confidence:.2%})")
                        break
                
                if detected_position is None:
                    logger.info(f"  [签到] 整合检测器未检测到签到按钮")
            else:
                logger.info(f"  [签到] 整合检测器未检测到元素")
        except Exception as e:
            logger.info(f"  [签到] 整合检测器检测失败: {e}")
        
        # 步骤2: 坐标合理性验证
        if detected_position:
            x, y = detected_position
            x_min, x_max, y_min, y_max = self.CHECKIN_BUTTON_VALID_RANGE
            
            # 检查是否在固定合理范围内
            if x_min <= x <= x_max and y_min <= y <= y_max:
                logger.info(f"  [签到] ✓ 坐标合理性验证通过: {detected_position}")
                
                # 记录成功坐标到学习器
                learner.record_success(button_name, detected_position, detection_confidence)
                
                # 缓存成功坐标
                self._cached_home_checkin_button = detected_position
                
                return detected_position
            else:
                logger.info(f"  [签到] ⚠️ 坐标不合理: {detected_position}，超出范围 {self.CHECKIN_BUTTON_VALID_RANGE}")
                logger.info(f"  [签到] 尝试使用学习器推荐坐标...")
        
        # 步骤3: 使用学习器推荐坐标
        learner_position = learner.get_best_position(
            button_name,
            min_samples=5,
            prefer_device=True
        )
        
        if learner_position:
            logger.info(f"  [签到] ✓ 学习器推荐坐标: {learner_position}")
            
            # 验证学习器推荐的坐标是否合理
            x, y = learner_position
            x_min, x_max, y_min, y_max = self.CHECKIN_BUTTON_VALID_RANGE
            if x_min <= x <= x_max and y_min <= y <= y_max:
                logger.info(f"  [签到] ✓ 学习器坐标验证通过")
                self._cached_home_checkin_button = learner_position
                return learner_position
            else:
                logger.info(f"  [签到] ⚠️ 学习器坐标不合理，继续降级...")
        else:
            logger.info(f"  [签到] 学习器无足够数据（需要至少5个样本）")
        
        # 步骤4: 使用缓存坐标
        if self._cached_home_checkin_button:
            logger.info(f"  [签到] 使用缓存坐标: {self._cached_home_checkin_button}")
            return self._cached_home_checkin_button
        
        # 步骤5: 降级到OCR识别
        if HAS_OCR and self._ocr_pool:
            logger.info(f"  [签到] 降级到OCR识别...")
            try:
                screenshot = await self.adb.screencap(device_id)
                if screenshot:
                    image = Image.open(BytesIO(screenshot))
                    
                    # 使用OCR线程池识别（减少超时）
                    from .timeouts_config import TimeoutsConfig
                    ocr_result = await self._ocr_pool.recognize(image, timeout=TimeoutsConfig.OCR_TIMEOUT_SHORT)
                    if ocr_result and ocr_result.texts:
                        # 查找"每日签到"或"签到"
                        for i, text in enumerate(ocr_result.texts):
                            if "每日签到" in text or (text == "签到" and i < len(ocr_result.boxes)):
                                box = ocr_result.boxes[i]
                                x_coords = [p[0] for p in box]
                                y_coords = [p[1] for p in box]
                                center_x = int(sum(x_coords) / len(x_coords))
                                center_y = int(sum(y_coords) / len(y_coords))
                                ocr_position = (center_x, center_y)
                                logger.info(f"  [签到] OCR找到签到按钮: {ocr_position}")
                                
                                # 验证OCR坐标是否合理
                                x_min, x_max, y_min, y_max = self.CHECKIN_BUTTON_VALID_RANGE
                                if x_min <= center_x <= x_max and y_min <= center_y <= y_max:
                                    logger.info(f"  [签到] ✓ OCR坐标验证通过")
                                    # 记录成功坐标
                                    learner.record_success(button_name, ocr_position, 1.0)
                                    self._cached_home_checkin_button = ocr_position
                                    return ocr_position
                                else:
                                    logger.info(f"  [签到] ⚠️ OCR坐标不合理，继续查找...")
            except Exception as e:
                logger.warning(f"  ⚠️ OCR识别签到按钮失败: {e}")
        
        # 步骤6: 最终降级到固定坐标
        logger.info(f"  [签到] 使用默认固定坐标: {self.CHECKIN_BUTTON}")
        return self.CHECKIN_BUTTON
