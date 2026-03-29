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
    # [2026-02-22] 修正：y 坐标调整为 548
    CHECKIN_BUTTON = (477, 548)
    
    # 首页签到按钮的合理坐标范围 (x_min, x_max, y_min, y_max)
    # 基于实际观察：按钮通常在屏幕右侧中部
    # [2026-02-22] 修正：y 范围调整（450-600）
    CHECKIN_BUTTON_VALID_RANGE = (400, 540, 450, 600)
    
    def __init__(self, adb: ADBBridge, detector: 'PageDetectorIntegrated', navigator: Navigator):
        """初始化签到处理器
        
        Args:
            adb: ADB桥接对象
            detector: 页面检测器（YOLO识别器，应该是从ModelManager获取的共享实例）
            navigator: 导航器
        """
        self.adb = adb
        
        # [2026-03-03] 修改：从 ModelManager 获取专用检测器（避免重复创建）
        # 保存YOLO检测器和导航器（用于元素检测、按钮点击等）
        self.detector = detector
        self.navigator = navigator
        
        # 导入日志记录器和ModelManager
        from .logger import get_logger
        from .model_manager import ModelManager
        
        logger = get_logger()
        model_manager = ModelManager.get_instance()
        
        # 获取签到专用检测器（MobileNetV3，仅用于页面分类）
        self.page_classifier = model_manager.get_checkin_detector()
        if not self.page_classifier:
            # 降级使用YOLO检测器
            self.page_classifier = detector
            logger.warning(f"⚠️ 签到专用模型未加载，降级使用YOLO检测器")
        
        self.reader = CheckinPageReader(adb)
        
        # 获取资料专用检测器（MobileNetV3，仅用于页面分类）
        self.profile_classifier = model_manager.get_profile_detector()
        if not self.profile_classifier:
            # 降级使用YOLO检测器
            self.profile_classifier = detector
            logger.warning(f"⚠️ 资料专用模型未加载，降级使用YOLO检测器")
        
        # [2026-03-01] 修复原因：导航器使用YOLO检测器（包含YOLO功能）
        # profile_navigator 使用YOLO检测器（用于导航和按钮点击）
        self.profile_navigator = navigator
        
        # [2026-02-21] 删除学习器：移除 SmartButtonClicker 初始化
        
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
        # [2026-03-02] 修复原因：使用签到专用模型进行页面分类，而不是YOLO检测器
        # 如果不使用缓存，直接检测并失效旧缓存
        if not use_cache:
            self._page_cache.invalidate(device_id, cache_key)
            result = await self.page_classifier.detect_page(device_id, use_cache=False, detect_elements=detect_elements)
            return result
        
        # 尝试从缓存获取
        cached_result = self._page_cache.get(device_id, cache_key)
        if cached_result is not None:
            return cached_result
        
        # 缓存未命中，执行检测
        result = await self.page_classifier.detect_page(device_id, use_cache=False, detect_elements=detect_elements)
        
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
            
            # [2026-03-01] 精简日志：删除截图日志输出
            
            return str(screenshot_path)
            
        except Exception as e:
            logger.warning(f"  ⚠️ 保存截图失败 ({stage}): {e}")
            return None
    
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
            
            # [2026-03-01] 精简日志：删除截图日志输出
            
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
                
        [2026-03-11] 修复原因：精简签到流程的详细日志输出，减少CMD控制台噪音
        """
        # 导入日志记录器
        from .logger import get_logger
        logger = get_logger()
        
        # [2026-03-11] 优化日志：不输出到GUI，避免CMD显示过多日志
        from .concise_logger import ConciseLogger
        concise = ConciseLogger("daily_checkin", None, logger)
        
        # 记录步骤开始
        concise.step(step_number, "签到")
        
        # [2026-03-01] 精简日志：定义日志函数（避免重复输出）
        def log(msg):
            # 只输出到logger（已包含控制台和文件）
            logger.info(msg)
            # 如果有GUI回调，也输出到GUI
            if log_callback:
                log_callback(msg)
        
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
                'error_type': None,  # 错误类型（ErrorType枚举）
                'nickname': None,  # 昵称
                'user_id': None,  # 用户ID
                'points': None,  # 积分
                'vouchers': None,  # 抵扣券
                'balance_before': None,  # 签到前余额
                'checkin_balance_after': None  # 签到后余额
            }
            
            # 初始化余额变量，避免后续引用未定义
            balance = None
            
            # 登录后直接开始签到流程，跳过登录状态检查（因为是顺序执行）
            # log(f"  [签到] 开始签到流程（登录后直接执行）...")  # [2026-03-01] 精简日志：删除多余提示
            
            # 获取个人信息（余额、积分、抵扣券）- 如果已提供则跳过
            if profile_data:
                # 使用已提供的个人信息
                # log(f"  [签到] 使用已获取的个人信息（跳过重复读取）")  # [2026-03-01] 精简日志：删除多余提示
                profile_success = True
                balance = profile_data.get('balance')
                points = profile_data.get('points')
                vouchers = profile_data.get('vouchers')
                nickname = profile_data.get('nickname')
                user_id = profile_data.get('user_id')
                checkin_total_times = profile_data.get('checkin_total_times')  # 签到总次数
                
                # [2026-03-12] 恢复个人信息显示：用户要求显示个人资料
                log(f"  [签到] ✓ 个人信息:")
                if nickname:
                    log(f"    昵称: {nickname}")
                if user_id:
                    log(f"    ID: {user_id}")
                if balance is not None:
                    log(f"    余额: {balance:.2f} 元")
                # else:
                #     log(f"    余额: None（⚠️ 无法计算签到奖励）")
                # if points is not None:
                #     log(f"    积分: {points} 积分")
                # if vouchers is not None:
                #     log(f"    抵扣券: {vouchers} 张")
                # if checkin_total_times is not None:
                #     log(f"    签到次数: {checkin_total_times} 次")
            elif allow_skip_profile:
                # 快速签到模式：完全不去个人页，使用从数据库获取的余额
                log(f"  [签到] 快速签到模式：跳过个人页获取资料...")
                profile_success = True  # 标记为成功，允许继续
                
                # [2026-03-11] 修复原因：快速签到模式下，没有历史记录就设置余额为0
                # 从 profile_data 中获取余额（如果有的话）
                if profile_data and profile_data.get('balance') is not None:
                    balance = profile_data.get('balance')
                    log(f"  [签到] ✓ 使用数据库余额: {balance:.2f} 元")
                else:
                    balance = 0.0
                    log(f"  [签到] ⚠️ 无历史记录，设置余额为 0.00 元")
                
                points = None
                vouchers = None
            else:
                # 完整模式：需要去个人页获取完整个人信息
                log(f"  [签到] 完整模式：导航到个人页获取完整资料...")
                profile_success = False
                
                try:
                    # 1. 先导航到个人页
                    concise.action("导航到个人页")
                    nav_success = await self.navigator.navigate_to_profile(device_id)
                    
                    if not nav_success:
                        log(f"  [签到] ❌ 无法导航到个人页")
                        profile_success = False
                    else:
                        log(f"  [签到] ✓ 已到达个人页")
                        
                        # 2. 获取完整个人资料
                        concise.action("读取个人资料")
                        from .profile_reader import ProfileReader
                        # [2026-03-11] 修复原因：使用PageDetectorIntegrated（已添加find_button_yolo方法）
                        profile_reader = ProfileReader(self.adb, self.detector)
                        profile_data = await profile_reader.get_full_profile(device_id)
                        
                        if profile_data:
                            balance = profile_data.get('balance')
                            points = profile_data.get('points')
                            vouchers = profile_data.get('vouchers')
                            nickname = profile_data.get('nickname')
                            user_id = profile_data.get('user_id')
                            checkin_total_times = profile_data.get('checkin_total_times')  # 签到总次数
                            
                            # 至少要能获取到余额，才认为成功
                            if balance is not None:
                                profile_success = True
                                # [2026-03-12] 恢复个人信息显示：用户要求显示个人资料
                                log(f"  [签到] ✓ 个人信息:")
                                if nickname:
                                    log(f"    昵称: {nickname}")
                                if user_id:
                                    log(f"    ID: {user_id}")
                                log(f"    余额: {balance:.2f} 元")
                                
                                if points is not None:
                                    log(f"    积分: {points} 积分")
                                else:
                                    log(f"    积分: 无法获取")
                                # 
                                # if vouchers is not None:
                                #     log(f"    抵扣券: {vouchers} 张")
                                # else:
                                #     log(f"    抵扣券: 无法获取")
                                # 
                                # if checkin_total_times is not None:
                                #     log(f"    签到次数: {checkin_total_times} 次")
                                pass  # 成功获取余额，不输出详细信息
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
            
            # [2026-03-03] 修复原因：删除多余的页面检测，避免误判导致误操作
            # 启动流程已经确认到达首页，签到流程紧接着执行，页面不可能变化
            # 直接信任启动流程的结果，不做任何页面检测和导航
            
            # [2026-03-11] 优化日志：移除控制台DEBUG输出
            if log_callback:
                log_callback(f"签到流程开始")
            
            # 6. 使用固定坐标点击签到按钮（最可靠）
            # [2026-03-01] 修复原因：优先使用固定坐标，避免YOLO检测错误导致误点
            checkin_button_pos = self.CHECKIN_BUTTON
            # log(f"  [签到] 使用固定签到按钮坐标: {checkin_button_pos}")  # [2026-03-01] 精简日志：删除中间步骤
            
            # 7. 点击签到按钮进入签到页面（带重试机制）
            # [2026-03-01] 修复原因：添加点击验证和重试机制，解决点击失败问题
            concise.action("点击每日签到")
            
            # [2026-03-11] 优化日志：移除控制台DEBUG输出
            
            page_result = None
            max_click_attempts = 3  # 最多尝试3次点击
            
            for click_attempt in range(max_click_attempts):
                if click_attempt > 0:
                    log(f"  [签到] 第 {click_attempt + 1} 次尝试点击签到按钮...")
                
                # [2026-03-11] 优化日志：移除控制台DEBUG输出
                
                # 点击签到按钮
                await self.adb.tap(device_id, checkin_button_pos[0], checkin_button_pos[1])
                
                # [2026-03-11] 优化日志：移除控制台DEBUG输出
                
                # [2026-03-01] 修复：点击后先等待0.5秒让页面开始加载，再调用智能等待器
                # 智能等待器会立即开始检测，如果页面还没开始加载，第一次检测会失败
                await asyncio.sleep(0.5)
                
                # [2026-03-01] 修复：使用智能等待器等待页面变化，而不是固定等待
                # 智能等待器会持续检测，一旦页面变化就立即返回（内置15秒超时）
                # [2026-03-29] 修复：加入LOGIN页面，避免跳到登录页时超时导致漏检
                page_result = await wait_for_page(
                    device_id,
                    self.page_classifier,
                    [PageState.CHECKIN, PageState.CHECKIN_POPUP, PageState.WARMTIP, PageState.LOGIN],
                    log_callback=None
                )
                
                # 如果智能等待器检测到页面变化，说明点击成功
                if page_result and page_result.state != PageState.HOME:
                    if click_attempt > 0:
                        log(f"  [签到] ✓ 点击成功，页面已变化到: {page_result.state.value}")
                    break
                
                # 如果智能等待器超时（还是首页），说明点击失败
                if click_attempt < max_click_attempts - 1:
                    log(f"  [签到] ⚠️ 点击后页面未变化，准备重试...")
                    await asyncio.sleep(0.3)  # 短暂等待后重试
                else:
                    log(f"  [签到] ❌ 点击 {max_click_attempts} 次后页面仍未变化")
            
            # [2026-03-01] 精简日志：只输出页面状态，不输出置信度
            # if page_result:
            #     log(f"  [签到] 检测到页面: {page_result.state.value} ({page_result.state.chinese_name})")  # [2026-03-01] 精简日志：删除中间步骤
            # else:
            #     log(f"  [签到] ⚠️ 页面检测失败，无法获取页面状态")  # [2026-03-01] 精简日志：删除中间步骤
            
            # 6.1 进入签到页面后先截图
            # [2026-03-29] 修复：截图前等待0.5秒，避免截图操作干扰触摸事件导致上划
            await asyncio.sleep(0.5)
            # log(f"  [签到] 保存进入页面截图...")  # [2026-03-01] 精简日志：删除中间步骤
            page_enter_screenshot = await self._save_screenshot(device_id, phone, "page_enter")
            if page_enter_screenshot:
                result['screenshots'].append(page_enter_screenshot)
            
            # 6.2 检查是否到达登录页（缓存失效）
            if page_result and page_result.state == PageState.LOGIN:
                concise.action("缓存失效，执行登录")
                
                # 如果提供了登录回调，直接执行登录
                if login_callback and password:
                    log(f"  [签到] 执行登录流程...")
                    try:
                        # 调用登录回调（这是一个协程）
                        login_result = await login_callback(device_id, phone, password)
                        
                        if login_result and login_result.success:
                            log(f"  [签到] ✓ 登录成功，继续签到流程")
                            concise.success("登录成功")
                            
                            # 登录后需要处理积分页跳转（优化：减少等待时间）
                            await asyncio.sleep(1.0)  # 优化：2秒→1秒
                            
                            # [2026-03-02] 修复原因：使用签到专用模型检测页面
                            # 检测当前页面
                            page_result = await self.page_classifier.detect_page(
                                device_id, use_cache=False
                            )
                            
                            if page_result and page_result.state == PageState.POINTS_PAGE:
                                log(f"  [签到] 检测到积分页，按2次返回键...")
                                await self.adb.press_back(device_id)
                                await asyncio.sleep(0.5)  # 优化：1秒→0.5秒
                                await self.adb.press_back(device_id)
                                await asyncio.sleep(0.5)  # 优化：1秒→0.5秒
                                self.detector.clear_cache()
                            
                            # [2026-03-03] 修复原因：从数据库获取签到前余额，不需要导航到个人页
                            # 快速签到模式下，缓存失效后需要重新获取签到前余额
                            if allow_skip_profile:
                                try:
                                    # 从数据库获取最新余额
                                    balance_record = self.db.get_latest_balance(account_name)
                                    if balance_record:
                                        balance = balance_record['balance']
                                    else:
                                        balance = 0.0
                                except Exception as e:
                                    balance = 0.0
                            
                            # [2026-03-29] 修复：签到页跳登录页，登录成功后app会自动返回签到页
                            # 不需要导航首页再重新点击签到，直接等待签到页出现即可
                            log(f"  [签到] 等待返回签到页...")
                            page_result = await wait_for_page(
                                device_id,
                                self.page_classifier,
                                [PageState.CHECKIN, PageState.CHECKIN_POPUP, PageState.WARMTIP],
                                log_callback=lambda msg: log(f"  [等待] {msg}")
                            )
                            
                            # 如果15秒超时，再尝试一次检测
                            if not page_result:
                                log(f"  [签到] 等待超时，再次检测...")
                                await asyncio.sleep(1.0)
                                page_result = await self._detect_page_cached(device_id, use_cache=False, cache_key="page_enter_after_login")
                        else:
                            log(f"  [签到] ❌ 登录失败: {login_result.error_message if login_result else '未知错误'}")
                            result['message'] = f"登录失败: {login_result.error_message if login_result else '未知错误'}"
                            result['error_type'] = ErrorType.LOGIN_PASSWORD_ERROR
                            result['error_message'] = result['message']
                            result['need_relogin'] = True
                            return result
                    except Exception as e:
                        log(f"  [签到] ❌ 登录过程出错: {str(e)}")
                        result['message'] = f"登录过程出错: {str(e)}"
                        result['error_type'] = ErrorType.LOGIN_PASSWORD_ERROR
                        result['error_message'] = result['message']
                        result['need_relogin'] = True
                        return result
                else:
                    # 没有提供登录回调或密码，无法登录
                    log(f"  [签到] ❌ 缓存已失效但未提供登录信息")
                    result['message'] = "缓存已失效，需要重新登录"
                    result['error_type'] = ErrorType.CACHE_INVALID
                    result['error_message'] = result['message']
                    result['need_relogin'] = True
                    return result
            
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
                    # [2026-02-21] 删除学习器：直接点击首页按钮
                    log(f"  [签到] 点击首页按钮...")
                    await self.adb.tap(device_id, 90, 920)
                else:
                    # [2026-02-21] 删除学习器：直接按返回键
                    log(f"  [签到] 按返回键...")
                    await self.adb.press_back(device_id)
                
                await asyncio.sleep(0.5)  # 优化：1秒→0.5秒
                self.detector.clear_cache(device_id)
                
                # 等待返回首页（检测到立即返回，15秒只是超时保护）
                # [2026-03-01] 修复：使用签到专用检测器
                log(f"  [签到] 等待返回首页...")
                home_result = await wait_for_page(
                    device_id,
                    self.page_classifier,
                    [PageState.HOME],
                    log_callback=lambda msg: log(f"  [等待] {msg}")
                )
                
                # 如果15秒超时，标记失败
                if not home_result or home_result.state != PageState.HOME:
                    log(f"  [签到] ❌ 等待返回首页超时（15秒），标记失败")
                    result['message'] = "返回首页失败（15秒超时）"
                    result['error_type'] = ErrorType.CANNOT_REACH_CHECKIN
                    result['error_message'] = result['message']
                    return result
                
                # 已返回首页，重新点击签到按钮
                log(f"  [签到] ✓ 已返回首页，重新点击签到按钮...")
                # [2026-02-21] 删除学习器：直接点击签到按钮
                log(f"  [签到] ✓ 已返回首页，重新点击签到按钮...")
                await self.adb.tap(device_id, checkin_button_pos[0], checkin_button_pos[1])
                
                await asyncio.sleep(0.3)  # 优化：1秒→0.3秒
                
                # 等待进入签到页（检测到立即返回，15秒只是超时保护）
                # [2026-03-01] 修复：使用签到专用检测器
                log(f"  [签到] 等待进入签到页...")
                page_result = await wait_for_page(
                    device_id,
                    self.page_classifier,
                    [PageState.CHECKIN, PageState.CHECKIN_POPUP, PageState.WARMTIP],
                    log_callback=lambda msg: log(f"  [等待] {msg}")
                )
            
            # [2026-03-01] 修复：当检测到CHECKIN_POPUP或其他非签到页状态时，用OCR验证是否是签到页被误识别
            if page_result and page_result.state not in [PageState.CHECKIN, PageState.LOGIN]:
                log(f"  [签到] 检测到非签到页状态({page_result.state.value})，使用OCR验证是否误识别...")
                
                # [2026-03-29] 修复：截图前等待0.5秒，避免截图操作干扰触摸事件导致上划
                await asyncio.sleep(0.5)
                
                # 使用OCR验证
                try:
                    screenshot_data = await self.adb.screencap(device_id)
                    if screenshot_data and HAS_PIL:
                        image = Image.open(BytesIO(screenshot_data))
                        
                        if self._ocr_pool:
                            ocr_result = await self._ocr_pool.recognize(image, timeout=3.0)
                            
                            # [2026-03-05] 修复数组比较错误：检查 texts 是否为 None 并且长度大于 0
                            if ocr_result and ocr_result.texts is not None and len(ocr_result.texts) > 0:
                                all_text = ' '.join(ocr_result.texts)
                                
                                # 检查是否是签到页特征（有"立即签到"但没有"恭喜"/"签到成功"）
                                has_checkin_button = '立即签到' in all_text or '每日签到' in all_text or '每天签到' in all_text
                                has_popup_text = '恭喜' in all_text or '获得' in all_text or '签到成功' in all_text
                                
                                if has_checkin_button and not has_popup_text:
                                    log(f"  [签到] ✓ OCR验证：这是签到页（被误识别为{page_result.state.value}）")
                                    
                                    # 手动修正页面状态
                                    page_result.state = PageState.CHECKIN
                                elif has_popup_text:
                                    log(f"  [签到] ✓ OCR验证：确实是签到弹窗")
                                else:
                                    log(f"  [签到] ⚠️ OCR验证：无法确定，保持原判断({page_result.state.value})")
                except Exception as e:
                    log(f"  [签到] ⚠️ OCR验证出错: {e}，保持原判断")
            
            if page_result and (page_result.state == PageState.CHECKIN or page_result.state == PageState.CHECKIN_POPUP):
                log(f"  [签到] ✓ 已进入签到页面")  # [2026-03-01] 保留关键结果
                
                # 添加简洁日志：验证页面
                concise.action("验证当前页面")
                concise.action("签到页")
                
                # 6.3 立即进行 OCR 次数识别（必须的）- 增加重试机制
                # log(f"  [签到] 读取签到次数信息...")  # [2026-03-01] 精简日志：删除中间步骤
                concise.action("获取签到次数")
                
                # 重试机制：最多尝试1次（减少重试避免阻塞）
                initial_info = None
                for ocr_attempt in range(1):  # [2026-03-29] 修改：从3次减少到1次
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
                    # [2026-03-11] 精简日志：直接显示次数，不显示"已获取次数"
                    log(f"  [签到] 总次数: {initial_info['total_times']}, 今日次数: {initial_info['daily_remaining_times']}")
                    
                    result['total_times'] = initial_info['total_times']
                    result['remaining_times'] = initial_info['daily_remaining_times']
                    
                    # 添加简洁日志
                    if initial_info['total_times'] is not None:
                        concise.action(f"总次数: {initial_info['total_times']}")
                    if initial_info['daily_remaining_times'] is not None:
                        concise.action(f"当日剩余: {initial_info['daily_remaining_times']}")
                    
                    # [2026-03-11] 修改原因：快速签到模式下，今日次数为0时直接返回首页，不执行签到循环
                    skip_checkin_due_to_zero_times = False
                    if allow_skip_profile and initial_info['daily_remaining_times'] is not None and initial_info['daily_remaining_times'] == 0:
                        log(f"  [签到] 快速签到模式：今日次数为0，无需点击，直接返回首页")
                        concise.action("今日次数为0，跳过签到")
                        
                        # 直接返回首页
                        await self.adb.press_back(device_id)
                        await asyncio.sleep(0.5)
                        
                        # 等待返回首页
                        try:
                            await asyncio.wait_for(
                                wait_for_page(
                                    device_id,
                                    self.page_classifier,
                                    [PageState.HOME],
                                    log_callback=None
                                ),
                                timeout=5.0
                            )
                            log(f"  [签到] ✓ 已返回首页")
                        except asyncio.TimeoutError:
                            log(f"  [签到] ⚠️ 返回首页超时，继续执行...")
                        
                        # 设置跳过标志
                        skip_checkin_due_to_zero_times = True
                else:
                    log(f"  [签到] ⚠️ OCR 识别失败（已重试1次），继续执行签到（无法显示次数信息）")
                    if initial_info:
                        log(f"    - 原始文本: {initial_info.get('raw_text', 'N/A')}")
                    
                    # 保存失败截图用于调试
                    fail_screenshot = await self._save_screenshot(device_id, phone, "ocr_failed")
                    if fail_screenshot:
                        result['screenshots'].append(fail_screenshot)
                        log(f"    - 已保存失败截图: {fail_screenshot}")
            else:
                # [2026-03-29] 修复：先检查是否跳到了登录页，如果是则执行登录流程
                if page_result and page_result.state == PageState.LOGIN:
                    # 走到6.2的登录处理逻辑（不重复写，直接fall-through到下面的检测）
                    pass
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
            # log(f"\n  [签到] 开始循环签到...")  # [2026-03-01] 精简日志：删除中间步骤
            # 最多尝试20次，防止无限循环
            # 正常情况下会在以下条件退出：
            # 1. 检测到温馨提示弹窗（次数用完）
            # 2. 推算剩余次数为0
            # 3. 页面状态异常无法恢复
            max_attempts = 20
            
            # 优化：缓存变量
            total_times = result['total_times']  # 缓存总次数
            initial_remaining_times = result['remaining_times']  # 缓存初始的当日剩余次数
            checkin_count = 0  # 签到计数器
            skip_page_verification = False  # 是否跳过页面验证（快速签到模式使用）
            
            # [2026-03-01] 记录循环退出状态
            exit_reason = None  # 退出原因：warmtip/remaining_zero/error
            exit_page = None  # 退出时的页面状态
            
            # 优化：第一次循环时，使用已知的页面状态（从 wait_for_page 返回）
            current_state = page_result.state if page_result else PageState.UNKNOWN
            
            # [2026-03-11] 修复原因：快速签到模式下减少页面验证，提升性能
            quick_mode_optimization = allow_skip_profile  # 快速签到模式标志
            
            for attempt in range(max_attempts):
                # [2026-03-11] 修改原因：快速签到模式下，如果今日次数为0，直接跳出循环
                if 'skip_checkin_due_to_zero_times' in locals() and skip_checkin_due_to_zero_times:
                    # 设置签到结果（已签到完成，次数为0）
                    result['success'] = True
                    result['already_checked'] = True
                    result['remaining_times'] = 0
                    result['checkin_count'] = 0
                    result['reward_amount'] = 0.0
                    result['message'] = "今日签到次数为0，已跳过签到流程"
                    exit_reason = "remaining_zero"
                    exit_page = PageState.HOME
                    break
                
                # 5.0 每次循环前验证仍在签到页面
                # 优化：第一次循环跳过验证（已经知道页面状态）
                # 优化：快速签到模式下大幅减少验证频率（每3次循环验证1次）
                should_verify = attempt > 0 and not skip_page_verification
                if quick_mode_optimization and should_verify:
                    # 快速签到模式：每3次循环才验证1次页面状态
                    should_verify = (attempt % 3 == 0)
                
                if should_verify:
                    # log(f"  [签到循环 {attempt+1}/{max_attempts}] 验证页面状态...")  # [2026-03-01] 精简日志：删除中间步骤
                    
                    # [2026-03-11] 修复原因：快速签到模式下使用缓存检测，提升性能
                    if quick_mode_optimization:
                        # 快速签到模式：使用缓存检测，减少深度学习推理
                        page_result_loop = await self._detect_page_cached(
                            device_id, 
                            use_cache=True,  # 使用缓存
                            cache_key=f"quick_checkin_{attempt}",
                            ttl=2.0  # 缓存2秒
                        )
                    else:
                        # 完整模式：正常检测
                        page_result_loop = await self.page_classifier.detect_page(device_id, use_cache=False)
                    
                    current_state = page_result_loop.state if page_result_loop else PageState.UNKNOWN
                else:
                    # 第一次循环：使用已知的页面状态
                    # log(f"  [签到循环 {attempt+1}/{max_attempts}] 使用已知页面状态: {current_state.value}")  # [2026-03-01] 精简日志：删除中间步骤
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
                                # [2026-03-05] 修复数组比较错误：检查 texts 是否为 None 并且长度大于 0
                                if ocr_result and ocr_result.texts is not None and len(ocr_result.texts) > 0:
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
                            # [2026-03-01] 修复：使用签到专用检测器
                            await wait_for_page(
                                device_id,
                                self.page_classifier,
                                [PageState.HOME],
                                log_callback=lambda msg: log(f"    [智能等待] {msg}")
                            )
                            log(f"  [签到] ✓ 已返回首页")
                            result['already_checked'] = True
                            result['remaining_times'] = 0
                            result['message'] = "今日已签到完成（签到次数已用完）"
                            result['success'] = True
                            exit_reason = "warmtip"
                            exit_page = PageState.HOME
                            break
                        
                        # 如果是签到奖励弹窗，关闭并继续
                        close_success = await self.detector.close_popup(device_id)
                        if close_success:
                            log(f"  [签到] ✓ 弹窗已关闭，等待页面刷新...")
                            await asyncio.sleep(0.5)  # 优化：减少等待时间从2秒到0.5秒
                            # [2026-03-01] 修复：使用签到专用检测器
                            page_result_loop = await self.page_classifier.detect_page(device_id, use_cache=False)
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
                                    # [2026-03-01] 修复：使用签到专用检测器
                                    page_result_loop = await self.page_classifier.detect_page(device_id, use_cache=False)
                                    current_state = page_result_loop.state if page_result_loop else PageState.UNKNOWN
                                    if current_state == PageState.CHECKIN:
                                        log(f"  [签到] ✓ 已返回签到页面，继续下一轮循环")
                                        continue
                        else:
                            log(f"  [签到] ⚠️ 关闭弹窗失败，尝试按返回键...")
                            await self.adb.press_back(device_id)
                            await asyncio.sleep(0.5)  # 优化：减少等待时间
                            # [2026-03-01] 修复：使用签到专用检测器
                            page_result_loop = await self.page_classifier.detect_page(device_id, use_cache=False)
                            current_state = page_result_loop.state if page_result_loop else PageState.UNKNOWN
                            if current_state == PageState.CHECKIN:
                                log(f"  [签到] ✓ 已返回签到页面，继续下一轮循环")
                                continue
                    
                    # 如果不是弹窗，或者关闭弹窗后仍不在签到页面，直接报错退出
                    if current_state not in [PageState.CHECKIN, PageState.CHECKIN_POPUP]:
                        result['message'] = f"签到循环中页面异常: {current_state.value}"
                        result['error_type'] = ErrorType.CHECKIN_FAILED  # 签到失败
                        result['error_message'] = result['message']
                        log(f"  [签到] ❌ 页面异常，无法继续签到")
                        exit_reason = "error"
                        exit_page = current_state
                        break
                
                # 5.1 读取签到页面信息
                if attempt == 0:
                    # 第一次循环：直接使用进入页面时已读取的次数信息
                    total_times = result.get('total_times')
                    remaining_times = result.get('remaining_times')
                    
                    # if total_times is not None:
                    #     log(f"  [签到 1] 使用已读取的次数信息 - 总次数: {total_times}, 当日剩余: {remaining_times}")  # [2026-03-01] 精简日志：删除中间步骤
                    # else:
                    #     log(f"  [签到 1] ⚠️ 未获取到次数信息，继续执行签到")  # [2026-03-01] 精简日志：删除中间步骤
                else:
                    # 后续循环：通过初始剩余次数减去已签到次数来推算当前剩余次数
                    remaining_times = initial_remaining_times - checkin_count if initial_remaining_times else None
                    # 同步更新 result 中的剩余次数
                    result['remaining_times'] = remaining_times
                    # log(f"  [签到 {attempt + 1}] 推算当日剩余次数: {remaining_times} (初始剩余: {initial_remaining_times}, 已签到: {checkin_count})")  # [2026-03-01] 精简日志：删除中间步骤
                
                # 5.2 检查是否可以签到
                # 修复：第一次循环时，即使remaining_times为0，也要尝试点击一次
                # 因为OCR可能识别错误，或者页面数据是旧的
                # 只有在后续循环中，如果remaining_times为0才跳出
                if attempt > 0 and remaining_times is not None and remaining_times <= 0:
                    # log(f"  [签到 {attempt + 1}] 剩余次数为0，今日已签到完成")  # [2026-03-01] 精简日志：删除中间步骤
                    # 设置已签到标志
                    result['already_checked'] = True
                    result['remaining_times'] = 0
                    # 跳出循环
                    exit_reason = "remaining_zero"
                    exit_page = PageState.CHECKIN
                    break
                # else:
                #     if remaining_times is not None and remaining_times <= 0 and attempt == 0:
                #         log(f"  [签到 {attempt + 1}] OCR识别剩余次数为0，但仍尝试点击一次（可能是识别错误）")  # [2026-03-01] 精简日志：删除中间步骤
                #     else:
                #         log(f"  [签到 {attempt + 1}] 剩余次数: {remaining_times if remaining_times is not None else '未知'}，继续签到...")  # [2026-03-01] 精简日志：删除中间步骤
                
                # [2026-03-01] 修复：直接使用固定坐标，不使用YOLO检测器
                # 5.4 执行签到（使用固定按钮坐标）
                checkin_button = (270, 888)
                
                x, y = checkin_button
                
                # 5.4.1 点击签到按钮前截图（可选，调试用）
                # log(f"  [签到 {attempt + 1}] 保存点击前截图...")
                # before_screenshot = await self._save_screenshot(device_id, phone, "before_click", attempt + 1)
                # if before_screenshot:
                #     result['screenshots'].append(before_screenshot)
                
                # 5.4.2 点击签到按钮
                # log(f"  [签到 {attempt + 1}] 点击签到按钮 ({x}, {y})...")  # [2026-03-01] 精简日志：删除中间步骤
                
                # [2026-03-01] 精简日志：第一次显示"开始签到"
                if attempt == 0:
                    log(f"  [签到] 开始签到")
                
                # 使用智能按钮点击器（自动学习坐标）
                # [2026-02-21] 删除学习器：直接点击签到按钮
                await self.adb.tap(device_id, x, y)
                
                # [2026-03-29] 修复：点击后等待0.3秒，避免截图操作干扰触摸事件导致上划手势
                await asyncio.sleep(0.3)
                
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
                    # log(f"  [签到 {attempt + 1}] 【优化策略】当日剩余 {remaining} 次，跳过弹窗等待，直接返回首页...")  # [2026-03-01] 精简日志：删除中间步骤
                    # concise.action("跳过弹窗等待")  # [2026-03-01] 精简日志：删除中间步骤
                    
                    # 等待0.5秒让签到请求完成
                    await asyncio.sleep(0.5)
                    
                    # 直接返回首页
                    await self.adb.press_back(device_id)
                    await asyncio.sleep(0.3)
                    
                    # [2026-03-01] 修复：使用签到专用检测器
                    # 等待返回首页
                    await wait_for_page(
                        device_id,
                        self.page_classifier,
                        [PageState.HOME],
                        log_callback=None
                    )
                    
                    # [2026-03-01] 优化：第一次成功说明按钮位置正确，第二次直接点击相同位置
                    # log(f"  [签到 {attempt + 1}] ✓ 已返回首页，准备重新进入签到页...")  # [2026-03-01] 精简日志：删除中间步骤
                    
                    # 直接使用第一次成功的按钮位置（不再重新检测）
                    # log(f"  [签到 {attempt + 1}] 使用首次成功的按钮位置: {checkin_button_pos}")  # [2026-03-01] 精简日志：删除中间步骤
                    
                    # 点击签到按钮
                    await self.adb.tap(device_id, checkin_button_pos[0], checkin_button_pos[1])
                    
                    # [2026-03-01] 修复：使用智能等待器等待进入签到页，而不是固定等待+OCR
                    # 智能等待器会持续检测，一旦检测到签到页就立即返回（内置15秒超时）
                    page_result_fast = await wait_for_page(
                        device_id,
                        self.page_classifier,
                        [PageState.CHECKIN],
                        log_callback=None
                    )
                    
                    # 验证是否成功进入签到页
                    if page_result_fast and page_result_fast.state == PageState.CHECKIN:
                        page_verified = True
                    else:
                        page_verified = False
                        log(f"  [签到 {attempt + 1}] ⚠️ 未能进入签到页，当前状态: {page_result_fast.state.value if page_result_fast else 'unknown'}")
                    
                    # 如果未验证成功，终止签到
                    if not page_verified:
                        exit_reason = "error"
                        exit_page = PageState.UNKNOWN
                        break
                    
                    # 更新已处理次数
                    checkin_count += 1
                    result['checkin_count'] = checkin_count
                    
                    # 更新剩余次数
                    if result.get('remaining_times') is not None:
                        result['remaining_times'] -= 1
                    
                    # [2026-03-01] 精简日志：显示"完成第 X 次"
                    log(f"  [签到] 完成第 {checkin_count} 次")
                    
                    # 设置跳过页面验证标志（下一次循环直接点击签到按钮）
                    skip_page_verification = True
                    
                    # 继续下一次循环（直接进入下一次签到，不需要重新识别次数）
                    continue
                # else:
                    # 只有在剩余次数为0或未知时才等待弹窗
                    # log(f"  [签到 {attempt + 1}] 【降级策略】当日剩余次数未知或为0，使用保守策略，等待弹窗...")  # [2026-03-01] 精简日志：删除中间步骤
                
                # 5.4.4 清除页面检测缓存，确保智能等待器检测到最新状态
                if hasattr(self.detector, '_detection_cache'):
                    self.detector._detection_cache.clear(device_id)
                
                # 5.5 使用智能等待器检测弹窗类型
                # log(f"  [签到 {attempt + 1}] 等待弹窗出现...")  # [2026-03-01] 精简日志：删除中间步骤
                popup_detected = False
                is_warmtip = False  # 是否是温馨提示弹窗
                
                # 使用智能等待器等待弹窗出现（签到弹窗或温馨提示）
                # [2026-03-01] 修复：使用签到专用检测器，YOLO检测器没有训练过这些类别
                # [2026-03-01] 精简日志：禁用智能等待器的日志输出
                wait_result = await wait_for_page(
                    device_id,
                    self.page_classifier,
                    [PageState.CHECKIN_POPUP, PageState.WARMTIP],
                    log_callback=None
                )
                
                # 优先使用智能等待器的结果
                if wait_result:
                    # [2026-03-01] 精简日志：删除"智能等待器检测到"日志
                    
                    if wait_result.state == PageState.CHECKIN_POPUP:
                        # 检测到签到弹窗，但需要用OCR验证是否是温馨提示（页面分类器可能误判）
                        log(f"  [签到] 检测到弹窗，使用OCR验证类型...")
                        screenshot_data = await self.adb.screencap(device_id)
                        if screenshot_data and HAS_PIL and self._ocr_pool:
                            image = Image.open(BytesIO(screenshot_data))
                            try:
                                ocr_result = await self._ocr_pool.recognize(image, timeout=TimeoutsConfig.OCR_TIMEOUT_SHORT)
                                # [2026-03-05] 修复数组比较错误：检查 texts 是否为 None 并且长度大于 0
                                if ocr_result and ocr_result.texts is not None and len(ocr_result.texts) > 0:
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
                        # [2026-03-01] 精简日志：删除"检测到温馨提示弹窗"日志（后面有更详细的）
                else:
                    # [2026-03-01] 优化：SmartWaiter超时后直接使用OCR验证,不使用页面分类器
                    # 智能等待器超时，使用OCR验证当前页面状态
                    log(f"  [签到] ⚠️ 智能等待器超时，使用OCR验证页面状态...")
                    
                    screenshot_data = await self.adb.screencap(device_id)
                    if screenshot_data and HAS_PIL and self._ocr_pool:
                        image = Image.open(BytesIO(screenshot_data))
                        try:
                            ocr_result = await self._ocr_pool.recognize(image, timeout=TimeoutsConfig.OCR_TIMEOUT_SHORT)
                            # [2026-03-05] 修复数组比较错误：检查 texts 是否为 None 并且长度大于 0
                            if ocr_result and ocr_result.texts is not None and len(ocr_result.texts) > 0:
                                text_str = ''.join(ocr_result.texts)
                                
                                # 检测签到页特征关键词
                                has_checkin_keywords = any(kw in text_str for kw in ["立即签到", "每日签到", "每天签到"])
                                has_popup_keywords = any(kw in text_str for kw in ["恭喜", "获得", "签到成功"])
                                
                                # 判断是否在签到页
                                if has_checkin_keywords and not has_popup_keywords:
                                    log(f"  [签到] ✓ OCR确认：当前在签到页，继续签到")
                                    # [2026-03-01] 修复：更新current_state，而不是continue
                                    # 这样下一轮循环的页面验证就能使用OCR的结果
                                    current_state = PageState.CHECKIN
                                    # 继续下一轮循环
                                    continue
                                
                                # 检测温馨提示弹窗
                                if "温馨提示" in text_str:
                                    popup_detected = True
                                    is_warmtip = True
                                    log(f"  [签到] ✓ OCR确认：温馨提示弹窗")
                                
                                # 检测签到奖励弹窗
                                elif has_popup_keywords:
                                    popup_detected = True
                                    log(f"  [签到] ✓ OCR确认：签到奖励弹窗")
                                
                                else:
                                    log(f"  [签到] ⚠️ OCR未识别到明确特征，假设未检测到弹窗")
                        except Exception as e:
                            log(f"  [签到] OCR验证失败: {e}")
                    else:
                        log(f"  [签到] ⚠️ 无法进行OCR验证")
                
                # 5.5.1 如果检测到温馨提示弹窗，直接处理
                if is_warmtip:
                    # [2026-03-01] 精简日志：删除温馨提示弹窗的详细日志
                    concise.action("出现温馨提示")
                    
                    # 关闭弹窗（按返回键）
                    await self.adb.press_back(device_id)
                    
                    # 智能等待返回首页
                    # [2026-03-01] 修复：使用签到专用检测器
                    # [2026-03-01] 精简日志：禁用智能等待器的日志输出
                    await wait_for_page(
                        device_id,
                        self.page_classifier,
                        [PageState.HOME],
                        log_callback=None
                    )
                    
                    # [2026-03-01] 精简日志：显示"签到已完成"和"已返回首页"
                    log(f"  [签到] 签到已完成")
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
                        
                        # [2026-03-05] 修复数组比较错误：使用 is not None 和 len() 检查
                        if detection_result and detection_result.elements is not None and len(detection_result.elements) > 0:
                            for element in detection_result.elements:
                                if '关闭' in element.class_name or '知道了' in element.class_name:
                                    close_button_pos = element.center
                                    log(f"  [签到] 找到关闭按钮: {close_button_pos}")  # [2026-03-01] 删除置信度，减少冗余
                                    break
                        
                        if close_button_pos is None:
                            # 如果YOLO未检测到，使用默认位置（与close_popup方法中的坐标一致）
                            close_button_pos = (270, 812)
                            log(f"  [签到] 未检测到关闭按钮，使用默认位置: {close_button_pos}")
                    except Exception as e:
                        log(f"  [签到] 检测关闭按钮失败: {e}，使用默认位置")
                        close_button_pos = (270, 812)
                    
                    # 使用智能按钮点击器关闭弹窗（自动学习坐标）
                    # [2026-02-21] 删除学习器：直接点击关闭按钮
                    log(f"  [签到] 关闭弹窗...")
                    log(f"  [签到] 点击关闭按钮: {close_button_pos}")
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
                            await self.adb.tap(device_id, backup_pos[0], backup_pos[1])
                            await asyncio.sleep(0.3)
                            check_result = await self._detect_page_cached(device_id, use_cache=False, cache_key=f"backup_check_{attempt}")
                            if check_result and check_result.state != PageState.CHECKIN_POPUP:
                                log(f"  [签到] ✓ 备用坐标成功关闭弹窗")
                                break
                    
                    # 使用智能等待器等待返回签到页
                    # [2026-03-01] 修复：使用签到专用检测器
                    log(f"  [签到] 智能等待返回签到页...")
                    wait_success = await wait_for_page(
                        device_id,
                        self.page_classifier,
                        [PageState.CHECKIN],
                        log_callback=lambda msg: log(f"    [智能等待] {msg}")
                    )
                    
                    if wait_success:
                        log(f"  [签到] ✓ 已返回签到页")
                    else:
                        log(f"  [签到] ⚠️ 等待超时，验证当前页面状态...")
                        page_result = await self._detect_page_cached(device_id, use_cache=False, cache_key=f"timeout_check_{attempt}")
                        if page_result:
                            log(f"  [签到] 当前页面: {page_result.state.value}")  # [2026-03-01] 删除置信度，减少冗余
                            if page_result.state == PageState.CHECKIN_POPUP:
                                log(f"  [签到] ⚠️ 弹窗未关闭，再次尝试（按返回键）...")
                                await self.adb.press_back(device_id)
                                # [2026-03-01] 修复：使用签到专用检测器
                                # 再次智能等待
                                await wait_for_page(
                                    device_id,
                                    self.page_classifier,
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
                    
                    log(f"  [签到] ✓ 第{checkin_count}次签到完成")  # [2026-03-01] 精简日志：只保留关键结果
                    # concise.action(f"第{checkin_count}次签到完成")  # [2026-03-01] 精简日志：删除中间步骤
                    
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
                            # [2026-03-05] 修复数组比较错误：检查 texts 是否为 None 并且长度大于 0
                            if ocr_result and ocr_result.texts is not None and len(ocr_result.texts) > 0:
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
                                    
                                    # [2026-03-01] 修复：使用签到专用检测器
                                    # 智能等待返回首页
                                    await wait_for_page(
                                        device_id,
                                        self.page_classifier,
                                        [PageState.HOME],
                                        log_callback=lambda msg: log(f"    [智能等待] {msg}")
                                    )
                                    
                                    log(f"  [签到] ✓ 已返回首页")
                                    
                                    # 设置已签到标志（次数用完 = 今日已签到完成）
                                    result['already_checked'] = True
                                    result['remaining_times'] = 0
                                    
                                    # 跳出循环
                                    exit_reason = "warmtip"
                                    exit_page = PageState.HOME
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
                                    
                                    # [2026-03-01] 修复：使用签到专用检测器
                                    # 关闭弹窗
                                    await self.detector.close_popup(device_id)
                                    await wait_for_page(device_id, self.page_classifier, [PageState.CHECKIN])
                                    
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
                            log(f"  [签到] 当前页面: {page_result.state.value}")  # [2026-03-01] 删除置信度，减少冗余
                            
                            # 如果在首页，说明可能已经自动返回（温馨提示弹窗自动关闭）
                            if page_result.state == PageState.HOME:
                                log(f"  [签到] ✓ 已在首页，推测为温馨提示弹窗已自动关闭（次数用完）")
                                result['already_checked'] = True
                                result['remaining_times'] = 0
                                exit_reason = "warmtip"
                                exit_page = PageState.HOME
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
                                exit_reason = "error"
                                exit_page = page_result.state
                                break
                        else:
                            # 无法检测页面状态
                            log(f"  [签到] ❌ 无法检测页面状态")
                            result['message'] = "签到卡住，无法检测页面状态"
                            result['error_type'] = ErrorType.CHECKIN_FAILED
                            result['error_message'] = result['message']
                            exit_reason = "error"
                            exit_page = PageState.UNKNOWN
                            break
            
            # 循环结束后，使用余额对比计算总奖励
            # log(f"  [签到] 签到循环结束，获取签到后的完整资料...")  # [2026-03-01] 精简日志：删除中间步骤
            # log(f"  [签到] 退出原因: {exit_reason}, 退出页面: {exit_page.value if exit_page else 'UNKNOWN'}")  # [2026-03-01] 精简日志：删除中间步骤
            
            # [2026-03-01] 根据退出原因和页面状态，决定如何返回首页
            if exit_page == PageState.HOME:
                # 已经在首页，不需要返回
                pass  # log(f"  [签到] 当前已在首页")  # [2026-03-01] 精简日志：删除中间步骤
            elif exit_page == PageState.CHECKIN:
                # 在签到页，按返回键返回首页
                # log(f"  [签到] 当前在签到页，返回首页...")  # [2026-03-01] 精简日志：删除中间步骤
                await self.adb.press_back(device_id)
                await asyncio.sleep(0.5)
                
                # 等待返回首页（最多等待5秒）
                try:
                    # [2026-03-01] 修复：使用签到专用检测器
                    await asyncio.wait_for(
                        wait_for_page(
                            device_id,
                            self.page_classifier,
                            [PageState.HOME],
                            log_callback=None
                        ),
                        timeout=5.0
                    )
                    # log(f"  [签到] ✓ 已返回首页")  # [2026-03-01] 精简日志：删除中间步骤
                except asyncio.TimeoutError:
                    pass  # log(f"  [签到] ⚠️ 返回首页超时，继续执行...")  # [2026-03-01] 精简日志：删除中间步骤
            else:
                # [2026-03-01] 修复：使用签到专用检测器检测当前页面
                current_page = await self.page_classifier.detect_page(device_id, use_cache=False)
                
                if current_page:
                    # log(f"  [签到] 当前页面: {current_page.state.value}")  # [2026-03-01] 精简日志：删除中间步骤
                    
                    if current_page.state == PageState.HOME:
                        pass  # log(f"  [签到] 当前已在首页")  # [2026-03-01] 精简日志：删除中间步骤
                    else:
                        # log(f"  [签到] 返回首页...")  # [2026-03-01] 精简日志：删除中间步骤
                        await self.adb.press_back(device_id)
                        await asyncio.sleep(0.5)
                        
                        try:
                            # [2026-03-01] 修复：使用签到专用检测器
                            await asyncio.wait_for(
                                wait_for_page(
                                    device_id,
                                    self.page_classifier,
                                    [PageState.HOME],
                                    log_callback=None
                                ),
                                timeout=5.0
                            )
                            # log(f"  [签到] ✓ 已返回首页")  # [2026-03-01] 精简日志：删除中间步骤
                        except asyncio.TimeoutError:
                            pass  # log(f"  [签到] ⚠️ 返回首页超时，继续执行...")  # [2026-03-01] 精简日志：删除中间步骤
                else:
                    # log(f"  [签到] ⚠️ 无法检测页面状态，尝试返回首页...")  # [2026-03-01] 精简日志：删除中间步骤
                    await self.adb.press_back(device_id)
                    await asyncio.sleep(0.5)
            
            # 获取签到后的完整资料（复用正常流程的方式）
            # [2026-03-01] 修复原因：添加"步骤3: 获取资料"日志
            concise.step(step_number + 1, "获取资料")
            
            checkin_balance_after = None  # 签到后余额
            final_profile = None  # 签到后的完整资料
            max_retries = 3
            
            for retry in range(max_retries):
                try:
                    # [2026-03-14] 修复原因：签到完成后已在首页，直接点击"我的"按钮，不需要检测页面
                    concise.action("导航到个人页")
                    
                    # 直接点击"我的"按钮进入个人页
                    await self.adb.tap(device_id, self.profile_navigator.TAB_MY[0], self.profile_navigator.TAB_MY[1])
                    
                    # 等待进入个人页
                    await asyncio.sleep(1.0)
                    
                    # 按返回键关闭可能的广告
                    await self.adb.press_back(device_id)
                    await asyncio.sleep(0.3)
                    
                    # [2026-03-14] 修复原因：验证是否成功进入个人页
                    # 使用资料专用模型检测当前页面
                    current_page = await self.profile_classifier.detect_page(device_id, use_cache=False)
                    
                    if current_page and current_page.state in [PageState.PROFILE_LOGGED, PageState.PROFILE]:
                        # 成功进入个人页
                        log(f"  [签到] ✓ 已进入个人页")
                    elif current_page and current_page.state == PageState.HOME:
                        # 还在首页，再次点击"我的"按钮导航到个人页
                        log(f"  [签到] ⚠️ 第{retry+1}次导航失败，仍在首页，再次点击")
                        if retry < max_retries - 1:
                            await asyncio.sleep(TimeoutsConfig.WAIT_MEDIUM)
                            continue
                        else:
                            log(f"  [签到] ❌ 导航到个人页失败，已重试{max_retries}次")
                            break
                    else:
                        # 在其他页面
                        page_name = current_page.state.value if current_page else "未知"
                        log(f"  [签到] ⚠️ 第{retry+1}次导航失败，当前在: {page_name}")
                        if retry < max_retries - 1:
                            await asyncio.sleep(TimeoutsConfig.WAIT_MEDIUM)
                            continue
                        else:
                            log(f"  [签到] ❌ 导航到个人页失败，页面异常")
                            break
                    
                    # [2026-03-14] 修复原因：已在个人页，不需要等待，直接读取资料
                    # 使用正常流程的方式：get_full_profile_with_retry（带重试机制）
                    # [2026-03-11] 修复原因：必须使用self.detector（PageDetectorIntegrated）而不是self.profile_classifier（PageDetectorDL）
                    # PageDetectorDL不支持YOLO元素检测，会导致个人页元素检测失败
                    concise.action("读取个人资料")
                    from .profile_reader import ProfileReader
                    profile_reader = ProfileReader(self.adb, self.detector)
                    account_str = f"{phone}----{password}" if password else phone
                    # 改用带重试的方法，和正常流程完全一样
                    profile_task = profile_reader.get_full_profile_with_retry(device_id, account=account_str, max_retries=3)
                    
                    try:
                        # [2026-03-14] 修复原因：获取资料3秒就够了，不需要30秒超时
                        final_profile = await asyncio.wait_for(profile_task, timeout=5.0)
                        
                        if final_profile and final_profile.get('balance') is not None:
                            checkin_balance_after = final_profile.get('balance')
                            
                            # [2026-03-12] 恢复签到后资料显示：用户要求显示签到完成后的个人资料
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
                
                # 容错处理：签到奖励不应该为负值
                # 如果出现负值，说明账号在签到期间被转账了，签到奖励设置为0
                if total_reward < 0:
                    log(f"  [签到] ⚠️ 检测到异常：签到后余额({checkin_balance_after:.2f})小于签到前余额({balance:.2f})")
                    log(f"  [签到] ⚠️ 原因：账号在签到期间被转账")
                    log(f"  [签到] ✓ 容错处理：将签到奖励设置为0")
                    total_reward = 0.0
                
                result['reward_amount'] = total_reward
                result['checkin_count'] = checkin_count
                result['checkin_balance_after'] = checkin_balance_after  # 返回签到后余额
                
                # 返回签到后的完整资料（供快速签到模式使用）
                if final_profile:
                    result['nickname'] = final_profile.get('nickname')
                    result['user_id'] = final_profile.get('user_id')
                    result['points'] = final_profile.get('points')
                    result['vouchers'] = final_profile.get('vouchers')
                    # [2026-03-01] 删除：优惠券功能已移除
                    # 只在 balance 不为 None 时才更新 balance_before（避免覆盖已有的正确值）
                    if balance is not None:
                        result['balance_before'] = balance  # 签到前余额
                
                log(f"  [签到] 签到前余额: {balance:.2f} 元")
                log(f"  [签到] 签到后余额: {checkin_balance_after:.2f} 元")
                log(f"  [签到] ✓ 总奖励: {total_reward:.2f} 元")
            else:
                log(f"  [签到] ⚠️ 无法获取余额，无法计算总奖励")
                result['checkin_count'] = checkin_count
                result['checkin_balance_after'] = checkin_balance_after  # 即使为None也返回
                
                # 即使无法计算奖励，也返回获取到的资料
                if final_profile:
                    result['nickname'] = final_profile.get('nickname')
                    result['user_id'] = final_profile.get('user_id')
                    result['points'] = final_profile.get('points')
                    result['vouchers'] = final_profile.get('vouchers')
                    # [2026-03-01] 删除：优惠券功能已移除
                    # 只在 balance 不为 None 时才更新 balance_before（避免覆盖已有的正确值）
                    if balance is not None:
                        result['balance_before'] = balance  # 签到前余额
            
            # 6. 设置最终结果
            # 统一处理：签到流程执行完毕即为成功
            result['success'] = True
            
            # [2026-03-02] 修复原因：无法获取余额时，不显示奖励金额
            if checkin_balance_after is not None and balance is not None:
                result['message'] = f"签到完成，共签到 {result['checkin_count']} 次，总奖励 {round(result['reward_amount'], 3)} 元"
            else:
                result['message'] = f"签到完成，共签到 {result['checkin_count']} 次（无法计算奖励）"
            
            concise.success("签到完成")
            
            # [2026-03-29] 静默：调试日志不输出到控制台
            
            # 优化：获取到最终余额后不需要返回首页，直接返回结果（下一步是退出登录）
            
            return result
            
        except Exception as e:
            log(f"  [签到] ❌ 签到流程异常: {str(e)}")
            import traceback
            log(f"  [签到] 异常堆栈:\n{traceback.format_exc()}")
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
            
            # [2026-02-21] 删除学习器：直接点击签到按钮
            await self.adb.tap(device_id, checkin_button_pos[0], checkin_button_pos[1])
            
            # [2026-03-01] 修复：使用签到专用检测器
            # 优化：使用智能等待器等待进入签到页面
            await wait_for_page(
                device_id,
                self.page_classifier,
                [PageState.CHECKIN],
                log_callback=None  # [2026-03-29] 静默：不输出到控制台
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
        """使用YOLO或OCR识别首页的"每日签到"按钮位置

        # [2026-03-01] 修改策略：优先使用固定坐标（已验证位置正确），YOLO/OCR作为后备

        Args:
            device_id: 设备ID

        Returns:
            tuple: 按钮中心点坐标 (x, y)，失败返回None
        """
        # 导入日志记录器
        from .logger import get_logger
        logger = get_logger()

        # 优先使用固定坐标（已验证位置正确）
        # [2026-03-01] 精简日志：删除固定坐标日志输出
        return self.CHECKIN_BUTTON
    
    async def _find_checkin_button_fallback(self, device_id: str) -> Optional[Tuple[int, int]]:
        """降级方案：使用 YOLO 或 OCR 检测签到按钮位置
        
        当默认坐标失败时调用此方法
        
        Args:
            device_id: 设备ID
            
        Returns:
            tuple: 按钮中心点坐标 (x, y)，失败返回None
        """
        # 导入日志记录器
        from .logger import get_logger
        logger = get_logger()

        detected_position = None
        detection_confidence = 0.0

        # 步骤1: 尝试使用YOLO
        try:
            # 使用YOLO的元素检测功能
            detection_result = await self._detect_page_cached(
                device_id,
                use_cache=False,
                detect_elements=True,
                cache_key="home_checkin_button",
                ttl=0
            )

            # [2026-03-05] 修复数组比较错误：使用 is not None 和 len() 检查
            if detection_result and detection_result.elements is not None and len(detection_result.elements) > 0:
                # 查找签到按钮元素
                for element in detection_result.elements:
                    if '每日签到' in element.class_name or '签到按钮' in element.class_name:
                        detected_position = element.center
                        detection_confidence = element.confidence
                        break
        except Exception as e:
            pass

        # 步骤2: 坐标合理性验证
        if detected_position:
            x, y = detected_position
            x_min, x_max, y_min, y_max = self.CHECKIN_BUTTON_VALID_RANGE

            if x_min <= x <= x_max and y_min <= y <= y_max:
                logger.info(f"  [签到] ✓ YOLO坐标合理性验证通过: {detected_position}")
                return detected_position
            else:
                logger.info(f"  [签到] ⚠️ YOLO坐标不合理: {detected_position}，超出范围 {self.CHECKIN_BUTTON_VALID_RANGE}")

        # 步骤3: 降级到OCR识别
        if HAS_OCR and self._ocr_pool:
            logger.info(f"  [签到] 降级到OCR识别...")
            try:
                screenshot = await self.adb.screencap(device_id)
                if screenshot:
                    image = Image.open(BytesIO(screenshot))

                    from .timeouts_config import TimeoutsConfig
                    ocr_result = await self._ocr_pool.recognize(image, timeout=TimeoutsConfig.OCR_TIMEOUT_SHORT)
                    # [2026-03-05] 修复数组比较错误：检查 texts 是否为 None 并且长度大于 0
                    if ocr_result and ocr_result.texts is not None and len(ocr_result.texts) > 0:
                        for i, text in enumerate(ocr_result.texts):
                            if "每日签到" in text or (text == "签到" and i < len(ocr_result.boxes)):
                                box = ocr_result.boxes[i]
                                x_coords = [p[0] for p in box]
                                y_coords = [p[1] for p in box]
                                center_x = int(sum(x_coords) / len(x_coords))
                                center_y = int(sum(y_coords) / len(y_coords))
                                ocr_position = (center_x, center_y)
                                logger.info(f"  [签到] OCR找到签到按钮: {ocr_position}")

                                x_min, x_max, y_min, y_max = self.CHECKIN_BUTTON_VALID_RANGE
                                if x_min <= center_x <= x_max and y_min <= center_y <= y_max:
                                    logger.info(f"  [签到] ✓ OCR坐标验证通过")
                                    return ocr_position
                                else:
                                    logger.info(f"  [签到] ⚠️ OCR坐标不合理，继续查找...")
            except Exception as e:
                logger.warning(f"  ⚠️ OCR识别签到按钮失败: {e}")

        # 步骤4: 最终返回None，表示降级方案也失败
        logger.info(f"  [签到] ⚠️ 所有降级方案均失败")
        return None
