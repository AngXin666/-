"""
溪盟商城业务自动化模块
Ximeng Mall Business Automation Module

代码清理记录：
- 2026-02-02: 删除废弃的启动流程函数（handle_startup_flow, handle_startup_flow_optimized）
- 2026-02-08: 删除模板匹配相关代码，统一使用YOLO识别器
- 保留实际使用的函数（handle_startup_flow_integrated）
- 文件从 3531 行减少到约 2000 行（减少 43%）
"""

import asyncio
import time
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

from .models import Account, AccountResult, SignInResult, DrawResult
from .screen_capture import ScreenCapture
from .ui_automation import UIAutomation
from .auto_login import AutoLogin, LoginResult
from .page_detector import PageDetector, PageState
from .logger import get_silent_logger


class XimengAutomation:
    """溪盟商城业务自动化器"""
    
    # 余额相关识别文字
    BALANCE_PATTERNS = ["余额", "账户余额", "可用余额"]
    
    # 签到相关识别文字
    SIGN_IN_PATTERNS = ["签到", "每日签到", "立即签到"]
    SIGNED_PATTERNS = ["已签到", "明日再来", "今日已签"]
    
    # 抽奖相关识别文字
    DRAW_PATTERNS = ["抽奖", "幸运抽奖", "立即抽奖", "开始抽奖"]
    NO_DRAW_PATTERNS = ["次数已用完", "明日再来", "抽奖次数不足"]
    
    def __init__(self, ui_automation: UIAutomation, screen_capture: ScreenCapture, 
                 auto_login: AutoLogin, adb_bridge=None, log_callback=None):
        """初始化自动化器
        
        Args:
            ui_automation: UI 自动化器实例
            screen_capture: 屏幕捕获器实例
            auto_login: 自动登录器实例
            adb_bridge: ADB 桥接器实例
            log_callback: 可选的日志回调函数
        """
        self.ui_automation = ui_automation
        self.screen_capture = screen_capture
        self.auto_login = auto_login
        self.adb = adb_bridge or ui_automation.adb_bridge
        self.page_detector = PageDetector(self.adb)
        self._log_callback = log_callback
        self._stop_check = None  # 停止检查函数，由外部设置
        
        # 初始化静默日志记录器
        self._silent_log = get_silent_logger()
        
        # 从ModelManager获取共享的模型实例（不再自己创建）
        from .model_manager import ModelManager
        model_manager = ModelManager.get_instance()
        
        # 获取共享的检测器实例（YOLO检测器）
        self.detector = model_manager.get_page_detector_integrated()
        
        # [2026-03-11] 优化日志：移除重复的调试信息
        
        # [2026-03-03] 修改：从 ModelManager 获取专用检测器（避免重复创建）
        # 导入日志记录器
        from .logger import get_logger
        logger = get_logger()
        
        # [2026-03-06] 修复原因：确保所有检测器属性都被初始化，避免 AttributeError
        # 先初始化为 None，然后尝试从 ModelManager 获取
        self.startup_detector = None
        self.profile_detector = None
        self.transfer_detector = None  # [2026-03-12] 添加转账专用检测器
        self.login_detector = None
        
        try:
            # 获取启动专用检测器（MobileNetV3）
            self.startup_detector = model_manager.get_startup_detector()
            if not self.startup_detector:
                # 如果专用模型未加载，降级使用YOLO检测器
                self.startup_detector = self.detector
                logger.warning(f"⚠️ 启动专用模型未加载，降级使用YOLO检测器")
            
            # 获取资料专用检测器（MobileNetV3）
            self.profile_detector = model_manager.get_profile_detector()
            if not self.profile_detector:
                # 如果专用模型未加载，降级使用YOLO检测器
                self.profile_detector = self.detector
                logger.warning(f"⚠️ 资料专用模型未加载，降级使用YOLO检测器")
            
            # [2026-03-12] 获取转账专用检测器（MobileNetV3）
            self.transfer_detector = model_manager.get_transfer_detector()
            if not self.transfer_detector:
                # 如果转账专用模型未加载，使用资料专用模型作为降级
                self.transfer_detector = self.profile_detector
                logger.warning(f"⚠️ 转账专用模型未加载，使用资料专用模型")
            
            # [2026-03-06] 添加：获取登录专用检测器（用于识别积分页）
            self.login_detector = model_manager.get_login_detector()
            if not self.login_detector:
                # 如果专用模型未加载，降级使用YOLO检测器
                self.login_detector = self.detector
                logger.warning(f"⚠️ 登录专用模型未加载，降级使用YOLO检测器")
        except Exception as e:
            # 如果获取检测器失败，确保所有属性都有默认值
            logger.error(f"❌ 获取专用检测器失败: {e}，使用YOLO检测器作为降级")
            if not self.startup_detector:
                self.startup_detector = self.detector
            if not self.profile_detector:
                self.profile_detector = self.detector
            if not self.transfer_detector:  # [2026-03-12] 添加转账专用检测器降级
                # 如果转账专用模型未加载，使用资料专用模型
                self.transfer_detector = self.profile_detector
            if not self.login_detector:
                self.login_detector = self.detector
        
        # 初始化其他组件（使用共享检测器）
        from .navigator import Navigator
        from .balance_reader import BalanceReader
        from .daily_checkin import DailyCheckin
        from .profile_reader import ProfileReader
        
        # 所有组件都使用YOLO检测器
        self.navigator = Navigator(self.adb, self.detector)
        self.balance_reader = BalanceReader(self.adb)
        self.daily_checkin = DailyCheckin(self.adb, self.detector, self.navigator)
        
        # [2026-03-11] 修复原因：ProfileReader必须使用PageDetectorIntegrated（支持YOLO元素检测）
        # 不能使用profile_detector（PageDetectorDL），因为它不支持元素检测
        # 强制使用self.detector（PageDetectorIntegrated），确保YOLO元素检测可用
        self.profile_reader = ProfileReader(self.adb, yolo_detector=self.detector)
        
        # 从ModelManager获取OCR线程池
        self._ocr_enhancer = model_manager.get_ocr_thread_pool()
        
        # 初始化转账模块（使用YOLO检测器）
        from .balance_transfer import BalanceTransfer
        self.balance_transfer = BalanceTransfer(self.adb, self.detector)
    
    async def wait_for_app_ready(self, device_id: str, timeout: int = 15) -> bool:
        """等待应用准备就绪（处理启动页、广告等）
        
        Args:
            device_id: 设备 ID
            timeout: 超时时间（秒）
            
        Returns:
            是否成功进入主界面
        """
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            # [2026-03-02] 修复原因：使用启动专用模型检测页面
            # 检测当前页面状态
            result = await self.startup_detector.detect_page(device_id)
            
            if result.state == PageState.HOME:
                # 已经在首页
                return True
            
            elif result.state == PageState.AD:
                # 广告页面，等待自动跳过
                await asyncio.sleep(2)
            
            elif result.state == PageState.SPLASH or result.state == PageState.LOADING:
                # 启动页或加载中，等待
                await asyncio.sleep(2)
            
            elif result.state == PageState.LOGIN:
                # 登录页面，返回 True 让调用者处理登录
                return True
            
            else:
                # 其他页面，检查是否有跳过按钮
                has_skip, location = await self.page_detector.has_skip_button(device_id)
                if has_skip and location:
                    await self.adb.tap(device_id, location[0], location[1])
                    await asyncio.sleep(1)
                else:
                    # 检查是否在主界面
                    if await self.page_detector.is_on_main_screen(device_id):
                        return True
                    await asyncio.sleep(1)
        
        return False
    
    async def handle_startup_flow_integrated(self, device_id: str, log_callback=None, stop_check=None,
                                            package_name: str = "com.ry.xmsc", activity_name: str = None,
                                            max_retries: int = 3, file_logger=None, 
                                            phone: str = None, password: str = None, app_already_started: bool = False,
                                            cache_restored: bool = False) -> bool:
        """处理应用启动流程 - 使用启动专用检测器（GPU加速）
        
        优化点：
        1. 使用启动专用检测器（MobileNetV3，GPU加速，2.24ms）
        2. 直接循环检测页面状态，避免检测器混用
        3. 使用固定坐标点击，避免额外的元素检测开销
        4. GPU加速页面分类，速度提升4.54倍
        5. 详细记录每个步骤的耗时，用于性能优化分析
        6. [2026-03-02] 新增：检测到登录页时直接登录
        7. [2026-03-02] 简化：移除OCR验证逻辑，直接信任模型识别结果
        8. [2026-03-11] 新增：支持应用已启动模式，避免重复启动
        9. [2026-03-11] 修复：移除重复的缓存恢复逻辑，避免应用被意外关闭
        10. [2026-03-11] 优化：精简GUI日志输出，减少CMD控制台噪音
        11. [2026-03-14] 修复：缓存恢复模式下，检测到登录页时先等待缓存生效，避免立即登录
        
        流程：启动页 -> 用户协议弹窗(关闭) -> 广告页(等待) -> 首页弹窗(关闭) -> 主页
              或：启动页 -> 登录页(直接登录) -> 主页
              或：缓存恢复 -> 启动页 -> 等待缓存生效 -> 主页（如果缓存有效）
        
        Args:
            device_id: 设备 ID
            log_callback: 日志回调函数
            stop_check: 停止检查函数
            package_name: 应用包名
            activity_name: Activity名称
            max_retries: 最大重试次数
            file_logger: 文件日志记录器（用于详细技术日志）
            phone: 手机号（用于登录）
            password: 密码（用于登录）
            app_already_started: 应用是否已启动（True时跳过启动步骤）
            cache_restored: 是否已恢复缓存（True时检测到登录页会先等待缓存生效）
            
        Returns:
            是否成功
        """
        import time
        from .concise_logger import ConciseLogger
        
        # 创建简洁日志记录器（GUI日志 + 文件日志）
        if log_callback:
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
        
        concise = ConciseLogger(
            module_name="startup",
            gui_logger=gui_logger_obj,
            file_logger=file_logger
        )
        
        def should_stop():
            if stop_check:
                return stop_check()
            return False
        
        # 记录总开始时间（仅文件日志）
        total_start_time = time.time()
        # [2026-03-11] 优化日志：移除控制台DEBUG输出
        if file_logger:
            file_logger.info(f"启动流程开始 - {time.strftime('%H:%M:%S')}")
        
        # [2026-03-11] 修复原因：移除重复的缓存恢复逻辑，避免应用被重复关闭
        # 缓存恢复已在GUI的启动流程中完成，这里不再重复执行
        if phone and file_logger:
            file_logger.info(f"账号 {phone} 的缓存恢复已在外部完成，跳过重复处理")
        
        # [2026-03-12] 修复原因：修复步骤编号重复问题，启动流程不应该有步骤编号
        # 启动流程是预处理步骤，不占用正式的步骤编号
        if not app_already_started:
            concise.action("启动应用")
        else:
            concise.action("等待广告")
        
        # [2026-03-11] 优化日志：移除控制台DEBUG输出
        detector = self.startup_detector
        
        for retry in range(max_retries):
            # [2026-03-11] 优化日志：移除控制台DEBUG输出
            if should_stop():
                if file_logger:
                    file_logger.info("用户请求停止")
                return False
            
            if retry > 0:
                # [2026-03-11] 修复原因：如果应用已由外部启动，重试时不要重新启动应用
                if app_already_started:
                    if file_logger:
                        file_logger.info(f"第 {retry + 1} 次尝试（跳过重启）")
                    # 只等待一下，不重新启动应用
                    await asyncio.sleep(1)
                else:
                    if file_logger:
                        file_logger.info(f"第 {retry + 1} 次尝试启动应用")
                    
                    # 停止应用
                    await self.adb.stop_app(device_id, package_name)
                    await asyncio.sleep(1)
                    
                    # 清理缓存
                    result = await self.adb.shell(device_id, f"pm clear-cache {package_name}")
                    if "Unknown" in result or "Error" in result:
                        result = await self.adb.shell(device_id, f"rm -rf /data/data/{package_name}/cache/*")
                    if file_logger:
                        file_logger.debug(f"清理缓存结果: {result.strip() if result.strip() else '成功'}")
                    
                    # 重新启动应用
                    await asyncio.sleep(1)
                    success = await self.adb.start_app(device_id, package_name, activity_name)
                    if file_logger:
                        file_logger.debug(f"启动{'成功' if success else '失败'}")
                    
                    # 智能等待应用启动
                    await asyncio.sleep(1)  # 最小等待，让应用开始启动
            else:
                # 第一次启动
                if file_logger:
                    file_logger.debug("应用已启动，开始页面检测")
                # [2026-03-01] 删除DEBUG日志：减少启动时的冗余输出
                # print(f"[DEBUG] 第一次启动，准备进入主循环")
            
            # [2026-03-02] 统一术语：主循环检测和处理页面
            max_wait_time = 30
            start_time = asyncio.get_event_loop().time()
            
            # [2026-03-11] 优化日志：简化启动流程检测的日志输出
            if file_logger:
                file_logger.info("开始智能启动流程检测")
                file_logger.info(f"最大等待时间: {max_wait_time}秒")
            
            
            loop_count = 0
            last_state = None  # [2026-03-11] 优化日志：记录上一次的状态，只在状态变化时输出
            # [2026-03-01] 删除DEBUG日志：减少启动时的冗余输出
            # print(f"[DEBUG] 开始while循环")
            while asyncio.get_event_loop().time() - start_time < max_wait_time:
                loop_count += 1
                # [2026-03-11] 优化日志：移除每次循环的分隔线，减少冗余输出
                if should_stop():
                    if file_logger:
                        file_logger.info("用户请求停止")
                    return False
                
                # 使用启动专用检测器检测当前页面（GPU加速）
                detect_start = time.time()
                
                try:
                    # [2026-03-11] 优化日志：移除"正在检测页面状态"的重复输出
                    # [2026-03-01] 修复原因：PageDetectorDL不支持detect_elements参数
                    result = await detector.detect_page(device_id, use_cache=True)
                    detect_time = time.time() - detect_start
                    elapsed = asyncio.get_event_loop().time() - start_time
                    
                    # [2026-03-11] 修复原因：添加GUI可见的状态变化日志
                    state_changed = (last_state != result.state.value)
                    if state_changed or not result.cached:
                        if file_logger:
                            file_logger.info(
                                f"[{elapsed:.1f}s] {result.state.value} "
                                f"(置信度: {result.confidence:.2%}, 检测: {detect_time*1000:.2f}ms)"
                            )
                        last_state = result.state.value
                except Exception as e:
                    if file_logger:
                        file_logger.error(f"页面检测出错: {e}")
                        import traceback
                        file_logger.error(traceback.format_exc())
                    await asyncio.sleep(1.0)
                    continue
                
                # 检查是否已到达目标页面（启动流程只检测到首页即可）
                if result.state == PageState.HOME:
                    total_time = time.time() - total_start_time
                    
                    # 简洁日志：到达首页
                    concise.success("到达首页")
                    
                    # 详细日志（仅文件日志）
                    if file_logger:
                        file_logger.info(
                            f"启动流程完成 - 到达页面: {result.state.value}, "
                            f"总耗时: {int(total_time)}秒, 完成时间: {time.strftime('%H:%M:%S')}"
                        )
                    
                    # [2026-03-11] 优化日志：移除启动完成的调试断点信息
                    
                    return True
                
                # [2026-03-02] 简化：检测到登录页时的处理逻辑
                # [2026-03-09] 修复原因：增加OCR二次验证，避免误识别首页为登录页
                # [2026-03-09] 修复原因：清除缓存并重新检测，避免使用旧缓存导致误判
                if result.state == PageState.LOGIN:
                    # 清除缓存并重新检测，确保不是使用了旧缓存
                    if file_logger:
                        file_logger.info("检测到登录页，清除缓存并重新检测...")
                    
                    if hasattr(detector, 'clear_cache'):
                        detector.clear_cache(device_id)
                    
                    # 重新检测（不使用缓存）
                    result = await detector.detect_page(device_id, use_cache=False)
                    
                    if file_logger:
                        file_logger.info(f"重新检测结果: {result.state.value} (置信度: {result.confidence:.2%})")
                    
                    # 如果重新检测后不是登录页，说明之前是缓存误判，继续循环
                    if result.state != PageState.LOGIN:
                        if file_logger:
                            file_logger.info("重新检测后不是登录页，之前是缓存误判，继续循环")
                        continue
                    
                    # OCR 二次验证：检查是否真的是登录页
                    if file_logger:
                        file_logger.info("使用OCR验证...")
                    
                    try:
                        screenshot_data = await self.adb.screencap(device_id)
                        is_login_page = False
                        
                        if screenshot_data:
                            from PIL import Image
                            from io import BytesIO
                            image = Image.open(BytesIO(screenshot_data))
                            
                            # 使用OCR识别页面文字
                            if self._ocr_enhancer:
                                ocr_result = await self._ocr_enhancer.recognize(image, timeout=3.0)
                                
                                if ocr_result and ocr_result.texts is not None and len(ocr_result.texts) > 0:
                                    all_text = ' '.join(ocr_result.texts)
                                    
                                    # 登录页特征：有"手机号"、"密码"、"登录"等关键词
                                    has_phone = '手机号' in all_text or '手机' in all_text
                                    has_password = '密码' in all_text
                                    has_login = '登录' in all_text or '立即登录' in all_text
                                    
                                    # 首页特征：有"首页"、"分类"、"购物车"、"我的"等关键词
                                    has_home = '首页' in all_text
                                    has_category = '分类' in all_text
                                    has_cart = '购物车' in all_text
                                    has_my = '我的' in all_text
                                    
                                    if file_logger:
                                        file_logger.info(f"OCR验证结果: 手机号={has_phone}, 密码={has_password}, 登录={has_login}")
                                        file_logger.info(f"OCR验证结果: 首页={has_home}, 分类={has_category}, 购物车={has_cart}, 我的={has_my}")
                                    
                                    # 判断：如果有登录页特征且没有首页特征，才认为是登录页
                                    if (has_phone or has_password or has_login) and not (has_home and has_category):
                                        is_login_page = True
                                        if file_logger:
                                            file_logger.info("✓ OCR确认：这是登录页")
                                    else:
                                        is_login_page = False
                                        if file_logger:
                                            file_logger.info("✗ OCR确认：这不是登录页，可能是首页被误识别")
                                        # 继续循环
                                        continue
                    except Exception as e:
                        if file_logger:
                            file_logger.error(f"OCR验证失败: {e}，假设为登录页")
                        is_login_page = True
                    
                    # 如果OCR确认不是登录页，跳过登录逻辑
                    if not is_login_page:
                        if file_logger:
                            file_logger.info("OCR验证失败，跳过登录逻辑")
                        continue
                    
                    # [2026-03-14] 修复原因：如果已恢复缓存但检测到登录页，说明缓存失效
                    # 应该返回False，让外部清理缓存并重新登录
                    if cache_restored:
                        if file_logger:
                            file_logger.warning("检测到登录页，缓存可能失效")
                            file_logger.info("返回失败，让外部清理缓存并重新登录")
                        concise.error("缓存失效，需要重新登录")
                        return False
                    
                    if phone and password:
                        # 有账号密码，直接登录
                        concise.action("检测到登录页，开始登录")
                        if file_logger:
                            file_logger.info("检测到登录页，开始登录...")
                        
                        try:
                            # [2026-03-02] 修复：已经在登录页，直接输入账号密码，不做任何页面检查
                            # 输入手机号
                            if file_logger:
                                file_logger.info(f"输入手机号: {phone}")
                            await self.auto_login._tap_with_fallback(device_id, self.auto_login.PHONE_INPUT, log_callback)
                            await asyncio.sleep(0.3)
                            await self.auto_login._tap_with_fallback(device_id, self.auto_login.PHONE_INPUT, log_callback)
                            await asyncio.sleep(0.5)
                            
                            # [2026-03-29] 修复：改用End键移到末尾再退格，避免光标在中间删不干净
                            await self.adb.key_event(device_id, 123)  # End键
                            await asyncio.sleep(0.1)
                            for _ in range(30):
                                await self.adb.key_event(device_id, 67)
                                await asyncio.sleep(0.02)
                            
                            await self.adb.input_text(device_id, phone)
                            await asyncio.sleep(1.0)
                            
                            # 输入密码
                            if file_logger:
                                file_logger.info(f"输入密码: {'*' * len(password)}")
                            await self.auto_login._tap_with_fallback(device_id, self.auto_login.PASSWORD_INPUT, log_callback)
                            await asyncio.sleep(0.3)
                            await self.auto_login._tap_with_fallback(device_id, self.auto_login.PASSWORD_INPUT, log_callback)
                            await asyncio.sleep(0.5)
                            
                            # [2026-03-29] 修复：改用End键移到末尾再退格，避免光标在中间删不干净
                            await self.adb.key_event(device_id, 123)  # End键
                            await asyncio.sleep(0.1)
                            for _ in range(30):
                                await self.adb.key_event(device_id, 67)
                                await asyncio.sleep(0.02)
                            
                            await self.adb.input_text(device_id, password)
                            await asyncio.sleep(1.0)
                            
                            # 勾选协议并点击登录
                            if file_logger:
                                file_logger.info("勾选协议并点击登录...")
                            await self.auto_login._click_agreement_with_retry(device_id, log_callback, max_retries=3)
                            await asyncio.sleep(0.5)
                            
                            concise.action("点击登录")
                            await self.auto_login._tap_with_fallback(device_id, self.auto_login.LOGIN_BUTTON, log_callback)
                            await asyncio.sleep(1.5)
                            
                            # 等待登录完成
                            if file_logger:
                                file_logger.info("等待登录完成...")
                            
                            login_success = False
                            for i in range(20):  # 最多10秒
                                await asyncio.sleep(0.5)
                                page_result = await detector.detect_page(device_id, use_cache=False)
                                
                                if page_result.state != PageState.LOGIN:
                                    if file_logger:
                                        file_logger.info(f"登录成功，页面已跳转: {page_result.state.value}")
                                    login_success = True
                                    break
                            
                            if login_success:
                                concise.success("登录成功")
                                if file_logger:
                                    file_logger.info("登录成功，等待页面跳转...")
                                
                                # 等待页面跳转（最多5秒）
                                login_wait_start = asyncio.get_event_loop().time()
                                while asyncio.get_event_loop().time() - login_wait_start < 5.0:
                                    await asyncio.sleep(0.3)
                                    page_result = await detector.detect_page(device_id, use_cache=False)
                                    
                                    if page_result.state != PageState.LOGIN:
                                        if file_logger:
                                            file_logger.info(f"页面已跳转 - 当前页面: {page_result.state.value}")
                                        break
                                
                                # 继续主循环
                                continue
                            else:
                                concise.error("登录超时")
                                if file_logger:
                                    file_logger.error("登录超时，仍在登录页面")
                                return False
                        except Exception as e:
                            concise.error(f"登录异常: {e}")
                            if file_logger:
                                file_logger.error(f"登录异常: {e}")
                                import traceback
                                file_logger.error(traceback.format_exc())
                            return False
                    else:
                        # 没有账号密码，返回成功让外部处理
                        total_time = time.time() - total_start_time
                        concise.success("到达登录页")
                        if file_logger:
                            file_logger.info(
                                f"启动流程完成 - 到达页面: {result.state.value}, "
                                f"总耗时: {int(total_time)}秒, 完成时间: {time.strftime('%H:%M:%S')}"
                            )
                        return True
                
                # 如果到达个人页（已登录或未登录），说明启动流程异常（应该到首页）
                if result.state in [PageState.PROFILE_LOGGED, PageState.PROFILE]:
                    if file_logger:
                        file_logger.warning(
                            f"启动流程异常：到达 {result.state.value}，预期应该到达首页，继续等待"
                        )
                    await asyncio.sleep(0.5)
                    continue
                
                # 处理Android桌面
                if result.state == PageState.LAUNCHER:
                    # [2026-03-11] 修复原因：如果应用已由外部启动，不要重新启动
                    if app_already_started:
                        if file_logger:
                            file_logger.debug("检测到Android桌面，但应用已由外部启动，等待应用加载")
                        await asyncio.sleep(1.0)
                        continue
                    
                    step_start = time.time()
                    if file_logger:
                        file_logger.debug("检测到Android桌面，启动应用")
                    
                    await self.adb.start_app(device_id, package_name, activity_name)
                    
                    # 等待应用启动（简单等待1秒）
                    await asyncio.sleep(1.0)
                    
                    if file_logger:
                        file_logger.debug(f"应用启动完成 (耗时: {int(time.time() - step_start)}秒)")
                    continue
                
                # 处理启动页服务弹窗
                if result.state == PageState.STARTUP_POPUP:
                    step_start = time.time()
                    
                    # 简洁日志：关闭服务弹窗
                    concise.action("关闭服务弹窗")
                    
                    # [2026-03-02] 修复检测器混用：直接使用固定坐标
                    await self.adb.tap(device_id, 270, 600)
                    
                    if file_logger:
                        file_logger.debug(f"点击'同意'按钮（固定坐标）")
                    
                    # 等待页面变化
                    await asyncio.sleep(1.0)
                    
                    step_time = time.time() - step_start
                    if file_logger:
                        file_logger.debug(f"弹窗处理完成 - 总耗时: {int(step_time)}秒")
                    continue
                
                # 处理首页公告弹窗
                if result.state == PageState.HOME_NOTICE:
                    step_start = time.time()
                    
                    # 简洁日志：关闭首页广告
                    concise.action("关闭首页广告")
                    
                    # [2026-03-02] 修复检测器混用：直接使用固定坐标
                    await self.adb.tap(device_id, 290, 210)  # X按钮位置
                    await asyncio.sleep(1.0)
                    
                    step_time = time.time() - step_start
                    if file_logger:
                        file_logger.debug(f"弹窗处理完成 - 总耗时: {int(step_time)}秒")
                    continue
                
                # 处理首页异常代码弹窗
                if result.state == PageState.HOME_ERROR_POPUP:
                    if file_logger:
                        file_logger.debug("检测到首页异常代码弹窗，点击'确认'按钮")
                    
                    # [2026-03-02] 修复检测器混用：直接使用固定坐标
                    await self.adb.tap(device_id, 265, 637)
                    
                    # 等待页面变化
                    await asyncio.sleep(1.0)
                    continue
                
                # [2026-03-02] 简化：处理广告页（等待广告自动消失）
                # 移除OCR验证逻辑，直接信任模型识别结果
                if result.state == PageState.AD:
                    step_start = time.time()
                    concise.action("等待广告")
                    
                    if file_logger:
                        file_logger.info("[广告页处理] 检测到广告页，等待广告消失...")
                    
                    # 等待广告消失（最多15秒）
                    ad_wait_start = asyncio.get_event_loop().time()
                    page_changed = False
                    
                    while asyncio.get_event_loop().time() - ad_wait_start < 15.0:
                        ad_result = await detector.detect_page(device_id, use_cache=False)
                        
                        if ad_result.state != PageState.AD:
                            page_changed = True
                            if file_logger:
                                file_logger.info(f"[广告等待] ✓ 广告已消失 - 当前页面: {ad_result.state.value}")
                            break
                        
                        await asyncio.sleep(0.5)
                    
                    # 清除缓存
                    if hasattr(detector, 'clear_cache'):
                        detector.clear_cache(device_id)
                    
                    step_time = time.time() - step_start
                    
                    if file_logger:
                        if page_changed:
                            file_logger.info(f"[广告等待] 广告处理完成 - 耗时: {int(step_time)}秒")
                        else:
                            file_logger.warning(f"[广告等待] 广告等待超时 - 耗时: {int(step_time)}秒")
                    continue
                
                # 处理加载页
                if result.state == PageState.LOADING:
                    # 等待加载完成
                    await asyncio.sleep(0.5)
                    continue
                
                # 处理未知页面
                if result.state == PageState.UNKNOWN:
                    if file_logger:
                        file_logger.debug("检测到未知页面，等待")
                    await asyncio.sleep(0.5)
                    continue
                
                # 其他状态，短暂等待
                await asyncio.sleep(0.2)
            
            # 超时，检查最终状态
            if file_logger:
                file_logger.warning("启动流程超时，检查最终状态")
            
            # [2026-03-01] 修复原因：PageDetectorDL不支持detect_elements参数
            final_result = await detector.detect_page(device_id, use_cache=False)
            
            if file_logger:
                file_logger.info(f"最终状态: {final_result.state.value}")
            
            if final_result.state == PageState.HOME:
                concise.success("到达首页")
                if file_logger:
                    file_logger.info("启动流程完成（超时后检测）")
                
                # [2026-03-11] 优化日志：移除控制台DEBUG输出，仅保留文件日志
                if file_logger:
                    file_logger.info(f"[DEBUG-启动完成] 启动流程完成（超时后检测），即将返回 True")
                
                  # 断点：启动流程完成（超时后检测）
                
                return True
            
            # 如果不是最后一次重试，继续
            if retry < max_retries - 1:
                # [2026-03-11] 修复原因：如果应用已由外部启动，不要重试，直接返回失败
                if app_already_started:
                    if file_logger:
                        file_logger.error("应用已由外部启动，启动流程超时，不进行重试")
                    return False
                
                if file_logger:
                    file_logger.warning("启动流程失败，准备重试")
                continue
        
        # 所有重试都失败
        if file_logger:
            file_logger.error("启动流程失败，已达到最大重试次数")
        return False
    async def _navigate_to_profile_with_ad_handling(self, device_id: str, log_callback=None) -> bool:
        """导航到个人页并处理广告（优化版）
        
        # [2026-03-17] 优化原因：提高只登录模式的导航速度
        核心逻辑：
        1. 优先使用固定坐标点击"我的"按钮（避免YOLO检测开销）
        2. 降低扫描频率（每200毫秒）减少页面检测调用
        3. 减少总扫描时间（3秒）提高响应速度
        4. 添加早期退出条件
        
        Args:
            device_id: 设备ID
            log_callback: 日志回调函数
            
        Returns:
            bool: 是否成功到达个人页
        """
        log = log_callback if log_callback else self._silent_log.info
        
        # [2026-03-17] 优化：优先使用固定坐标，避免YOLO检测开销
        log(f"  点击'我的'按钮（使用固定坐标）...")
        await self.adb.tap(device_id, 480, 920)  # 使用Navigator的标准坐标
        
        # 点击"我的"按钮后，等待1秒，然后按返回键（预防性关闭广告）
        log(f"  等待1秒后按返回键（预防性关闭广告）...")
        await asyncio.sleep(1.0)
        await self.adb.press_back(device_id)
        
        # 清除缓存
        if hasattr(self.detector, 'clear_cache'):
            self.detector.clear_cache(device_id)
        if hasattr(self.profile_detector, 'clear_cache'):
            self.profile_detector.clear_cache(device_id)
        
        # [2026-03-17] 优化：降低扫描频率和总时间，提高性能
        max_scan_time = 3.0  # 从5秒减少到3秒
        scan_interval = 0.2  # 从50毫秒增加到200毫秒，减少80%的检测调用
        start_time = asyncio.get_event_loop().time()
        
        ad_closed_count = 1  # 已经按了一次返回键（预防性关闭）
        consecutive_success_count = 0  # 连续成功检测计数
        
        while (asyncio.get_event_loop().time() - start_time) < max_scan_time:
            # [2026-03-02] 修复原因：使用资料专用模型检测页面
            # 检测当前页面状态
            page_result = await self.profile_detector.detect_page(
                device_id, use_cache=False
            )
            
            if not page_result or not page_result.state:
                await asyncio.sleep(scan_interval)
                continue
            
            from .page_detector import PageState
            current_state = page_result.state
            
            # 检测到正常个人页 → 连续检测确认后返回成功
            if current_state in [PageState.PROFILE, PageState.PROFILE_LOGGED]:
                consecutive_success_count += 1
                
                # [2026-03-17] 优化：连续2次检测到个人页就确认成功，避免过度等待
                if consecutive_success_count >= 2:
                    elapsed = asyncio.get_event_loop().time() - start_time
                    log(f"  ✓ 到达个人页（耗时: {elapsed:.1f}秒，关闭广告: {ad_closed_count}次）")
                    return True
                
                # 第一次检测到，等待确认
                await asyncio.sleep(scan_interval)
                continue
            else:
                # 重置连续成功计数
                consecutive_success_count = 0
            
            # 检测到广告 → 立即关闭
            if current_state == PageState.PROFILE_AD:
                log(f"  ⚠️ 检测到个人页广告，立即关闭...")
                
                # [2026-03-17] 优化：直接使用返回键关闭，避免YOLO检测开销
                await self.adb.press_back(device_id)
                ad_closed_count += 1
                
                # 等待0.3秒让广告关闭动画完成
                await asyncio.sleep(0.3)
                
                # 清除缓存
                if hasattr(self.detector, 'clear_cache'):
                    self.detector.clear_cache(device_id)
                if hasattr(self.profile_detector, 'clear_cache'):
                    self.profile_detector.clear_cache(device_id)
                
                # 继续扫描（可能还有广告，或者已经到达个人页）
                continue
            
            # 其他状态 → 继续扫描
            await asyncio.sleep(scan_interval)
        
        # 超时
        elapsed = asyncio.get_event_loop().time() - start_time
        log(f"  ❌ 导航到个人页超时（耗时: {elapsed:.1f}秒，关闭广告: {ad_closed_count}次）")
        return False
    
    async def run_full_workflow(self, device_id: str, account: Account, skip_login: bool = False, workflow_config: dict = None) -> AccountResult:
        """执行完整工作流：登录->获取初始数据->签到->获取最终数据->退出
        
        增强版数据收集流程：
        1. 登录账号（如果 skip_login=True 则跳过）
        2. 获取完整个人资料（昵称、ID、手机号、余额、积分、抵扣券）
        3. 获取签到信息并执行所有签到
        4. 再次获取余额（最终余额）
        5. 退出登录
        
        Args:
            device_id: 设备 ID
            account: 账号信息
            skip_login: 是否跳过登录步骤（缓存登录已验证时使用）
            workflow_config: 流程控制配置字典，包含：
                - enable_login: 是否执行登录流程
                - enable_profile: 是否获取个人资料
                - enable_checkin: 是否执行签到
                - enable_transfer: 是否执行转账
            
        Returns:
            账号处理结果（包含完整数据）
        """
        import time
        from .concise_logger import ConciseLogger
        import logging
        from .page_detector import PageState  # 确保 PageState 在函数开头导入
        
        # [2026-03-10] 修复原因：初始化 file_logger，避免未定义错误
        file_logger = logging.getLogger(__name__)
        
        # [2026-03-10] 修复原因：检查是否有缓存验证时获取的资料数据
        cached_profile_info = getattr(self, '_cached_profile_info', None)
        if cached_profile_info:
            file_logger.info("="*60)
            file_logger.info("检测到缓存验证时已获取的资料数据")
            file_logger.info(f"缓存资料: {cached_profile_info}")
            file_logger.info("="*60)
        
        # 默认流程配置（全部启用）
        if workflow_config is None:
            workflow_config = {
                'enable_login': True,
                'enable_profile': True,
                'enable_checkin': True,
                'enable_transfer': True,
            }
        
        # 记录工作流开始时间
        workflow_start = time.time()
        
        # 定义日志输出函数（优先使用回调，否则使用print）
        def log(msg):
            if self._log_callback:
                self._log_callback(msg)
            else:
                print(msg)
        
        # 获取文件日志记录器
        file_logger = logging.getLogger(__name__)
        
        # 获取账号日志管理器
        from .account_logger import get_account_logger
        account_logger = get_account_logger()
        
        # 记录账号处理开始
        account_logger.log_account_start(account.phone)
        
        # 创建简洁日志记录器（GUI日志 + 文件日志 + 账号独立日志）
        # 使用一个简单的包装类来避免参数传递问题
        if self._log_callback:
            class GuiLogger:
                def __init__(self, callback):
                    self.callback = callback
                def info(self, msg):
                    self.callback(msg)
                def error(self, msg):
                    self.callback(msg)
            gui_logger_obj = GuiLogger(self._log_callback)
        else:
            gui_logger_obj = None
        
        concise = ConciseLogger(
            module_name="workflow",
            gui_logger=gui_logger_obj,
            file_logger=file_logger,
            account_logger=account_logger,
            phone=account.phone
        )
        
        # 输出工作流开始信息（仅文件日志）
        file_logger.info(f"工作流开始 - {time.strftime('%H:%M:%S')}")
        file_logger.info(f"账号: {account.phone}")
        
        # [2026-03-11] 优化日志：移除DEBUG配置日志，减少冗余输出
        
        # 定义可中断的 sleep 函数
        async def interruptible_sleep(seconds: float, check_interval: float = 0.1):
            """可中断的 sleep，每隔 check_interval 检查一次停止标志"""
            elapsed = 0.0
            while elapsed < seconds:
                if self._stop_check and self._stop_check():
                    raise Exception("用户中断操作")
                await asyncio.sleep(min(check_interval, seconds - elapsed))
                elapsed += check_interval
        
        result = AccountResult(
            phone=account.phone,
            success=False,
            timestamp=datetime.now()
        )
        
        try:
            # ==================== 步骤1: 登录账号 ====================
            step_number = 1
            step1_start = time.time()
            
            # 添加简洁日志：步骤1 - 登录账号
            concise.step(step_number, "登录账号")
            
            # 文件日志记录详细信息
            file_logger.info(f"步骤{step_number}: 登录账号 - {account.phone}")
            
            # 如果 skip_login=True，说明缓存登录已验证，当前已在个人页
            if skip_login:
                concise.action("使用缓存登录")
                file_logger.info("缓存登录已验证，当前已在个人页，直接获取个人信息")
                step1_time = time.time() - step1_start
                file_logger.info(f"步骤1完成 - 耗时: {int(step1_time)}秒（跳过登录）")
                concise.success("登录成功（缓存）")
                # 缓存登录不需要处理登录和积分页，直接跳到获取个人资料
            else:
                # 执行正常登录流程
                concise.action("输入账号密码")
                login_start = time.time()
                login_result = await self.auto_login.login(
                    device_id, account.phone, account.password
                )
                login_time = time.time() - login_start
                file_logger.info(f"登录操作耗时: {int(login_time)}秒")
                
                if not login_result.success:
                    # 根据登录失败类型设置error_type
                    from .models.error_types import ErrorType
                    if login_result.error_type == "phone_not_exist":
                        result.error_type = ErrorType.LOGIN_PHONE_NOT_EXIST
                    elif login_result.error_type == "wrong_password":
                        result.error_type = ErrorType.LOGIN_PASSWORD_ERROR
                    else:
                        # 其他登录错误，暂时归类为密码错误
                        result.error_type = ErrorType.LOGIN_PASSWORD_ERROR
                    
                    result.error_message = f"登录失败: {login_result.error_message}"
                    concise.error(f"登录失败: {login_result.error_message}")
                    file_logger.error(f"登录失败: {login_result.error_message}")
                    return result
                
                concise.success("登录成功")
                file_logger.info("登录成功")
                # 登录后会跳转到积分页，需要返回到个人页
                file_logger.info("登录后处理积分页跳转...")
                # [2026-03-02] 优化：减少等待时间 2秒→1秒
                await asyncio.sleep(1)
                
                # [2026-03-06] 修复原因：资料专用模型不包含积分页，使用登录专用模型检测
                # 检测当前页面
                page_result = await self.login_detector.detect_page(
                    device_id, use_cache=False
                )
                
                if page_result and page_result.state == PageState.POINTS_PAGE:
                    log(f"检测到积分页，需要按2次返回键到个人页...")
                    
                    # 第1次返回键
                    await self.adb.press_back(device_id)
                    await asyncio.sleep(1)
                    
                    # 第2次返回键
                    await self.adb.press_back(device_id)
                    await asyncio.sleep(1)  # 等待页面加载
                    
                    # 清理缓存（页面已改变）
                    self.detector.clear_cache()
                    
                    # ===== 关键修复：使用统一的广告处理方法 =====
                    # 按返回键后可能到达个人页，也可能遇到个人页广告
                    # 使用高频扫描方法处理广告并确认到达个人页
                    max_scan_time = 5.0
                    scan_interval = 0.05  # 每50毫秒扫描一次
                    scan_start = asyncio.get_event_loop().time()
                    ad_closed_count = 0
                    
                    while (asyncio.get_event_loop().time() - scan_start) < max_scan_time:
                        # [2026-03-06] 修复原因：检测积分页用登录模型，检测个人页用资料模型
                        # 先用登录模型检测是否还在积分页
                        page_result_login = await self.login_detector.detect_page(
                            device_id, use_cache=False
                        )
                        
                        # 如果仍在积分页，再按一次返回键
                        if page_result_login and page_result_login.state == PageState.POINTS_PAGE:
                            log(f"⚠️ 仍在积分页，再按一次返回键...")
                            await self.adb.press_back(device_id)
                            await asyncio.sleep(1)
                            self.detector.clear_cache()
                            continue
                        
                        # 用资料模型检测个人页状态
                        page_result = await self.profile_detector.detect_page(
                            device_id, use_cache=False
                        )
                        
                        if not page_result or not page_result.state:
                            await asyncio.sleep(scan_interval)
                            continue
                        
                        current_state = page_result.state
                        
                        # 检测到正常个人页 → 成功
                        if current_state == PageState.PROFILE_LOGGED:
                            log(f"✓ 已返回到个人页（关闭广告: {ad_closed_count}次）")
                            break
                        
                        # 检测到个人页广告 → 立即关闭
                        elif current_state == PageState.PROFILE_AD:
                            log(f"⚠️ 检测到个人页广告，立即关闭...")
                            
                            # 使用返回键关闭广告（最可靠）
                            await self.adb.press_back(device_id)
                            ad_closed_count += 1
                            
                            # 等待0.3秒让广告关闭动画完成
                            await asyncio.sleep(0.3)
                            
                            # 清除缓存
                            self.detector.clear_cache(device_id)
                            
                            # 继续扫描
                            continue
                        
                        # 其他状态 → 继续扫描
                        else:
                            await asyncio.sleep(scan_interval)
                    else:
                        # 超时，检查最终状态
                        log(f"⚠️ 等待个人页超时，检查最终状态...")
                        # [2026-03-02] 修复原因：使用资料专用模型检测页面
                        final_result = await self.profile_detector.detect_page(
                            device_id, use_cache=False
                        )
                        if final_result:
                            log(f"最终状态: {final_result.state.value}")
                else:
                    log(f"当前页面: {page_result.state.value if page_result else 'unknown'}")
                    # [2026-03-02] 优化：减少等待时间 2秒→1秒
                    await asyncio.sleep(1)
            
            # ==================== 步骤2: 获取初始个人资料 ====================
            # 快速签到模式逻辑：
            # - 如果账号有登录缓存 → 跳过获取资料（真正的快速模式）
            # - 如果账号没有登录缓存 → 执行完整流程（获取资料并保存缓存）
            enable_profile = workflow_config.get('enable_profile', True)
            has_login_cache = False
            
            # 检查是否有登录缓存（检查 login_cache 目录）
            if not enable_profile and self.auto_login.enable_cache and self.auto_login.cache_manager:
                # 检查登录缓存目录是否存在
                # 需要检查两种格式：
                # 1. 手机号（旧格式）
                # 2. 手机号_用户ID（新格式）
                cache_base_dir = self.auto_login.cache_manager.cache_dir
                has_login_cache = False
                
                # 检查旧格式：手机号
                old_format_dir = cache_base_dir / account.phone
                if old_format_dir.exists() and any(old_format_dir.iterdir()):
                    has_login_cache = True
                    file_logger.debug(f"找到旧格式缓存: {old_format_dir}")
                
                # 检查新格式：手机号_*
                if not has_login_cache and cache_base_dir.exists():
                    for cache_dir in cache_base_dir.iterdir():
                        if cache_dir.is_dir() and cache_dir.name.startswith(f"{account.phone}_"):
                            if any(cache_dir.iterdir()):
                                has_login_cache = True
                                file_logger.debug(f"找到新格式缓存: {cache_dir}")
                                break
                
                if has_login_cache:
                    # [2026-03-11] 优化日志：简化快速签到模式的日志输出
                    file_logger.info("快速签到模式：检测到有登录缓存，跳过获取资料")
                    concise.action("快速模式：有缓存，跳过获取资料")
                else:
                    # [2026-03-11] 优化日志：简化快速签到模式的日志输出
                    # [2026-03-19] 修复原因：快速签到模式检测到无缓存时，必须切换为完整流程并获取资料
                    file_logger.info("快速签到模式：检测到无登录缓存，自动切换为完整流程")
                    concise.action("快速模式：无缓存，切换为完整流程")
                    # 自动切换为完整流程，确保会获取资料并保存缓存
                    enable_profile = True
                    file_logger.info("已启用资料获取，登录后将保存缓存")
            
            # 根据最终的 enable_profile 决定是否获取资料
            if not enable_profile:
                # 有缓存，跳过获取资料
                profile_success = False
                profile_data = None
                file_logger.info("跳过获取个人资料")
            else:
                step_number += 1
                
                # 添加简洁日志：步骤2 - 获取资料
                concise.step(step_number, "获取资料")
                
                # 文件日志记录详细信息
                file_logger.info(f"步骤{step_number}: 获取初始个人资料")
                
                # [2026-03-10] 修复原因：优先使用缓存验证时获取的资料数据
                if cached_profile_info:
                    file_logger.info("使用缓存验证时已获取的资料数据")
                    profile_success = True
                    profile_data = cached_profile_info
                    
                    # 清除缓存数据，避免重复使用
                    self._cached_profile_info = None
                    
                    # 显示使用缓存数据的日志
                    concise.success("使用缓存资料")
                    file_logger.info(f"缓存资料数据: {profile_data}")
                else:
                    # 正常获取资料流程
                    profile_success = False
                    profile_data = None
                
                # 尝试最多3次获取个人资料
                for attempt in range(3):
                    try:
                        if attempt > 0:
                            file_logger.info(f"尝试 {attempt + 1}/3 重新获取个人资料...")
                        
                        # 如果是缓存登录，跳过导航（已经在个人页）
                        if skip_login:
                            cache_check_start = time.time()
                            concise.action("验证当前页面")
                            file_logger.info("缓存登录 - 验证当前页面状态...")
                            nav_success = True
                            
                            # 立即检测页面状态（不等待）- 使用YOLO（GPU加速）
                            detect_start = time.time()
                            from .page_detector import PageState
                            # [2026-03-02] 修复原因：使用资料专用模型检测页面
                            page_result = await self.profile_detector.detect_page(
                                device_id, use_cache=True
                            )
                            detect_time = time.time() - detect_start
                            file_logger.info(f"页面检测耗时: {int(detect_time)}秒")
                            
                            if not page_result or not page_result.state:
                                file_logger.warning("无法检测当前页面状态")
                                if attempt < 2:
                                    # [2026-03-02] 优化：减少重试等待 2秒→1秒
                                    await asyncio.sleep(1)
                                    continue
                                else:
                                    result.error_message = "无法确认当前页面状态"
                                    return result
                            
                            file_logger.info(f"当前页面: {page_result.state.value}（置信度{page_result.confidence:.2%}）")
                            
                            # 确认在个人页（已登录）
                            if page_result.state != PageState.PROFILE_LOGGED:
                                file_logger.warning(f"当前不在个人页（已登录），尝试导航...")
                                nav_start = time.time()
                                # 使用统一的广告处理方法
                                nav_success = await self._navigate_to_profile_with_ad_handling(device_id, log)
                                nav_time = time.time() - nav_start
                                file_logger.info(f"导航耗时: {int(nav_time)}秒")
                                if not nav_success:
                                    if attempt < 2:
                                        # [2026-03-02] 优化：减少重试等待 2秒→1秒
                                        await asyncio.sleep(1)
                                        continue
                                    else:
                                        result.error_message = "无法导航到个人页"
                                        return result
                            else:
                                file_logger.info("确认在个人页（已登录）")
                                file_logger.info("页面已就绪，直接获取个人资料")
                            
                            cache_check_time = time.time() - cache_check_start
                            file_logger.info(f"缓存登录验证总耗时: {int(cache_check_time)}秒")
                        else:
                            # 导航到个人资料页面（使用统一的广告处理方法）
                            # [2026-03-13] 修复原因：登录成功后可能已经在个人页，先检测再决定是否导航
                            # 检测当前页面状态
                            page_result = await self.profile_detector.detect_page(device_id, use_cache=False)
                            
                            if page_result and page_result.state == PageState.PROFILE_LOGGED:
                                # 已经在个人页，跳过导航
                                file_logger.info("当前已在个人页（已登录），跳过导航")
                                nav_success = True
                            else:
                                # 不在个人页，需要导航
                                concise.action("进入个人页")
                                
                                # [2026-03-05] 添加调试日志：导航前的状态
                                file_logger.info("[DEBUG-导航] 准备导航到个人页...")
                                file_logger.info(f"[DEBUG-导航] 设备ID: {device_id}")
                                file_logger.info(f"[DEBUG-导航] 尝试次数: {attempt + 1}/3")
                                
                                nav_start = time.time()
                                nav_success = await self._navigate_to_profile_with_ad_handling(device_id, log)
                                nav_time = time.time() - nav_start
                                
                                # [2026-03-05] 添加调试日志：导航结果
                                file_logger.info(f"[DEBUG-导航] 导航结果: {'成功' if nav_success else '失败'}")
                                file_logger.info(f"[DEBUG-导航] 导航耗时: {int(nav_time)}秒")
                            
                            if not nav_success:
                                log(f"⚠️ 导航到个人资料页面失败")
                                
                                # [2026-03-02] 修复原因：使用资料专用模型检测页面
                                # 检查当前页面状态
                                from .page_detector import PageState
                                page_result = await self.profile_detector.detect_page(
                                    device_id, use_cache=True
                                )
                                
                                # 检查返回值是否有效
                                if not page_result or not page_result.state:
                                    log(f"  ⚠️ 无法检测当前页面状态")
                                    if attempt < 2:
                                        # [2026-03-02] 优化：减少重试等待 2秒→1秒
                                        await asyncio.sleep(1)
                                        continue
                                    else:
                                        break
                                
                                log(f"  当前页面: {page_result.state.value}")
                                
                                # 如果已经在个人资料页面，继续
                                if page_result.state in [PageState.PROFILE, PageState.PROFILE_LOGGED]:
                                    log(f"  ✓ 检测到已在个人资料页面，继续获取数据")
                                else:
                                    # 尝试按返回键回到首页，然后重新导航
                                    log(f"  尝试返回首页...")
                                    await self.adb.press_back(device_id)
                                    # [2026-03-02] 优化：减少等待时间 2秒→1秒
                                    await asyncio.sleep(1)
                                    
                                    # 重新导航
                                    nav_success = await self.navigator.navigate_to_profile(device_id)
                                    if not nav_success:
                                        if attempt < 2:
                                            log(f"  ⚠️ 导航仍然失败，等待1秒后重试...")
                                            # [2026-03-02] 优化：减少重试等待 2秒→1秒
                                            await asyncio.sleep(1)
                                            continue
                                        else:
                                            from .models.error_types import ErrorType
                                            result.error_type = ErrorType.CANNOT_REACH_PROFILE
                                            result.error_message = "无法导航到个人资料页面"
                                            log(f"✗ 无法获取个人资料，终止流程\n")
                                            return result
                        
                        # ===== 关键优化：直接尝试获取资料，通过获取结果判断是否成功 =====
                        # 使用 ProfileReader 获取完整个人资料（带重试）
                        # 传递账号字符串（格式：手机号----密码），用于提取手机号
                        
                        # [2026-03-12] 优化日志：移除获取资料的详细技术日志
                        # [2026-03-12] 优化日志：移除获取资料的详细技术日志
                        
                        # [2026-03-12] 优化日志：移除获取资料的详细技术日志
                        account_str = f"{account.phone}----{account.password}"
                        
                        # [2026-03-13] 修复：添加计时起点
                        profile_read_start = time.time()
                        
                        try:
                            profile_data = await self.profile_reader.get_full_profile_with_retry(
                                device_id, 
                                account=account_str,
                                gui_logger=gui_logger_obj,
                                step_number=step_number
                            )
                        except Exception as profile_error:
                            # [2026-03-05] 添加调试日志：捕获获取资料的异常
                            file_logger.error(f"[DEBUG-获取资料] 获取资料时发生异常: {profile_error}")
                            import traceback
                            file_logger.error(f"[DEBUG-获取资料] 异常堆栈:\n{traceback.format_exc()}")
                            raise  # 重新抛出异常，让外层处理
                        
                        profile_read_time = time.time() - profile_read_start
                        
                        # [2026-03-05] 添加调试日志：获取资料结果
                        file_logger.info(f"[DEBUG-获取资料] 获取资料耗时: {int(profile_read_time)}秒")
                        file_logger.info(f"[DEBUG-获取资料] 获取到的数据: {profile_data}")
                        
                        file_logger.info(f"获取个人资料耗时: {int(profile_read_time)}秒")
                        
                        # 检查是否成功获取数据（必须获取到余额、昵称和用户ID）
                        has_balance = profile_data and profile_data.get('balance') is not None
                        has_nickname = profile_data and profile_data.get('nickname') is not None
                        has_user_id = profile_data and profile_data.get('user_id') is not None
                        instance_closed = profile_data and profile_data.get('instance_closed') is True
                        
                        # [2026-03-14] 修复原因：检测到实例关闭，重启实例并重试
                        if instance_closed:
                            file_logger.warning("检测到实例已关闭，尝试重启实例...")
                            log("⚠️ 实例已关闭，正在重启...")
                            
                            # 从device_id提取实例ID
                            try:
                                port = int(device_id.split(':')[1])
                                instance_id = (port - 16384) // 32
                            except:
                                instance_id = 0
                            
                            # 重启实例
                            from .emulator_controller import EmulatorController
                            controller = EmulatorController()
                            restart_success = await controller.launch_instance(instance_id, timeout=60)
                            
                            if restart_success:
                                file_logger.info(f"实例 {instance_id} 重启成功")
                                log(f"✓ 实例 {instance_id} 重启成功，继续执行...")
                                
                                # 等待实例完全启动
                                await asyncio.sleep(3)
                                
                                # 重新启动应用
                                await self.adb.start_app(device_id, "com.ry.xmsc")
                                await asyncio.sleep(2)
                                
                                # 继续重试
                                if attempt < 2:
                                    log(f"  等待1秒后重试...")
                                    await asyncio.sleep(1)
                                continue
                            else:
                                file_logger.error(f"实例 {instance_id} 重启失败")
                                log(f"❌ 实例 {instance_id} 重启失败")
                                break
                        
                        # ===== 核心逻辑：基于获取资料的结果判断是否成功 =====
                        if has_balance and has_nickname and has_user_id:
                            file_logger.info("成功获取个人资料数据")
                            profile_success = True
                            
                            # 简洁日志已在 profile_reader.read_profile 中输出
                            # 这里不需要重复输出
                            
                            break  # 获取成功，退出循环
                        else:
                            missing = []
                            if not has_balance:
                                missing.append("余额")
                            if not has_nickname:
                                missing.append("昵称")
                            if not has_user_id:
                                missing.append("用户ID")
                            log(f"⚠️ 获取的数据不完整，缺少: {', '.join(missing)}")
                            
                            # 数据不完整，说明可能不在个人页或页面加载未完成
                            # 重新导航并重试
                            if attempt < 2:
                                log(f"  重新导航到个人页...")
                                await self.navigator.navigate_to_profile(device_id)
                                log(f"  等待1秒后重试...")
                                # [2026-03-02] 优化：减少重试等待 2秒→1秒
                                await asyncio.sleep(1)
                        
                    except Exception as e:
                        # [2026-03-05] 添加调试日志：捕获异常的详细信息
                        import traceback
                        error_trace = traceback.format_exc()
                        file_logger.error(f"[DEBUG-异常] 获取个人资料出错: {str(e)}")
                        file_logger.error(f"[DEBUG-异常] 异常类型: {type(e).__name__}")
                        file_logger.error(f"[DEBUG-异常] 异常堆栈:\n{error_trace}")
                        
                        log(f"⚠️ 获取个人资料出错: {str(e)}")
                        if attempt < 2:
                            log(f"  等待1秒后重试...")
                            # [2026-03-02] 优化：减少重试等待 2秒→1秒
                            await asyncio.sleep(1)
                        else:
                            from .models.error_types import ErrorType
                            result.error_type = ErrorType.CANNOT_READ_PROFILE
                            result.error_message = f"获取个人资料失败: {str(e)}"
                            log(f"✗ 无法获取个人资料，终止流程\n")
                            
                            # [2026-03-05] 添加调试日志：最终失败信息
                            file_logger.error(f"[DEBUG-最终失败] 3次尝试后仍然失败")
                            file_logger.error(f"[DEBUG-最终失败] 错误类型: {result.error_type}")
                            file_logger.error(f"[DEBUG-最终失败] 错误消息: {result.error_message}")
                            
                            return result
            
            # 如果3次尝试后仍未成功（但快速签到模式除外）
            # 快速签到模式下，允许跳过获取资料，在签到时获取
            if not enable_profile:
                # 快速签到模式：跳过资料检查，继续执行
                file_logger.info("快速签到模式：跳过资料检查，将在签到时获取资料")
            elif not profile_success or not profile_data:
                # 非快速签到模式：必须成功获取资料
                from .models.error_types import ErrorType
                result.error_type = ErrorType.CANNOT_READ_PROFILE
                result.error_message = "获取个人资料失败（3次尝试后）"
                log(f"✗ 无法获取个人资料，终止流程\n")
                return result
            
            # ==================== 步骤2后处理：保存数据和缓存 ====================
            # 只有在成功获取资料后才执行
            if profile_success and profile_data:
                # ==================== 存储初始数据到 result ====================
                result.nickname = profile_data.get('nickname')  # 直接使用OCR结果，不使用历史数据
                result.user_id = profile_data.get('user_id')  # 直接使用OCR结果，不使用历史数据
                # phone 已经在初始化时设置
                result.balance_before = profile_data.get('balance')
                result.points = profile_data.get('points')
                result.vouchers = profile_data.get('vouchers')
                # [2026-03-01] 删除优惠券：个人页已经没有优惠券了
                
                # [2026-03-11] 优化日志：简化账户信息输出
                file_logger.info("账户信息")
                file_logger.info(f"昵称: {result.nickname or 'N/A'}")
                file_logger.info(f"ID: {result.user_id or 'N/A'}")
                file_logger.info(f"手机号: {result.phone}")
                if result.balance_before is not None:
                    file_logger.info(f"余额: {result.balance_before:.2f} 元")
                if result.points is not None:
                    file_logger.info(f"积分: {result.points} 积分")
                if result.vouchers is not None:
                    file_logger.info(f"抵扣券: {result.vouchers} 张")
                
                # [2026-03-11] 优化日志：移除调试信息，减少冗余输出
                
                # ==================== 步骤3.5: 保存登录缓存（包含用户ID）====================
                # 修复：同步保存缓存，确保一定会保存成功
                if self.auto_login.enable_cache and self.auto_login.cache_manager:
                    if result.user_id:
                        file_logger.info(f"保存登录缓存（包含用户ID: {result.user_id}）...")
                        concise.action("保存登录缓存")
                    else:
                        file_logger.info("保存登录缓存（未获取到用户ID）...")
                        concise.action("保存登录缓存")
                    
                    try:
                        # 同步等待缓存保存完成
                        cache_saved = await self.auto_login.cache_manager.save_login_cache(
                            device_id, 
                            account.phone, 
                            user_id=result.user_id
                        )
                        if cache_saved:
                            file_logger.info("✓ 登录缓存已保存")
                            concise.success("缓存已保存")
                        else:
                            file_logger.warning("✗ 登录缓存保存失败")
                            concise.error("缓存保存失败")
                    except Exception as e:
                        file_logger.error(f"✗ 保存登录缓存时出错: {str(e)}")
                        concise.error(f"缓存保存出错: {str(e)}")
                else:
                    if not self.auto_login.enable_cache:
                        file_logger.info("缓存功能未启用，跳过缓存保存")
                    elif not self.auto_login.cache_manager:
                        file_logger.warning("缓存管理器未初始化，跳过缓存保存")
                
                # 优化：移除不必要的1秒等待
                # await asyncio.sleep(1)  # 已移除
                
                # ==================== 步骤2.5: 验证用户ID（正常模式）====================
                # 验证当前用户ID是否与缓存中的ID匹配
                if result.user_id and self.auto_login.enable_cache and self.auto_login.cache_manager:
                    file_logger.info("="*60)
                    file_logger.info("验证用户ID")
                    file_logger.info("="*60)
                    
                    # 从缓存中获取预期的用户ID
                    expected_user_id = self.auto_login.cache_manager._get_expected_user_id(account.phone)
                    current_user_id = result.user_id
                    
                    if expected_user_id:
                        file_logger.info(f"当前用户ID: {current_user_id}")
                        file_logger.info(f"缓存用户ID: {expected_user_id}")
                        
                        if current_user_id != expected_user_id:
                            # ID不匹配，清理缓存并返回错误
                            file_logger.error(f"用户ID不匹配！")
                            file_logger.error(f"  当前: {current_user_id}")
                            file_logger.error(f"  缓存: {expected_user_id}")
                            
                            concise.error(f"ID不匹配: {current_user_id} ≠ {expected_user_id}")
                            log(f"❌ 用户ID验证失败: 当前({current_user_id}) ≠ 缓存({expected_user_id})")
                            
                            # 清理缓存
                            file_logger.info("清理登录缓存...")
                            await self.auto_login.cache_manager.clear_app_login_data(
                                device_id, 
                                "com.xmwl.shop"
                            )
                            
                            # 返回错误
                            from .models.error_types import ErrorType
                            result.error_type = ErrorType.CACHE_USER_ID_MISMATCH
                            result.error_message = f"用户ID不匹配（当前: {current_user_id}, 缓存: {expected_user_id}）"
                            result.success = False
                            return result
                        else:
                            file_logger.info("✓ 用户ID验证通过")
                            concise.success("ID验证通过")
                            log(f"✓ 用户ID验证通过: {current_user_id}")
                    else:
                        file_logger.info("缓存中无用户ID记录，跳过验证")
                        concise.action("无缓存ID，跳过验证")
                    
                    file_logger.info("="*60)
                
                # ==================== 优化：跳过重复获取个人资料 ====================
                # 步骤2已经获取了完整的个人资料，不需要在步骤4重复获取
                # 直接使用步骤2的数据，节省时间
                file_logger.info("="*60)
                file_logger.info("使用步骤2已获取的个人资料数据")
                file_logger.info(f"用户ID: {result.user_id}")
                file_logger.info(f"昵称: {result.nickname}")
                if result.balance_before:
                    file_logger.info(f"余额: {result.balance_before:.2f} 元")
                file_logger.info(f"积分: {result.points}")
                file_logger.info(f"抵扣券: {result.vouchers}")
                # [2026-03-01] 删除优惠券：个人页已经没有优惠券了
                file_logger.info("="*60)
            
            # ==================== 步骤3: 执行签到 ====================
            # 检查是否启用签到流程
            if not workflow_config.get('enable_checkin', True):
                file_logger.info("="*60)
                file_logger.info("跳过签到流程")
                file_logger.info("原因: 流程控制已禁用签到")
                file_logger.info("="*60)
                # 标记为成功但跳过签到
                result.success = True
                # 跳过签到，不增加step_number
            else:
                # 执行签到流程
                step_number += 1
                
                # [2026-03-01] 修复原因：删除重复的步骤日志，do_checkin内部会输出
                # 文件日志记录详细信息
                file_logger.info("="*60)
                file_logger.info(f"步骤{step_number}: 执行签到")
                file_logger.info("="*60)
                
                try:
                    # 准备个人资料数据
                    # 如果步骤2获取了资料，使用步骤2的数据
                    # 如果跳过了步骤2（快速签到模式），从数据库历史记录中获取
                    if profile_success and profile_data:
                        updated_profile_data = {
                            'balance': result.balance_before,  # 使用步骤2获取的余额
                            'points': result.points,
                            'vouchers': result.vouchers,
                            # [2026-03-01] 删除优惠券：个人页已经没有优惠券了
                            'nickname': result.nickname,
                            'user_id': result.user_id
                        }
                    else:
                        # 快速签到模式：从数据库历史记录中获取用户信息
                        file_logger.info("快速签到模式：从数据库历史记录中获取用户信息")
                        from .local_db import LocalDatabase
                        db = LocalDatabase()
                        
                        # [2026-03-03] 修复原因：获取最近的历史记录（排除今天）
                        # 如果没有昨天的记录，继续往前查找，直到找到最近的一条记录
                        today = datetime.now().strftime('%Y-%m-%d')
                        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                        
                        # 先尝试获取昨天或之前的记录（排除今天）
                        latest_record = db.get_history_records(phone=account.phone, end_date=yesterday, limit=1)
                        latest_record = latest_record[0] if latest_record else None
                        
                        # [2026-03-03] 修复原因：如果没有找到记录，尝试获取所有历史记录中的最新一条
                        if not latest_record:
                            file_logger.info("未找到昨天或之前的记录，尝试获取所有历史记录中的最新一条...")
                            all_records = db.get_history_records(phone=account.phone, limit=1)
                            latest_record = all_records[0] if all_records else None
                            
                            if latest_record:
                                file_logger.info(f"找到历史记录（日期: {latest_record.get('timestamp', '未知')}）")
                        
                        # [2026-03-03] 修复原因：快速签到模式不导航到个人页，余额从数据库获取或设为0
                        if latest_record:
                            # 从历史记录中提取信息
                            result.nickname = latest_record.get('nickname', '未知')
                            result.user_id = latest_record.get('user_id', '未知')
                            # 优先使用 balance_after，如果为 None 则使用 balance_before
                            balance_after = latest_record.get('balance_after')
                            balance_before = latest_record.get('balance_before')
                            if balance_after is not None:
                                result.balance_before = balance_after
                            elif balance_before is not None:
                                result.balance_before = balance_before
                            else:
                                result.balance_before = 0.0
                            result.points = latest_record.get('points')
                            result.vouchers = latest_record.get('vouchers')
                            
                            file_logger.info(f"从历史记录获取:")
                            file_logger.info(f"  - 昵称: {result.nickname}")
                            file_logger.info(f"  - 用户ID: {result.user_id}")
                            file_logger.info(f"  - 余额前: {result.balance_before:.2f} 元")
                            if result.points is not None:
                                file_logger.info(f"  - 积分: {result.points}")
                            if result.vouchers is not None:
                                file_logger.info(f"  - 抵扣券: {result.vouchers}")
                        else:
                            file_logger.warning("数据库中未找到历史记录，余额设置为 0.00 元")
                            result.balance_before = 0.0
                        
                        # 构建 profile_data 字典，传递给 do_checkin
                        # [2026-03-11] 修复原因：只要有余额记录就构建profile_data，不依赖nickname和user_id
                        if result.balance_before is not None:
                            updated_profile_data = {
                                'balance': result.balance_before,  # 从数据库或个人页获取的余额
                                'points': result.points,
                                'vouchers': result.vouchers,
                                # [2026-03-01] 删除优惠券：个人页已经没有优惠券了
                                'nickname': result.nickname,
                                'user_id': result.user_id
                            }
                        else:
                            file_logger.warning("未获取到任何用户信息，用户信息将在签到后获取")
                            updated_profile_data = None
                    
                    # 直接调用 do_checkin，它会自动处理导航和返回首页
                    # 传递登录回调，以便在缓存失效时可以直接登录
                    # [2026-03-29] 修复：签到页跳登录页时，用签到专用模型判断是否离开登录页
                    # 不能用auto_login.login()，因为登录专用模型没有登录页检测，会误判
                    checkin_page_classifier = self.daily_checkin.page_classifier
                    
                    async def login_callback_wrapper(dev_id, phone_num, pwd):
                        """登录回调包装器（签到页跳登录页专用，使用签到专用模型检测）"""
                        try:
                            # 直接输入账号密码（已在登录页，不需要导航）
                            await self.auto_login._tap_with_fallback(dev_id, self.auto_login.PHONE_INPUT, None)
                            await asyncio.sleep(0.3)
                            await self.auto_login._tap_with_fallback(dev_id, self.auto_login.PHONE_INPUT, None)
                            await asyncio.sleep(0.5)
                            # [2026-03-29] 修复：改用End键移到末尾再退格，避免光标在中间删不干净
                            await self.adb.key_event(dev_id, 123)  # End键
                            await asyncio.sleep(0.1)
                            for _ in range(30):
                                await self.adb.key_event(dev_id, 67)
                                await asyncio.sleep(0.02)
                            await self.adb.input_text(dev_id, phone_num)
                            await asyncio.sleep(1.0)
                            
                            await self.auto_login._tap_with_fallback(dev_id, self.auto_login.PASSWORD_INPUT, None)
                            await asyncio.sleep(0.3)
                            await self.auto_login._tap_with_fallback(dev_id, self.auto_login.PASSWORD_INPUT, None)
                            await asyncio.sleep(0.5)
                            # [2026-03-29] 修复：改用End键移到末尾再退格，避免光标在中间删不干净
                            await self.adb.key_event(dev_id, 123)  # End键
                            await asyncio.sleep(0.1)
                            for _ in range(30):
                                await self.adb.key_event(dev_id, 67)
                                await asyncio.sleep(0.02)
                            await self.adb.input_text(dev_id, pwd)
                            await asyncio.sleep(1.0)
                            
                            await self.auto_login._click_agreement_with_retry(dev_id, None, max_retries=3)
                            await asyncio.sleep(0.5)
                            await self.auto_login._tap_with_fallback(dev_id, self.auto_login.LOGIN_BUTTON, None)
                            await asyncio.sleep(1.5)
                            
                            # 用签到专用模型等待离开登录页（最多10秒）
                            for _ in range(20):
                                await asyncio.sleep(0.5)
                                page_r = await checkin_page_classifier.detect_page(dev_id, use_cache=False)
                                if page_r and page_r.state != PageState.LOGIN:
                                    return LoginResult(success=True)
                            return LoginResult(success=False, error_message="登录超时")
                        except Exception as e:
                            return LoginResult(success=False, error_message=str(e))
                    
                    # [2026-03-13] 修复原因：签到前必须导航到首页
                    # 签到流程需要在首页点击签到按钮
                    file_logger.info("签到前导航到首页...")
                    await self.navigator.navigate_to_home(device_id)
                    await asyncio.sleep(1)  # 等待页面稳定
                    
                    # [2026-03-03] 修复原因：快速签到模式始终跳过个人页导航
                    # 快速模式不需要导航到个人页，直接使用数据库余额
                    checkin_result = await self.daily_checkin.do_checkin(
                        device_id, 
                        phone=account.phone,
                        password=account.password,
                        login_callback=login_callback_wrapper,  # 传递登录回调，用于缓存失效时登录
                        log_callback=log,  # 传递日志回调，输出签到流程日志
                        profile_data=updated_profile_data,  # 传递个人信息（可能为None）
                        allow_skip_profile=True,  # 快速签到模式：始终跳过个人页导航
                        step_number=2,  # [2026-03-01] 修复：传递步骤编号，使获取资料显示为"步骤3"
                        gui_logger=gui_logger_obj  # 传递GUI日志记录器
                    )
                    
                    # 记录签到结果
                    if checkin_result.get('success'):
                        # 保存签到后余额（用于计算签到奖励）
                        checkin_balance_after = checkin_result.get('checkin_balance_after')
                        if checkin_balance_after is not None:
                            result.checkin_balance_after = checkin_balance_after
                        
                        # 保存总签到次数
                        result.checkin_total_times = checkin_result.get('total_times')
                        
                        # [2026-03-25] 修复缩进错误：添加条件判断
                        # 如果签到成功但未获取到总次数,从数据库历史记录中获取
                        if not result.checkin_total_times:
                            from .local_db import LocalDatabase
                            db = LocalDatabase()
                            latest_record = db.get_latest_record_by_phone(account.phone)
                            if latest_record and latest_record.get('checkin_total_times'):
                                result.checkin_total_times = latest_record.get('checkin_total_times')
                                file_logger.info(f"✓ 从数据库获取到总次数: {result.checkin_total_times}")
                            else:
                                file_logger.warning("⚠️ 数据库中也没有总次数记录")
                        
                        # 从签到结果中提取完整资料（签到流程已获取）
                        # 智能合并：只用有效值更新，保留原有值
                        
                        # 昵称：只有当签到结果的昵称有效且不是错误识别时才更新
                        checkin_nickname = checkin_result.get('nickname')
                        if checkin_nickname and checkin_nickname not in ['西', '1 0', '10', '1', '0']:
                            result.nickname = checkin_nickname
                        
                        # 用户ID：只有当签到结果的用户ID有效时才更新
                        checkin_user_id = checkin_result.get('user_id')
                        if checkin_user_id and checkin_user_id != '未知':
                            result.user_id = checkin_user_id
                        elif not profile_success:
                            # 快速签到模式：如果签到后没有获取到user_id，从数据库历史记录中获取
                            if not result.user_id:
                                file_logger.info("签到后未获取到用户ID，尝试从数据库历史记录中获取...")
                                from .local_db import LocalDatabase
                                db = LocalDatabase()
                                history_data = db.get_latest_account_data(account.phone)
                                if history_data and history_data.get('user_id'):
                                    result.user_id = history_data.get('user_id')
                                    file_logger.info(f"✓ 从数据库获取到用户ID: {result.user_id}")
                                else:
                                    file_logger.warning("⚠️ 数据库中也没有用户ID记录")
                        
                        # 其他字段：只要不是None就更新
                        if checkin_result.get('points') is not None:
                            result.points = checkin_result.get('points')
                        if checkin_result.get('vouchers') is not None:
                            result.vouchers = checkin_result.get('vouchers')
                        # [2026-03-01] 删除优惠券：个人页已经没有优惠券了
                        if checkin_result.get('balance_before') is not None:
                            result.balance_before = checkin_result.get('balance_before')
                        
                        # ==================== 快速签到模式：验证用户ID ====================
                        # 如果是快速签到模式，签到后获取了完整资料，需要验证ID
                        if not profile_success and result.user_id and self.auto_login.enable_cache and self.auto_login.cache_manager:
                            # [2026-03-11] 优化日志：简化快速签到模式的验证日志
                            file_logger.info("快速签到模式：验证用户ID")
                            
                            # 从缓存中获取预期的用户ID
                            expected_user_id = self.auto_login.cache_manager._get_expected_user_id(account.phone)
                            current_user_id = result.user_id
                            
                            if expected_user_id:
                                # [2026-02-27] 修复ID比对问题：标准化ID格式（转字符串、去空格）
                                current_user_id_normalized = str(current_user_id).strip()
                                expected_user_id_normalized = str(expected_user_id).strip()
                                
                                # 输出详细的调试信息
                                file_logger.info(f"当前用户ID（原始）: '{current_user_id}' (类型: {type(current_user_id).__name__})")
                                file_logger.info(f"当前用户ID（标准化）: '{current_user_id_normalized}'")
                                file_logger.info(f"缓存用户ID（原始）: '{expected_user_id}' (类型: {type(expected_user_id).__name__})")
                                file_logger.info(f"缓存用户ID（标准化）: '{expected_user_id_normalized}'")
                                
                                if current_user_id_normalized != expected_user_id_normalized:
                                    # ID不匹配，清理缓存并返回错误
                                    file_logger.error(f"用户ID不匹配！")
                                    file_logger.error(f"  当前: {current_user_id}")
                                    file_logger.error(f"  缓存: {expected_user_id}")
                                    
                                    concise.error(f"ID不匹配: {current_user_id} ≠ {expected_user_id}")
                                    log(f"❌ 用户ID验证失败: 当前({current_user_id}) ≠ 缓存({expected_user_id})")
                                    
                                    # 清理缓存
                                    file_logger.info("清理登录缓存...")
                                    await self.auto_login.cache_manager.clear_app_login_data(
                                        device_id, 
                                        "com.xmwl.shop"
                                    )
                                    
                                    # 返回错误
                                    from .models.error_types import ErrorType
                                    result.error_type = ErrorType.CACHE_USER_ID_MISMATCH
                                    result.error_message = f"用户ID不匹配（当前: {current_user_id}, 缓存: {expected_user_id}）"
                                    result.success = False
                                    return result
                                else:
                                    file_logger.info("✓ 用户ID验证通过")
                                    concise.success("ID验证通过")
                                    log(f"✓ 用户ID验证通过: {current_user_id}")
                            else:
                                file_logger.info("缓存中无用户ID记录，跳过验证")
                                concise.action("无缓存ID，跳过验证")
                            
                            # [2026-03-11] 优化日志：移除分隔线
                        
                        # 使用签到模块返回的奖励值（不重复计算）
                        result.checkin_reward = checkin_result.get('reward_amount', 0.0)
                        
                        # 记录日志
                        file_logger.info("签到完成")
                        concise.success("签到完成")
                        
                        # [2026-03-14] 修复原因：验证签到次数识别是否正确
                        # 如果签到次数=0但有余额，从数据库获取历史记录验证
                        if result.checkin_total_times == 0 and result.checkin_balance_after and result.checkin_balance_after > 0:
                            from .local_db import LocalDatabase
                            db = LocalDatabase()
                            latest_record = db.get_latest_record_by_phone(account.phone)
                            if latest_record and latest_record.get('checkin_total_times', 0) > 0:
                                file_logger.warning(f"⚠️ 签到次数识别可能错误：当前识别为0，但历史记录为{latest_record.get('checkin_total_times')}")
                                concise.action(f"签到次数异常：当前0，历史{latest_record.get('checkin_total_times')}")
                        
                        # 记录签到次数和奖励
                        if result.checkin_total_times:
                            file_logger.info(f"总签到次数: {result.checkin_total_times}")
                        file_logger.info(f"签到奖励: {result.checkin_reward:.2f} 元")
                        
                    else:
                        # 签到失败，设置错误类型
                        from .models.error_types import ErrorType
                        result.error_type = ErrorType.CHECKIN_FAILED
                        result.error_message = checkin_result.get('message', '未知错误')
                        file_logger.error(f"签到失败: {result.error_message}")
                        
                        # 添加简洁日志：签到失败
                        concise.error(f"签到失败: {result.error_message}")
                        
                        # [2026-03-29] 修复：签到失败直接返回，不继续执行后续步骤（避免误判为成功）
                        result.success = False
                        return result
                    
                except Exception as e:
                    # 签到异常，设置错误类型
                    from .models.error_types import ErrorType
                    result.error_type = ErrorType.CHECKIN_EXCEPTION
                    result.error_message = str(e)
                    file_logger.error(f"签到流程出错: {str(e)}")
                    
                    # [2026-03-29] 修复：签到异常直接返回，不继续执行后续步骤
                    result.success = False
                    return result
            
            # 优化：移除签到后的1秒等待，直接进入下一步
            # await asyncio.sleep(1)  # 已移除
            
            # 调试日志：确认执行到这里
            file_logger.info("签到流程结束，准备执行步骤4")
            
            # ==================== 步骤4: 设置最终余额 ====================
            # 注意：此处不再提前设置 balance_after
            # 最终余额的设置逻辑：
            # 1. 如果不需要转账 → 在转账判断后设置 balance_after = checkin_balance_after
            # 2. 如果需要转账 → 转账完成后重新获取实际余额并设置 balance_after
            # 这样逻辑更清晰，避免混淆
            
            # 只登录模式：使用初始余额作为最终余额
            if not workflow_config.get('enable_checkin', True):
                file_logger.info("="*60)
                file_logger.info("跳过获取最终余额")
                file_logger.info("原因: 流程控制已禁用签到，无需获取最终余额")
                file_logger.info("="*60)
                
                file_logger.info(f"只登录模式 - balance_before = {result.balance_before}")
                if result.balance_before is not None:
                    result.balance_after = result.balance_before
                    file_logger.info(f"使用初始余额作为最终余额: {result.balance_after:.2f} 元")
                    concise.success(f"最终余额: {result.balance_after:.2f}元（使用登录数据）")
                else:
                    file_logger.warning("balance_before 为 None，无法设置 balance_after")
                    concise.error("未获取到余额")
            
            # 显示执行总结（仅文件日志）
            file_logger.info("="*60)
            file_logger.info("执行总结")
            if result.balance_before is not None:
                file_logger.info(f"余额前: {result.balance_before:.2f} 元")
            file_logger.info(f"签到奖励: {result.checkin_reward:.2f} 元")
            if result.balance_after is not None:
                file_logger.info(f"余额后: {result.balance_after:.2f} 元")
            if result.balance_change is not None:
                file_logger.info(f"余额变化: {result.balance_change:+.2f} 元")
            file_logger.info("="*60)
            
            # 调试日志：确认执行到这里
            file_logger.info("步骤4完成，准备执行步骤5")
            
            # ==================== 步骤5: 自动转账（每次处理账号时重新读取配置）====================
            step_number += 1
            file_logger.info("="*60)
            file_logger.info(f"步骤{step_number}: 检查自动转账")
            file_logger.info("="*60)
            
            # 检查是否启用转账流程
            if not workflow_config.get('enable_transfer', True):
                file_logger.info("跳过转账流程")
                file_logger.info("原因: 流程控制已禁用转账")
                file_logger.info("="*60)
            else:
                try:
                    # 每次处理账号时重新读取转账配置
                    from .transfer_config import get_transfer_config
                    from .transfer_history import get_transfer_history
                    from .recipient_selector import RecipientSelector
                    
                    transfer_config = get_transfer_config()
                    transfer_history = get_transfer_history()
                    
                    # 创建收款人选择器（根据配置的策略）
                    recipient_selector = RecipientSelector(strategy=transfer_config.recipient_selection_strategy)
                    
                    should_auto_transfer = transfer_config.enabled
                    
                    # 获取账号级别（用于判断是否是收款账号）
                    account_level = 0
                    if result.user_id:
                        account_level = transfer_config.get_account_level(result.user_id)
                    
                    if should_auto_transfer:
                        log(f"✓ 自动转账功能已启用，开始检查转账条件...")
                        
                        # 检查1：有用户ID吗？
                        if not result.user_id:
                            log(f"  ⚠️ 无用户ID，跳过转账")
                            # 不需要转账，设置最终余额
                            if result.checkin_balance_after is not None:
                                result.balance_after = result.checkin_balance_after
                                file_logger.info(f"最终余额: {result.balance_after:.2f} 元（签到后余额，无用户ID）")
                            elif result.balance_before is not None:
                                result.balance_after = result.balance_before
                                file_logger.info(f"最终余额: {result.balance_after:.2f} 元（签到前余额，无用户ID）")
                        
                        # 检查2：有签到后余额吗？
                        elif result.checkin_balance_after is None:
                            log(f"  ⚠️ 无签到后余额，跳过转账")
                            # 不需要转账，使用签到前余额作为最终余额
                            if result.balance_before is not None:
                                result.balance_after = result.balance_before
                                file_logger.info(f"最终余额: {result.balance_after:.2f} 元（签到前余额，无签到后余额）")
                        
                        # 检查3：余额达到转账条件吗？
                        # [2026-03-25] 修复：传递签到次数参数，签到流程已获取并存储
                        elif transfer_config.should_transfer(result.user_id, result.checkin_balance_after, current_level=0, checkin_total_times=result.checkin_total_times):
                            # 检查4：有收款人配置吗？
                            recipient_id = transfer_config.get_transfer_recipient_enhanced(
                                phone=account.phone,
                                user_id=result.user_id,
                                current_level=0,
                                selector=recipient_selector
                            )
                            
                            if recipient_id:
                                log(f"  ✓ 满足转账条件，准备转账到 ID: {recipient_id}")
                                
                                try:
                                    # [2026-03-02] 修复：签到完成后已在个人页，直接转账
                                    # 初始化 transfer_result 避免 UnboundLocalError
                                    transfer_result = None
                                    
                                    # 转账重试机制（最多重试3次）
                                    max_transfer_retries = 3
                                    
                                    for transfer_attempt in range(max_transfer_retries):
                                        if transfer_attempt > 0:
                                            log(f"  ⚠️ 第 {transfer_attempt + 1} 次尝试转账...")
                                            # [2026-03-02] 优化：减少重试等待 3秒→1.5秒
                                            await asyncio.sleep(1.5)  # 重试前等待
                                            
                                            # 重试前重新导航到个人页面
                                            log(f"  重新导航到个人页面...")
                                            nav_success = await self.navigator.navigate_to_profile(device_id)
                                            if not nav_success:
                                                log(f"  ⚠️ 导航失败，跳过本次重试")
                                                continue  # 跳过本次重试，不继续转账
                                            # [2026-03-02] 优化：减少等待时间 2秒→1秒
                                            await asyncio.sleep(1)
                                        
                                        # 执行转账（传入签到后余额作为转账前余额）
                                        transfer_result = await self.balance_transfer.transfer_balance(
                                            device_id,
                                            recipient_id,
                                            initial_balance=result.checkin_balance_after,
                                            log_callback=log,
                                            gui_logger=gui_logger_obj
                                        )
                                        
                                        # 如果转账成功，跳出重试循环
                                        if transfer_result['success']:
                                            if transfer_attempt > 0:
                                                log(f"  ✓ 重试成功！")
                                            break
                                        else:
                                            # 转账失败，记录失败原因
                                            error_msg = transfer_result.get('message', '未知错误')
                                            if transfer_attempt < max_transfer_retries - 1:
                                                log(f"  ❌ 转账失败: {error_msg}，准备重试...")
                                            else:
                                                log(f"  ❌ 转账失败: {error_msg}，已达到最大重试次数")
                                    
                                    # [2026-03-02] 修复原因：检查 transfer_result 是否为 None
                                    if transfer_result and transfer_result['success']:
                                        log(f"  ✓ 转账成功")
                                        if transfer_result.get('amount'):
                                            log(f"    - 转账金额: {transfer_result['amount']:.2f} 元")
                                            
                                            # 保存转账信息到result对象
                                            result.transfer_amount = transfer_result.get('amount', 0.0)
                                            result.transfer_recipient = recipient_id
                                            log(f"    - 已保存转账信息: {result.transfer_amount:.2f} 元 → {result.transfer_recipient}")
                                            
                                            # 转账成功后，重新获取实际余额（不要用计算，要获取真实余额）
                                            log(f"    - 重新获取转账后的实际余额...")
                                            
                                            # 添加重试机制（最多3次）
                                            max_retries = 3
                                            balance_retrieved = False
                                            
                                            for retry in range(max_retries):
                                                try:
                                                    if retry > 0:
                                                        log(f"    - 第{retry+1}次尝试获取转账后余额...")
                                                        await asyncio.sleep(1.0)  # 重试前等待
                                                    
                                                    # [2026-03-12] 修复原因：使用转账专用模型检测页面，支持钱包页识别
                                                    # 检测当前页面状态，如果不在个人页则导航过去
                                                    page_result = await self.transfer_detector.detect_page(device_id, use_cache=False)
                                                    
                                                    if page_result and page_result.state == PageState.WALLET:
                                                        # 在钱包页，按返回键回到个人页
                                                        log(f"    - 当前在钱包页，返回个人页...")
                                                        await self.adb.press_back(device_id)
                                                        await asyncio.sleep(1.0)
                                                    elif page_result and page_result.state not in [PageState.PROFILE, PageState.PROFILE_LOGGED]:
                                                        # 不在个人页，导航到个人页
                                                        log(f"    - 导航到个人页...")
                                                        await self.navigator.navigate_to_profile(device_id)
                                                        await asyncio.sleep(1.0)
                                                    else:
                                                        # 已经在个人页，直接获取余额
                                                        log(f"    - 已在个人页，直接获取余额...")
                                                        await asyncio.sleep(0.5)  # 短暂等待页面稳定
                                                    
                                                    # 获取转账后的完整个人资料
                                                    account_str = f"{account.phone}----{account.password}"
                                                    profile_task = self.profile_reader.get_full_profile_parallel(device_id, account=account_str)
                                                    final_profile = await asyncio.wait_for(profile_task, timeout=15.0)
                                                    
                                                    # 更新最终余额
                                                    balance = final_profile.get('balance')
                                                    if balance is not None:
                                                        if isinstance(balance, str):
                                                            try:
                                                                result.balance_after = float(balance)
                                                            except ValueError:
                                                                result.balance_after = None
                                                        else:
                                                            result.balance_after = balance
                                                        
                                                        if result.balance_after is not None:
                                                            log(f"    - ✓ 转账后实际余额: {result.balance_after:.2f} 元")
                                                            balance_retrieved = True
                                                            
                                                            # 更新其他可能变化的字段
                                                            if final_profile.get('points') is not None:
                                                                result.points = final_profile.get('points')
                                                            if final_profile.get('vouchers') is not None:
                                                                result.vouchers = final_profile.get('vouchers')
                                                            # [2026-03-01] 删除优惠券：个人页已经没有优惠券了
                                                            
                                                            break  # 成功获取，跳出重试循环
                                                        else:
                                                            log(f"    - ⚠️ 第{retry+1}次获取余额为None")
                                                    else:
                                                        log(f"    - ⚠️ 第{retry+1}次未能获取余额")
                                                        
                                                except asyncio.TimeoutError:
                                                    log(f"    - ⚠️ 第{retry+1}次获取余额超时")
                                                except Exception as e:
                                                    log(f"    - ⚠️ 第{retry+1}次获取余额失败: {e}")
                                            
                                            # 如果所有重试都失败，使用计算值
                                            if not balance_retrieved:
                                                log(f"    - ⚠️ 所有重试都失败，使用计算值")
                                                # 使用签到后余额减去转账金额
                                                if result.checkin_balance_after is not None:
                                                    result.balance_after = result.checkin_balance_after - transfer_result['amount']
                                                    log(f"    - 转账后余额(计算): {result.balance_after:.2f} 元")
                                                else:
                                                    log(f"    - ⚠️ 无法计算转账后余额(缺少签到后余额)")
                                                    result.balance_after = None
                                            
                                            # 保存转账记录到数据库
                                            try:
                                                # 获取收款人信息
                                                recipient_name = transfer_result.get('recipient_name', '')
                                                # 收款人手机号暂时使用用户ID（因为配置中没有存储手机号）
                                                recipient_phone = recipient_id
                                                
                                                # 获取转账策略描述
                                                if transfer_config.multi_level_enabled:
                                                    strategy = f"多级转账(最多{transfer_config.max_transfer_level}级)"
                                                else:
                                                    strategy = "单级转账"
                                                
                                                # 获取账号的管理员信息
                                                owner_name = "未分配"  # 默认值
                                                try:
                                                    from .user_manager import UserManager
                                                    user_manager = UserManager()
                                                    user = user_manager.get_account_user(account.phone)
                                                    if user:
                                                        owner_name = user.user_name
                                                except Exception as e:
                                                    log(f"    - ⚠️ 获取管理员信息失败: {e}")
                                                
                                                # 保存转账记录
                                                save_success = transfer_history.save_transfer_record(
                                                    sender_phone=account.phone,
                                                    sender_user_id=result.user_id,
                                                    sender_name=result.nickname or account.phone,
                                                    recipient_phone=recipient_phone,
                                                    recipient_name=recipient_name or recipient_id,
                                                    amount=transfer_result['amount'],
                                                    strategy=strategy,
                                                    success=True,
                                                    error_message="",
                                                    owner=owner_name
                                                )
                                                
                                                if save_success:
                                                    log(f"    - ✓ 转账记录已保存到数据库")
                                                else:
                                                    log(f"    - ⚠️ 转账记录保存失败")
                                            except Exception as e:
                                                log(f"    - ⚠️ 保存转账记录异常: {e}")
                                    else:
                                        # [2026-03-02] 修复：处理transfer_result为None的情况
                                        # 转账失败，设置错误类型和失败状态
                                        from .models.error_types import ErrorType
                                        result.error_type = ErrorType.TRANSFER_FAILED
                                        
                                        # 安全获取错误消息
                                        if transfer_result is None:
                                            result.error_message = "转账失败：无返回结果"
                                        else:
                                            result.error_message = transfer_result.get('message', '未知错误')
                                        
                                        result.success = False  # 标记整体失败
                                        log(f"  ❌ 转账失败: {result.error_message}")
                                        # 转账失败时，设置收款人为"失败"
                                        result.transfer_amount = 0.0
                                        result.transfer_recipient = "失败"
                                        
                                        # 保存失败记录到数据库
                                        try:
                                            # 安全获取收款人信息
                                            if transfer_result is None:
                                                recipient_name = ''
                                                recipient_phone = recipient_id
                                            else:
                                                recipient_name = transfer_result.get('recipient_name', '')
                                                recipient_phone = recipient_id
                                            
                                            # 获取转账策略描述
                                            if transfer_config.multi_level_enabled:
                                                strategy = f"多级转账(最多{transfer_config.max_transfer_level}级)"
                                            else:
                                                strategy = "单级转账"
                                            
                                            # 获取账号的管理员信息
                                            owner_name = "未分配"  # 默认值
                                            try:
                                                from .user_manager import UserManager
                                                user_manager = UserManager()
                                                user = user_manager.get_account_user(account.phone)
                                                if user:
                                                    owner_name = user.user_name
                                            except Exception as e:
                                                log(f"    - ⚠️ 获取管理员信息失败: {e}")
                                            
                                            transfer_history.save_transfer_record(
                                                sender_phone=account.phone,
                                                sender_user_id=result.user_id,
                                                sender_name=result.nickname or account.phone,
                                                recipient_phone=recipient_phone,
                                                recipient_name=recipient_name or recipient_id,
                                                amount=0.0,
                                                strategy=strategy,
                                                success=False,
                                                error_message=result.error_message,
                                                owner=owner_name
                                            )
                                            log(f"    - ✓ 失败记录已保存到数据库")
                                        except Exception as e:
                                            log(f"    - ⚠️ 保存失败记录异常: {e}")
                                except Exception as e:
                                    # 转账异常，设置错误类型和失败状态
                                    from .models.error_types import ErrorType
                                    result.error_type = ErrorType.TRANSFER_FAILED
                                    result.error_message = str(e)
                                    result.success = False  # 标记整体失败
                                    log(f"  ❌ 转账异常: {e}")
                                    import traceback
                                    log(f"  详细错误: {traceback.format_exc()}")
                                    # 转账异常时，设置收款人为"失败"
                                    result.transfer_amount = 0.0
                                    result.transfer_recipient = "失败"
                                    
                                    # 保存异常记录到数据库
                                    try:
                                        strategy = transfer_config.strategy
                                        
                                        # 获取账号的管理员信息
                                        owner_name = "未分配"  # 默认值
                                        try:
                                            from .user_manager import UserManager
                                            user_manager = UserManager()
                                            user = user_manager.get_account_user(account.phone)
                                            if user:
                                                owner_name = user.user_name
                                        except Exception as e:
                                            log(f"    - ⚠️ 获取管理员信息失败: {e}")
                                        
                                        transfer_history.save_transfer_record(
                                            sender_phone=account.phone,
                                            sender_user_id=result.user_id,
                                            sender_name=result.nickname or account.phone,
                                            recipient_phone=recipient_id,
                                            recipient_name=recipient_id,
                                            amount=0.0,
                                            strategy=strategy,
                                            success=False,
                                            error_message=result.error_message,
                                            owner=owner_name
                                        )
                                        log(f"    - ✓ 异常记录已保存到数据库")
                                    except Exception as save_error:
                                        log(f"    - ⚠️ 保存异常记录失败: {save_error}")
                            else:
                                log(f"  ⚠️ 未配置收款人ID")
                                # 不需要转账，设置最终余额
                                result.balance_after = result.checkin_balance_after
                                file_logger.info(f"最终余额: {result.balance_after:.2f} 元（签到后余额，无收款人）")
                        else:
                            # 判断不转账的原因
                            if account_level > 0:
                                # 是收款账号
                                log(f"  ℹ️ 当前账号是 {account_level} 级收款账号，不进行转账")
                                if not transfer_config.multi_level_enabled:
                                    log(f"    - 多级转账功能未启用")
                                elif account_level >= transfer_config.max_transfer_level:
                                    log(f"    - 已达到最大转账级别 ({transfer_config.max_transfer_level})")
                                else:
                                    next_level = account_level + 1
                                    if not transfer_config.get_recipients(next_level):
                                        log(f"    - 未配置 {next_level} 级收款账号")
                            elif not transfer_config.recipient_ids:
                                log(f"  ⚠️ 未配置收款账号")
                            else:
                                # 余额不足
                                log(f"  ℹ️ 余额未达标")
                            
                            # 不需要转账，设置最终余额
                            result.balance_after = result.checkin_balance_after
                            file_logger.info(f"最终余额: {result.balance_after:.2f} 元（签到后余额，不满足转账条件）")
                    else:
                        log(f"  ℹ️ 自动转账功能未启用，跳过转账")
                        # 不需要转账，设置最终余额
                        if result.checkin_balance_after is not None:
                            result.balance_after = result.checkin_balance_after
                            file_logger.info(f"最终余额: {result.balance_after:.2f} 元（签到后余额，转账功能未启用）")
                        elif result.balance_before is not None:
                            result.balance_after = result.balance_before
                            file_logger.info(f"最终余额: {result.balance_after:.2f} 元（签到前余额，转账功能未启用）")
                    
                    log("")  # 空行
                except Exception as e:
                    log(f"  ❌ 转账检查出错: {str(e)}\n")
            
            # ==================== 步骤6: 退出登录 ====================
            step_number += 1
            
            file_logger.info("="*60)
            file_logger.info(f"步骤{step_number}: 退出登录")
            file_logger.info("="*60)
            
            try:
                await self.auto_login.logout(device_id)
                file_logger.info("已退出登录")
            except Exception as e:
                file_logger.error(f"退出登录失败: {str(e)}")
            
            # 最终检查：如果最终余额为 None，标记为失败
            if result.balance_after is None:
                result.success = False
                if not result.error_message:
                    result.error_message = "最终余额获取失败"
                file_logger.error(f"[最终检查] 最终余额为 None，标记为失败")
            else:
                result.success = True
            
            # 记录账号处理结束
            account_logger.log_account_end(
                account.phone,
                result.success,
                result.error_message if not result.success else "处理成功"
            )
            
            return result
            
        except Exception as e:
            result.error_message = str(e)
            file_logger.error(f"工作流程出错: {str(e)}")
            
            # [2026-03-05] 添加完整的堆栈跟踪，方便调试
            import traceback
            full_traceback = traceback.format_exc()
            file_logger.error(f"完整错误堆栈:\n{full_traceback}")
            print(f"\n{'='*60}")
            print(f"[错误] 账号 {account.phone} 处理失败")
            print(f"错误信息: {str(e)}")
            print(f"完整堆栈跟踪:")
            print(full_traceback)
            print(f"{'='*60}\n")
            
            # 记录账号处理结束（异常情况）
            account_logger.log_account_end(
                account.phone,
                False,
                f"异常: {str(e)}"
            )
            
            return result
    
    def _log_profile_data(self, result: AccountResult):
        """显示收集到的账户信息
        
        Args:
            result: 账号处理结果
        """
        print(f"\n{'='*60}")
        print(f"账户信息")
        print(f"{'='*60}")
        print(f"  昵称: {result.nickname or 'N/A'}")
        print(f"  ID: {result.user_id or 'N/A'}")
        print(f"  手机号: {result.phone}")
        
        if result.balance_before is not None:
            print(f"  余额: {result.balance_before:.2f} 元")
        else:
            print(f"  余额: N/A")
        
        if result.points is not None:
            print(f"  积分: {result.points} 积分")
        else:
            print(f"  积分: N/A")
        
        if result.vouchers is not None:
            print(f"  抵扣券: {result.vouchers} 张")
        else:
            print(f"  抵扣券: N/A")
        
        print(f"{'='*60}\n")
    
    def _log_checkin_results(self, rewards: List[float], total: float):
        """显示签到结果
        
        Args:
            rewards: 签到奖励列表
            total: 总奖励金额
        """
        print(f"\n{'='*60}")
        print(f"签到结果")
        print(f"{'='*60}")
        
        for i, reward in enumerate(rewards, 1):
            print(f"  [签到 {i}/{len(rewards)}] 奖励: {reward:.2f} 元")
        
        print(f"  ✓ 签到完成，总奖励: {total:.2f} 元")
        print(f"{'='*60}\n")
    
    def _log_summary(self, result: AccountResult):
        """显示执行总结
        
        Args:
            result: 账号处理结果
        """
        print(f"\n{'='*60}")
        print(f"执行总结")
        print(f"{'='*60}")
        
        if result.balance_before is not None:
            print(f"  余额前: {result.balance_before:.2f} 元")
        else:
            print(f"  余额前: N/A")
        
        print(f"  签到奖励: {result.checkin_reward:.2f} 元")
        
        if result.checkin_balance_after is not None:
            print(f"  余额: {result.checkin_balance_after:.2f} 元")
        else:
            print(f"  余额: N/A")
        
        if result.balance_after is not None:
            print(f"  余额: {result.balance_after:.2f} 元")
        else:
            print(f"  余额: N/A")
        
        if result.balance_change is not None:
            print(f"  余额变化: {result.balance_change:+.2f} 元")
        else:
            print(f"  余额变化: 无变化")
        
        print(f"{'='*60}\n")
    
    def _log_collection_error(self, field: str, error: Exception):
        """记录字段获取失败
        
        Args:
            field: 字段名称（如"昵称"、"余额"等）
            error: 异常对象
        """
        print(f"  ⚠️ 获取{field}失败: {str(error)}")
    
    def _log_partial_success(self, collected_fields: List[str], failed_fields: List[str]):
        """记录部分成功的数据收集结果
        
        Args:
            collected_fields: 成功收集的字段列表
            failed_fields: 收集失败的字段列表
        """
        if collected_fields:
            print(f"  ✓ 成功获取: {', '.join(collected_fields)}")
        
        if failed_fields:
            print(f"  ✗ 获取失败: {', '.join(failed_fields)}")


