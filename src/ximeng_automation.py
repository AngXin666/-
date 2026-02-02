"""
溪盟商城业务自动化模块
Ximeng Mall Business Automation Module
"""

import asyncio
import time
from typing import Optional, List, Dict, Any
from datetime import datetime

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
        
        # 获取共享的检测器实例
        self.integrated_detector = model_manager.get_page_detector_integrated()
        self.hybrid_detector = model_manager.get_page_detector_hybrid()
        
        # 优先使用整合检测器（深度学习，更快更准确）
        self.detector = self.integrated_detector
        
        # 初始化其他组件（使用共享检测器）
        from .navigator import Navigator
        from .balance_reader import BalanceReader
        from .daily_checkin import DailyCheckin
        from .profile_reader import ProfileReader
        from .page_state_guard import PageStateGuard
        
        # 所有组件都使用整合检测器（深度学习）
        self.navigator = Navigator(self.adb, self.detector)
        self.balance_reader = BalanceReader(self.adb)
        self.daily_checkin = DailyCheckin(self.adb, self.detector, self.navigator)
        
        # 初始化ProfileReader，传入整合检测器
        self.profile_reader = ProfileReader(self.adb, yolo_detector=self.detector)
        
        # 初始化页面状态守卫（使用整合检测器）
        self.guard = PageStateGuard(self.adb, self.detector)
        
        # 从ModelManager获取OCR线程池
        self._ocr_enhancer = model_manager.get_ocr_thread_pool()
        
        # 初始化转账模块（使用整合检测器）
        from .balance_transfer import BalanceTransfer
        self.balance_transfer = BalanceTransfer(self.adb, self.detector)
    
    async def wait_for_app_ready(self, device_id: str, timeout: int = 30) -> bool:
        """等待应用准备就绪（处理启动页、广告等）
        
        Args:
            device_id: 设备 ID
            timeout: 超时时间（秒）
            
        Returns:
            是否成功进入主界面
        """
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            # 检测当前页面状态
            result = await self.page_detector.detect_page(device_id)
            
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
    
    async def handle_startup_flow(self, device_id: str, log_callback=None, stop_check=None, 
                                   package_name: str = "com.ry.xmsc", activity_name: str = None,
                                   max_retries: int = 3,
                                   stuck_timeout: int = 15, max_wait_time: int = 60,
                                   enable_debug: bool = True) -> bool:
        """处理应用启动流程 - 智能自适应检测，无需手动配置参数
        
        流程：启动页(等待) -> 用户协议弹窗(关闭) -> 广告页(智能等待) -> 首页弹窗(关闭) -> 主页
        如果白屏卡死，清理缓存并重启应用
        
        智能特性：
        - 自适应广告等待：每2秒检测一次，广告消失后立即继续（最多15秒）
        - 自适应额外检测：连续2次检测到相同状态则提前结束（最多5次）
        - 无需配置 ad_wait_time 和 extra_check_attempts 参数
        
        Args:
            device_id: 设备 ID
            log_callback: 日志回调函数
            stop_check: 停止检查函数，返回 True 表示需要停止
            package_name: 应用包名
            max_retries: 最大重试次数（默认3次）
            stuck_timeout: 白屏卡住检测时间（秒，默认15秒）
            max_wait_time: 最大等待时间（秒，默认60秒）
            enable_debug: 是否启用调试日志（默认True）
            
        Returns:
            是否成功
        """
        def log(msg):
            if log_callback:
                log_callback(msg)
        
        def should_stop():
            if stop_check:
                return stop_check()
            return False
        
        # 初始化调试日志
        debug_logger = None
        if enable_debug:
            from .debug_logger import DebugLogger
            debug_logger = DebugLogger()
            debug_logger.log_step("开始应用启动流程", f"包名: {package_name}")
            log(f"调试日志已启用，保存到: {debug_logger.session_dir}")
        
        # 复用已有的混合检测器（避免重复加载模板）
        hybrid_detector = self.hybrid_detector
        
        for retry in range(max_retries):
            if should_stop():
                log("用户请求停止")
                if debug_logger:
                    debug_logger.log_warning("用户请求停止")
                    debug_logger.close()
                return False
            
            if retry > 0:
                log(f"⚠️ 第 {retry + 1} 次尝试启动应用...")
                if debug_logger:
                    debug_logger.log_step(f"重试 {retry + 1}/{max_retries}", "白屏卡死，重新启动")
                
                # 停止应用
                await self.adb.stop_app(device_id, package_name)
                await asyncio.sleep(1)
                # 只清理缓存，不清理数据（保留登录状态）
                log("清理应用缓存（保留登录数据）...")
                # 方法1：尝试使用 pm clear-cache（Android 6.0+）
                result = await self.adb.shell(device_id, f"pm clear-cache {package_name}")
                if "Unknown" in result or "Error" in result:
                    # 方法2：如果 pm clear-cache 不支持，直接删除缓存目录
                    log("pm clear-cache 不支持，使用 rm 命令清理缓存...")
                    result = await self.adb.shell(device_id, f"rm -rf /data/data/{package_name}/cache/*")
                log(f"清理结果: {result.strip() if result.strip() else '成功'}")
                if debug_logger:
                    debug_logger.log_action("清理缓存", {"结果": result.strip() if result.strip() else '成功'})
                await asyncio.sleep(2)
                # 重新启动应用
                success = await self.adb.start_app(device_id, package_name, activity_name)
                log(f"启动{'成功' if success else '失败'}")
                if debug_logger:
                    debug_logger.log_action("启动应用", {"成功": success})
                await asyncio.sleep(3)
            else:
                # 第一次启动，应用已经由调用者启动
                # 优化：去掉固定等待，立即开始检测
                log("[优化] 应用已启动，立即开始检测...")
            
            loading_count = 0  # 连续LOADING状态的次数，用于检测白屏卡死
            stuck = False  # 是否卡死需要重试
            
            # 定义启动流程的优先级模板列表（只检测6个相关模板）
            startup_templates = [
                '加载卡死白屏.png',      # 最可能：白屏卡死
                '启动页服务弹窗.png',    # 可能：用户协议弹窗
                '广告.png',              # 可能：广告页
                '首页公告.png',          # 可能：首页公告弹窗
                '首页.png',              # 可能：首页
                '登陆.png',              # 可能：登录页
            ]
            
            # 尝试检测页面状态（每秒检查一次，但每0.2秒检查停止信号）
            for attempt in range(max_wait_time):
                # 高频检查停止信号（每0.2秒检查一次，共5次 = 1秒）
                for _ in range(5):
                    if should_stop():
                        log("用户请求停止")
                        if debug_logger:
                            debug_logger.log_warning("用户请求停止")
                            debug_logger.close()
                        return False
                    await asyncio.sleep(0.2)
                
                # 使用优先级模板检测（只匹配启动流程相关的7个模板）
                result = await hybrid_detector.detect_page_with_priority(
                    device_id,
                    startup_templates,
                    use_cache=True
                )
                log(f"[优化] [{attempt+1}/{max_wait_time}] {result.state.value}: {result.details}")
                
                # 保存截图和OCR结果到调试日志
                if debug_logger and attempt % 5 == 0:  # 每5秒保存一次
                    await debug_logger.save_screenshot(
                        self.adb, device_id, 
                        f"attempt_{attempt+1}",
                        f"状态: {result.state.value}"
                    )
                    # 获取OCR文本（使用公开方法）
                    texts = hybrid_detector.get_last_screenshot_texts()
                    if texts:
                        debug_logger.log_page_detection(
                            result.state.value, result.confidence, result.details, texts
                        )
                
                # 如果已经到达主页或我的页面，说明启动完成
                if result.state in [PageState.HOME, PageState.PROFILE, PageState.PROFILE_LOGGED]:
                    log("[优化] ✓ 已进入主界面")
                    if debug_logger:
                        debug_logger.log_result(True, f"成功到达: {result.state.value}")
                        log_path = debug_logger.close()
                        log(f"[优化] 调试日志已保存: {log_path}")
                    return True
                
                # 如果是登录页面，说明已经跳过广告
                if result.state == PageState.LOGIN:
                    log("[优化] ✓ 已到达登录页面")
                    if debug_logger:
                        debug_logger.log_result(True, "成功到达登录页面")
                        log_path = debug_logger.close()
                        log(f"[优化] 调试日志已保存: {log_path}")
                    return True
                
                # 如果是Android桌面，说明应用还没启动或已退出
                if result.state == PageState.LAUNCHER:
                    log("检测到Android桌面，尝试启动应用...")
                    if debug_logger:
                        debug_logger.log_step("检测到Android桌面", result.details)
                        await debug_logger.save_screenshot(
                            self.adb, device_id, "launcher", "Android桌面"
                        )
                    # 尝试启动应用
                    success = await self.adb.start_app(device_id, package_name, activity_name)
                    if success:
                        log("✓ 应用启动成功，等待加载...")
                    else:
                        log("⚠️ 应用启动失败")
                    loading_count = 0
                    await asyncio.sleep(3)
                    continue
                
                # 处理启动页服务弹窗（用户协议）- 使用YOLO检测按钮
                if result.state == PageState.STARTUP_POPUP:
                    log("[YOLO] 检测到启动页服务弹窗，使用YOLO检测'同意'按钮...")
                    if debug_logger:
                        debug_logger.log_step("处理启动页服务弹窗（YOLO）", result.details)
                        await debug_logger.save_screenshot(
                            self.adb, device_id, "startup_popup_before", "关闭前"
                        )
                    
                    # 使用YOLO检测并点击"同意"按钮
                    success = await hybrid_detector.click_button_yolo(
                        device_id, 'startup_popup', '同意按钮', conf_threshold=0.5
                    )
                    
                    if success:
                        log("[YOLO] ✓ 成功点击'同意'按钮")
                    else:
                        log("[YOLO] ⚠️ 未找到'同意'按钮，尝试使用固定坐标...")
                        # 降级到固定坐标
                        await self.adb.tap(device_id, 270, 600)
                    
                    await asyncio.sleep(1.5)
                    
                    if debug_logger:
                        debug_logger.log_result(success, "关闭启动页服务弹窗")
                        await debug_logger.save_screenshot(
                            self.adb, device_id, "startup_popup_after", "关闭后"
                        )
                    
                    loading_count = 0
                    await asyncio.sleep(1.5)
                    continue
                
                # 处理首页公告弹窗 - 使用YOLO检测关闭按钮
                if result.state == PageState.HOME_NOTICE:
                    log("[YOLO] 检测到首页公告弹窗，使用YOLO检测关闭按钮...")
                    if debug_logger:
                        debug_logger.log_step("处理首页公告弹窗（YOLO）", result.details)
                        await debug_logger.save_screenshot(
                            self.adb, device_id, "home_notice_before", "关闭前"
                        )
                    
                    # 使用YOLO检测首页公告关闭按钮
                    try:
                        buttons = await hybrid_detector.detect_buttons_yolo(device_id, "首页公告")
                        if buttons and len(buttons) > 0:
                            # 找到确认按钮
                            confirm_button = None
                            for btn in buttons:
                                if btn.class_name == '确认按钮':
                                    confirm_button = btn
                                    break
                            
                            if confirm_button:
                                # 点击确认按钮
                                center_x, center_y = confirm_button.center
                                log(f"[YOLO] YOLO检测到确认按钮，位置: ({center_x}, {center_y}), 置信度: {confirm_button.confidence:.2f}")
                                await self.adb.tap(device_id, center_x, center_y)
                            else:
                                log("[YOLO] YOLO未检测到确认按钮，使用固定坐标...")
                                await self.adb.tap(device_id, 270, 690)
                        else:
                            log("[YOLO] YOLO未检测到按钮，使用固定坐标...")
                            await self.adb.tap(device_id, 270, 690)
                    except Exception as e:
                        log(f"[YOLO] YOLO检测失败: {e}，使用固定坐标...")
                        await self.adb.tap(device_id, 270, 690)
                    
                    await asyncio.sleep(1.5)
                    
                    if debug_logger:
                        debug_logger.log_result(True, "关闭首页公告弹窗")
                        await debug_logger.save_screenshot(
                            self.adb, device_id, "home_notice_after", "关闭后"
                        )
                    
                    loading_count = 0
                    await asyncio.sleep(1.5)
                    continue
                
                # 处理弹窗（用户协议、首页公告等）
                if result.state == PageState.POPUP:
                    log("[优化] 检测到弹窗，使用预加载优化关闭...")
                    if debug_logger:
                        debug_logger.log_step("处理弹窗（预加载）", result.details)
                        await debug_logger.save_screenshot(
                            self.adb, device_id, "popup_before", "关闭前"
                        )
                    
                    # 优化：在关闭弹窗前开始预加载下一页面检测
                    hybrid_detector.preload_detection(device_id)
                    
                    success = await hybrid_detector.close_popup(device_id)
                    if success:
                        log("[优化] ✓ 成功关闭弹窗")
                    else:
                        log("[优化] ⚠️ 关闭弹窗失败")
                    
                    # 等待页面切换
                    await asyncio.sleep(1.5)
                    
                    # 获取预加载结果（几乎0延迟）
                    preload_result = await hybrid_detector.get_preloaded_result(device_id)
                    if preload_result:
                        log(f"[优化] ✓ 预加载检测完成: {preload_result.state.value}（感知延迟0ms）")
                        
                        # 如果已到达目标页面，直接返回
                        if preload_result.state in [PageState.HOME, PageState.PROFILE, PageState.PROFILE_LOGGED, PageState.LOGIN]:
                            log(f"[优化] ✓ 关闭弹窗后直接到达目标页面")
                            if debug_logger:
                                debug_logger.log_result(True, f"成功到达: {preload_result.state.value}")
                                log_path = debug_logger.close()
                                log(f"[优化] 调试日志已保存: {log_path}")
                            return True
                    
                    if debug_logger:
                        debug_logger.log_result(success, "关闭弹窗")
                        await debug_logger.save_screenshot(
                            self.adb, device_id, "popup_after", "关闭后"
                        )
                    
                    loading_count = 0  # 重置计数
                    await asyncio.sleep(1.5)
                    continue
                
                # 处理广告页（直接拦截，不点击）
                if result.state == PageState.AD:
                    log(f"[优化] 检测到广告页，智能快速轮询...")
                    if debug_logger:
                        debug_logger.log_step("处理广告页（优化）", "智能快速轮询")
                        await debug_logger.save_screenshot(
                            self.adb, device_id, "ad_detected", "检测到广告"
                        )
                    
                    # 等待广告自动跳过
                    log(f"[优化] 等待广告自动跳过（每0.5秒检测一次）...")
                    if debug_logger:
                        debug_logger.log_step("等待广告", "快速轮询")
                    
                    # 优化：广告页快速轮询（0.5秒/次，而不是2秒）
                    max_ad_wait = 15
                    check_interval = 0.5  # 停止检查间隔
                    ad_check_interval = 0.5  # 优化：从2秒改为0.5秒
                    ad_wait_elapsed = 0
                    last_ad_check = 0
                    
                    while ad_wait_elapsed < max_ad_wait:
                        # 先检查停止信号（高频）
                        if should_stop():
                            log("用户请求停止")
                            if debug_logger:
                                debug_logger.log_warning("用户请求停止")
                                debug_logger.close()
                            return False
                        
                        await asyncio.sleep(check_interval)
                        ad_wait_elapsed += check_interval
                        
                        # 优化：每0.5秒检测一次广告状态（而不是2秒）
                        if ad_wait_elapsed - last_ad_check >= ad_check_interval:
                            last_ad_check = ad_wait_elapsed
                            
                            # 检查广告是否已消失 - 使用优先级模板（广告相关）
                            ad_check_templates = ['广告.png', '首页.png', '启动页服务弹窗.png']
                            result_after = await hybrid_detector.detect_page_with_priority(
                                device_id, ad_check_templates, use_cache=False
                            )
                            if result_after.state != PageState.AD:
                                log(f"[优化] ✓ 广告已消失（用时{ad_wait_elapsed:.1f}秒）")
                                if debug_logger:
                                    debug_logger.log_result(True, f"广告消失（用时{ad_wait_elapsed:.1f}秒）")
                                    await debug_logger.save_screenshot(
                                        self.adb, device_id, "ad_after", "广告消失"
                                    )
                                break
                    
                    # 如果超时仍未消失，继续流程
                    if ad_wait_elapsed >= max_ad_wait:
                        log(f"[优化] ⚠️ 广告等待超时（{max_ad_wait}秒），继续流程...")
                        if debug_logger:
                            debug_logger.log_warning(f"广告等待超时（{max_ad_wait}秒）")
                            await debug_logger.save_screenshot(
                                self.adb, device_id, "ad_timeout", "超时"
                            )
                    
                    loading_count = 0  # 重置计数
                    continue
                
                # 处理未知页面（可能是异常页面，需要返回）
                if result.state == PageState.UNKNOWN:
                    log(f"检测到未知页面: {result.details}")
                    if debug_logger:
                        debug_logger.log_step("处理未知页面", result.details)
                        await debug_logger.save_screenshot(
                            self.adb, device_id, "unknown_page", "未知页面"
                        )
                    
                    # 检查是否是异常页面（需要返回）
                    if any(keyword in result.details for keyword in ["异常页面", "商品列表", "商品详情", "活动页面", "文章列表"]):
                        log("检测到异常页面，按返回键...")
                        await self.adb.press_back(device_id)
                        await asyncio.sleep(2)
                        loading_count = 0
                        continue
                    
                    # 其他未知页面，等待一下看是否会变化
                    log("未知页面，等待页面加载...")
                    loading_count = 0
                    await asyncio.sleep(1)
                    continue
                
                # 如果是加载中状态
                if result.state == PageState.LOADING:
                    loading_count += 1
                    
                    # 如果连续LOADING超过 stuck_timeout 秒，标记为白屏卡死
                    if loading_count >= stuck_timeout:
                        log(f"⚠️ 检测到白屏卡死（连续{loading_count}秒LOADING）")
                        if debug_logger:
                            debug_logger.log_warning(f"白屏卡死（连续{loading_count}秒LOADING）")
                            await debug_logger.save_screenshot(
                                self.adb, device_id, "stuck_screen", "白屏卡死"
                            )
                        stuck = True
                        break  # 跳出内层循环，进行重试
                    
                    if loading_count % 5 == 0:
                        log(f"仍在加载中... ({loading_count}秒)")
                    
                    await asyncio.sleep(1)
                    continue
                else:
                    # 不是LOADING状态，重置计数器
                    loading_count = 0
                
                # 其他未处理的状态，等待一下
                await asyncio.sleep(1)
            
            # 循环结束，检查最终状态
            if not stuck:
                # 再次检测当前页面状态 - 使用优先级模板（启动流程相关）
                final_check_templates = ['首页.png', '登陆.png', '已登陆个人页.png', '加载卡死白屏.png']
                final_result = await hybrid_detector.detect_page_with_priority(
                    device_id, final_check_templates, use_cache=False
                )
                log(f"循环结束，最终状态: {final_result.state.value}")
                
                # 如果已经到达目标页面，返回成功
                if final_result.state in [PageState.HOME, PageState.PROFILE, PageState.PROFILE_LOGGED, PageState.LOGIN]:
                    log("✓ 启动流程完成")
                    if debug_logger:
                        debug_logger.log_result(True, f"成功到达: {final_result.state.value}")
                        log_path = debug_logger.close()
                        log(f"调试日志已保存: {log_path}")
                    return True
                
                # 如果是弹窗或其他中间状态，智能处理
                elif final_result.state in [PageState.POPUP, PageState.SPLASH, PageState.LOADING]:
                    log(f"循环结束时状态为 {final_result.state.value}，智能处理...")
                    
                    # 如果是弹窗，尝试关闭
                    if final_result.state == PageState.POPUP:
                        log("尝试关闭弹窗...")
                        await hybrid_detector.close_popup(device_id)
                        await asyncio.sleep(3)
                    else:
                        # 其他状态，等待一下
                        await asyncio.sleep(3)
                    
                    # 自适应额外检测：连续2次检测到相同稳定状态则提前结束
                    max_extra_attempts = 5  # 最多5次
                    last_state = None
                    same_state_count = 0
                    
                    for extra_attempt in range(max_extra_attempts):
                        # 额外检测 - 使用优先级模板（启动流程相关）
                        extra_check_templates = ['首页.png', '登陆.png', '已登陆个人页.png', '首页公告.png']
                        final_result = await hybrid_detector.detect_page_with_priority(
                            device_id, extra_check_templates, use_cache=False
                        )
                        log(f"额外检测 {extra_attempt+1}/{max_extra_attempts}: {final_result.state.value}")
                        
                        # 检查是否到达目标页面
                        if final_result.state in [PageState.HOME, PageState.PROFILE, PageState.PROFILE_LOGGED, PageState.LOGIN]:
                            log("✓ 启动流程完成")
                            if debug_logger:
                                debug_logger.log_result(True, f"成功到达: {final_result.state.value}")
                                log_path = debug_logger.close()
                                log(f"调试日志已保存: {log_path}")
                            return True
                        
                        # 智能判断：如果连续2次检测到相同的非目标状态，说明页面稳定了
                        if final_result.state == last_state:
                            same_state_count += 1
                            if same_state_count >= 2:
                                log(f"⚠️ 连续{same_state_count}次检测到相同状态 {final_result.state.value}，页面已稳定，停止额外检测")
                                break
                        else:
                            same_state_count = 1
                            last_state = final_result.state
                        
                        # 如果还是弹窗，继续关闭
                        if final_result.state == PageState.POPUP:
                            log("仍有弹窗，继续关闭...")
                            await hybrid_detector.close_popup(device_id)
                        
                        await asyncio.sleep(2)
                    
                    # 额外尝试后仍未成功，标记为卡死
                    log(f"⚠️ 多次尝试后仍未到达目标页面: {final_result.state.value}")
                    stuck = True
                
                else:
                    # 其他未知状态，标记为卡死
                    log(f"⚠️ 超时未到达目标页面，当前状态: {final_result.state.value}")
                    if debug_logger:
                        debug_logger.log_warning(f"超时未到达目标页面: {final_result.state.value}")
                        await debug_logger.save_screenshot(
                            self.adb, device_id, "timeout", "超时"
                        )
                    stuck = True
            
            # 如果卡死了，继续下一次重试
            if stuck:
                log(f"⚠️ 启动失败，准备重试 ({retry + 1}/{max_retries})")
        
        # 所有重试都失败
        log(f"✗ 启动失败，已重试 {max_retries} 次，跳过此账号")
        if debug_logger:
            debug_logger.log_error(f"启动失败，已重试 {max_retries} 次")
            log_path = debug_logger.close()
            log(f"调试日志已保存: {log_path}")
        return False
    
    async def navigate_to_balance(self, device_id: str) -> bool:
        """导航到余额页面
        
        Args:
            device_id: 设备 ID
            
        Returns:
            是否成功导航
        """
        # 尝试点击"我的"或"个人中心"
        if await self.ui_automation.click_by_text(device_id, "我的", timeout=5):
            await asyncio.sleep(1)
            return True
        
        if await self.ui_automation.click_by_text(device_id, "个人中心", timeout=5):
            await asyncio.sleep(1)
            return True
        
        return False
    
    async def get_balance_and_profile_parallel(
        self, 
        device_id: str, 
        account: Optional[str] = None
    ) -> tuple:
        """并行获取余额和个人信息（优化版）
        
        优化策略：
        1. 只截图一次
        2. 并行解析余额和其他字段
        3. 减少总耗时
        
        Args:
            device_id: 设备ID
            account: 登录账号（可选）
            
        Returns:
            tuple: (余额, 个人信息字典)
        """
        try:
            self._silent_log.info(f" 开始并行获取余额和个人信息...")
            
            # 使用并行方法获取完整个人信息（包含余额）
            profile_data = await self.profile_reader.get_full_profile_parallel(device_id, account)
            
            # 从个人信息中提取余额
            balance = profile_data.get('balance') if profile_data else None
            
            if balance is not None:
                self._silent_log.info(f" ✓ 余额: {balance:.2f} 元")
            else:
                self._silent_log.info(f" ⚠️ 未能获取余额")
            
            self._silent_log.info(f" ✓ 并行获取完成")
            return balance, profile_data
            
        except Exception as e:
            self._silent_log.info(f" ❌ 异常: {e}")
            import traceback
            traceback.print_exc()
            return None, {}
    
    async def get_profile_info_optimized(
        self,
        device_id: str,
        profile_manager,
        account: Optional[str] = None,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """获取个人信息（优化版 - 使用管理器）
        
        这是一个封装方法,统一使用 ProfileInfoManager 来避免重复获取
        
        Args:
            device_id: 设备ID
            profile_manager: ProfileInfoManager实例
            account: 登录账号（可选）
            force_refresh: 是否强制刷新缓存
            
        Returns:
            dict: 个人信息
        """
        return await profile_manager.get_profile_info(
            device_id,
            self.profile_reader,
            account,
            force_refresh
        )
    
    async def _navigate_to_profile_with_ad_handling(self, device_id: str, log_callback=None) -> bool:
        """导航到个人页并处理广告（统一方法）
        
        核心逻辑：
        1. 点击"我的"按钮
        2. 高频扫描页面状态（每0.05秒）
        3. 检测到广告 → 立即关闭 → 继续扫描
        4. 检测到正常个人页 → 返回成功
        5. 超时（5秒）→ 返回失败
        
        Args:
            device_id: 设备ID
            log_callback: 日志回调函数
            
        Returns:
            bool: 是否成功到达个人页
        """
        log = log_callback if log_callback else self._silent_log.info
        
        # 点击底部导航栏"我的"按钮
        MY_TAB = (450, 920)
        await self.adb.tap(device_id, MY_TAB[0], MY_TAB[1])
        
        # 高频扫描，最多5秒
        max_scan_time = 5.0
        scan_interval = 0.05  # 每50毫秒扫描一次
        start_time = asyncio.get_event_loop().time()
        
        ad_closed_count = 0  # 记录关闭广告的次数
        
        while (asyncio.get_event_loop().time() - start_time) < max_scan_time:
            # 检测当前页面状态
            if self.integrated_detector:
                page_result = await self.integrated_detector.detect_page(
                    device_id, use_cache=False, detect_elements=False
                )
            else:
                # 降级到混合检测器
                balance_templates = ['已登陆个人页.png', '未登陆个人页.png', '个人页广告.png']
                page_result = await self.hybrid_detector.detect_page_with_priority(
                    device_id, balance_templates, use_cache=False
                )
            
            if not page_result or not page_result.state:
                await asyncio.sleep(scan_interval)
                continue
            
            from .page_detector_hybrid import PageState
            current_state = page_result.state
            
            # 检测到正常个人页 → 成功
            if current_state in [PageState.PROFILE, PageState.PROFILE_LOGGED]:
                elapsed = asyncio.get_event_loop().time() - start_time
                log(f"  ✓ 到达个人页（耗时: {elapsed:.2f}秒，关闭广告: {ad_closed_count}次）")
                return True
            
            # 检测到广告 → 立即关闭
            elif current_state == PageState.PROFILE_AD:
                log(f"  ⚠️ 检测到个人页广告，立即关闭...")
                
                # 使用YOLO检测关闭按钮
                close_button_pos = None
                if self.integrated_detector and hasattr(self.integrated_detector, '_yolo_detector') and self.integrated_detector._yolo_detector:
                    close_button_pos = await self.integrated_detector.find_button_yolo(
                        device_id, 
                        '个人页广告',
                        '确认按钮',
                        conf_threshold=0.5
                    )
                
                if close_button_pos:
                    log(f"  YOLO检测到'确认按钮': {close_button_pos}，点击关闭")
                    await self.adb.tap(device_id, close_button_pos[0], close_button_pos[1])
                else:
                    log(f"  使用固定坐标关闭广告")
                    await self.adb.tap(device_id, 437, 554)
                
                ad_closed_count += 1
                
                # 等待0.3秒让广告关闭动画完成
                await asyncio.sleep(0.3)
                
                # 清除缓存
                if self.integrated_detector:
                    self.integrated_detector.clear_cache(device_id)
                
                # 继续扫描（可能还有广告，或者已经到达个人页）
                continue
            
            # 其他状态 → 继续扫描
            else:
                await asyncio.sleep(scan_interval)
        
        # 超时
        elapsed = asyncio.get_event_loop().time() - start_time
        log(f"  ❌ 导航到个人页超时（耗时: {elapsed:.2f}秒，关闭广告: {ad_closed_count}次）")
        return False
    
    async def get_balance(self, device_id: str, from_cache_login: bool = False) -> Optional[float]:
        """获取账户余额（简化版 - 使用 profile_reader）
        
        Args:
            device_id: 设备 ID
            from_cache_login: 是否来自缓存登录（True=需要导航，False=已在个人页面）
            
        Returns:
            余额数值，获取失败返回 None
        """
        try:
            self._silent_log.info(f" 开始获取余额... (from_cache_login={from_cache_login})")
            
            # 如果不在个人页面，使用统一的导航方法
            if from_cache_login:
                self._silent_log.info(f" 导航到个人页...")
                success = await self._navigate_to_profile_with_ad_handling(device_id, self._silent_log.info)
                if not success:
                    self._silent_log.info(f" ❌ 导航失败")
                    return None
            else:
                self._silent_log.info(f" 已在个人页面，直接读取")
            
            # 直接获取个人资料（不再需要检测和处理广告）
            self._silent_log.info(f" 开始OCR识别...")
            # 使用并行处理方法提升性能
            profile_data = await self.profile_reader.get_full_profile_parallel(device_id)
            
            self._silent_log.info(f" OCR识别完成")
            
            if profile_data and profile_data.get('balance') is not None:
                balance = profile_data['balance']
                self._silent_log.info(f" ✓ 成功读取: {balance:.2f} 元")
                return balance
            
            print("  [余额] ❌ 未能读取到余额（profile_data中balance为None）")
            if profile_data:
                self._silent_log.info(f" 其他字段: nickname={profile_data.get('nickname')}, user_id={profile_data.get('user_id')}, points={profile_data.get('points')}, vouchers={profile_data.get('vouchers')}")
            else:
                self._silent_log.info(f" profile_data为空或None")
            return None
            
        except Exception as e:
            self._silent_log.info(f" ❌ 异常: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def get_profile_info(self, device_id: str, account: Optional[str] = None, from_cache_login: bool = False) -> Dict[str, any]:
        """获取个人信息（优化版：减少重复检测，支持并行处理）
        
        优化点：
        1. 减少重复页面检测
        2. 如果已在个人页，直接获取信息
        3. 导航时不再重复检测
        
        Args:
            device_id: 设备 ID
            account: 登录账号（可选），用于提取手机号
            from_cache_login: 是否来自缓存登录（True=需要导航，False=已在个人页面）
            
        Returns:
            dict: 个人信息
        """
        try:
            # 优化：如果明确知道来自缓存登录，直接导航，不检测页面
            if from_cache_login:
                self._silent_log.info(f" 缓存登录，直接导航到个人页面")
                success = await self.navigator.navigate_to_profile(device_id)
                if not success:
                    print("  [个人信息] ❌ 无法导航到个人页面")
                    return {
                        'nickname': None,
                        'user_id': None,
                        'phone': None,
                        'points': None,
                        'vouchers': None,
                        'total_draw_times': None
                    }
            else:
                # 只在不确定位置时才检测页面
                # 使用整合检测器（GPU加速深度学习）进行快速检测
                if self.integrated_detector:
                    self._silent_log.info(f" 使用整合检测器（GPU加速）检测页面...")
                    page_result = await self.integrated_detector.detect_page(
                        device_id, use_cache=True, detect_elements=False
                    )
                else:
                    # 降级到混合检测器
                    self._silent_log.info(f" 使用混合检测器检测页面...")
                    # 定义获取个人信息的优先级模板列表（只检测3个相关模板）
                    profile_templates = [
                        '已登陆个人页.png',      # 最可能：已登录个人页
                        '未登陆个人页.png',      # 可能：未登录个人页
                        '首页.png',              # 可能：首页
                    ]
                    
                    # 使用优先级模板检测当前页面状态
                    page_result = await self.hybrid_detector.detect_page_with_priority(
                        device_id,
                        profile_templates,
                        use_cache=True
                    )
                
                if not page_result or not page_result.state:
                    self._silent_log.info(f" ❌ 无法检测页面状态")
                    return {
                        'nickname': None,
                        'user_id': None,
                        'phone': None,
                        'points': None,
                        'vouchers': None,
                        'total_draw_times': None
                    }
                
                current_state = page_result.state
                
                # 检查是否未登录
                if current_state == PageState.PROFILE:
                    self._silent_log.info(f" ❌ 检测到未登录状态")
                    return {
                        'nickname': None,
                        'user_id': None,
                        'phone': None,
                        'points': None,
                        'vouchers': None,
                        'total_draw_times': None,
                        'need_login': True  # 标记需要登录
                    }
                
                # 如果不在个人页面，需要导航
                if current_state not in [PageState.PROFILE, PageState.PROFILE_LOGGED]:
                    self._silent_log.info(f" 当前页面: {current_state.value}，需要导航到个人页面")
                    success = await self.navigator.navigate_to_profile(device_id)
                    if not success:
                        print("  [个人信息] ❌ 无法导航到个人页面")
                        return {
                            'nickname': None,
                            'user_id': None,
                            'phone': None,
                            'points': None,
                            'vouchers': None,
                            'total_draw_times': None
                        }
                else:
                    # 已经在个人页面，不需要导航
                    self._silent_log.info(f" 当前在个人页面: {current_state.value}，无需导航")
            
            # 获取完整个人信息
            # 使用并行处理方法提升性能
            info = await self.profile_reader.get_full_profile_parallel(device_id, account)
            
            if 'total_draw_times' not in info:
                info['total_draw_times'] = None
            
            self._silent_log.info(f" ✓ 成功读取")
            return info
        except Exception as e:
            self._silent_log.info(f" ❌ 异常: {e}")
            return {
                'nickname': None,
                'user_id': None,
                'phone': None,
                'points': None,
                'vouchers': None,
                'total_draw_times': None
            }
    
    async def navigate_to_sign_in(self, device_id: str) -> bool:
        """导航到签到页面
        
        Args:
            device_id: 设备 ID
            
        Returns:
            是否成功导航
        """
        for pattern in self.SIGN_IN_PATTERNS:
            if await self.ui_automation.click_by_text(device_id, pattern, timeout=5):
                await asyncio.sleep(1)
                return True
        return False

    async def _check_already_signed(self, device_id: str) -> bool:
        """检查是否已签到
        
        Args:
            device_id: 设备 ID
            
        Returns:
            是否已签到
        """
        for pattern in self.SIGNED_PATTERNS:
            location = await self.screen_capture.find_text_location(device_id, pattern)
            if location:
                return True
        return False
    
    async def daily_sign_in(self, device_id: str, phone: str = "unknown", password: str = None, log_callback=None, profile_data: Optional[Dict] = None) -> SignInResult:
        """执行每日签到
        
        Args:
            device_id: 设备 ID
            phone: 手机号（用于截图文件命名）
            password: 密码（如果需要重新登录）
            log_callback: 日志回调函数（可选）
            profile_data: 个人信息数据（可选，如果提供则跳过获取个人信息步骤）
            
        Returns:
            签到结果（包含奖励金额和截图路径）
        """
        def log(msg):
            if log_callback:
                log_callback(msg)
            else:
                print(msg)
        
        try:
            # 定义登录回调函数
            async def login_callback(dev_id, ph, pwd):
                return await self.auto_login.login(dev_id, ph, pwd)
            
            log("开始执行签到流程...")
            
            # 使用 DailyCheckin 模块执行签到
            result = await self.daily_checkin.do_checkin(device_id, phone, password, login_callback, log_callback=log, profile_data=profile_data)
            
            # 如果需要重新登录但没有密码，返回失败
            if result.get('need_relogin', False) and not password:
                log("登录已失效，需要重新登录")
                return SignInResult(
                    success=False,
                    error_message="登录已失效，需要重新登录"
                )
            
            # 注意：转账检查已移至 GUI 层，在脚本开始时检查开关
            # 这里不再调用 _check_and_auto_transfer
            
            # 转换为 SignInResult，并添加奖励信息
            if result['success']:
                # 检查是否是"今日已签到"的情况
                if result.get('already_checked', False):
                    log("✓ 今日已签到（签到次数已用完）")
                    sign_in_result = SignInResult(
                        success=True,
                        already_signed=True
                    )
                    sign_in_result.total_times = result.get('total_times')  # 添加总次数
                    sign_in_result.reward_amount = 0.0  # 已签到，奖励为0
                    return sign_in_result
                else:
                    log(f"✓ 签到成功，奖励: {result.get('reward_amount', 0.0):.2f} 元")
                    sign_in_result = SignInResult(
                        success=True,
                        already_signed=False
                    )
                    # 添加额外信息
                    sign_in_result.reward_amount = result.get('reward_amount', 0.0)
                    sign_in_result.total_times = result.get('total_times')  # 添加总次数
                    sign_in_result.screenshot_path = result.get('screenshot_path')
                    sign_in_result.ocr_texts = result.get('ocr_texts', [])
                    return sign_in_result
            else:
                log(f"✗ 签到失败: {result.get('message', '未知错误')}")
                return SignInResult(
                    success=False,
                    error_message=result.get('message', '签到失败'),
                    error_type=result.get('error_type')  # 传递错误类型
                )
        except Exception as e:
            log(f"✗ 签到异常: {e}")
            from .models.error_types import ErrorType
            return SignInResult(
                success=False, 
                error_message=str(e),
                error_type=ErrorType.CHECKIN_EXCEPTION  # 签到异常
            )
    
    async def _check_and_auto_transfer(self, device_id: str, phone: str, log_callback=None):
        """检查余额并自动转账
        
        注意：此方法不检查 enabled 开关，调用者应该在调用前检查
        
        Args:
            device_id: 设备ID
            phone: 手机号（用于日志显示）
            log_callback: 日志回调函数
        """
        def log(msg):
            if log_callback:
                log_callback(msg)
            else:
                print(msg)
        
        try:
            # 导入转账配置和收款人选择器
            from .transfer_config import get_transfer_config
            from .recipient_selector import RecipientSelector
            
            transfer_config = get_transfer_config()
            
            # 创建收款人选择器（使用轮询策略）
            selector = RecipientSelector(strategy="rotation")
            
            log("  [自动转账] 开始检查余额...")
            
            # 先导航到首页
            log("  [自动转账] 返回首页...")
            from .page_detector_hybrid import PageState
            success = await self.guard.ensure_page_state(
                device_id,
                PageState.HOME,
                self.navigator.navigate_to_home,
                "自动转账前返回首页"
            )
            
            if not success:
                log("  [自动转账] ❌ 无法返回首页")
                return
            
            await asyncio.sleep(1)
            
            # 导航到个人页（使用统一的广告处理方法）
            log("  [自动转账] 进入个人页...")
            success = await self._navigate_to_profile_with_ad_handling(device_id, log)
            
            if not success:
                log("  [自动转账] ❌ 无法进入个人页")
                return
            
            # 获取完整的个人信息（包括用户ID）
            log("  [自动转账] 读取个人信息...")
            # 使用并行处理方法提升性能
            profile_data = await self.profile_reader.get_full_profile_parallel(device_id)
            if not profile_data:
                log("  [自动转账] ❌ 无法读取个人信息")
                return
            
            balance = profile_data.get('balance')
            user_id = profile_data.get('user_id')
            
            if balance is None:
                log("  [自动转账] ❌ 无法读取余额")
                return
            
            if not user_id:
                log("  [自动转账] ⚠️ 无法读取用户ID，使用手机号作为标识")
                user_id = phone
            
            log(f"  [自动转账] 当前余额: {balance:.2f} 元")
            log(f"  [自动转账] 用户ID: {user_id}")
            
            # 尝试获取管理员信息
            try:
                from .user_manager import UserManager
                manager = UserManager()
                user = manager.get_account_user(phone)
                if user:
                    log(f"  [自动转账] 管理员: {user.user_name} ({user.user_id})")
                    # 显示收款人列表
                    if user.transfer_recipients:
                        log(f"  [自动转账] 收款人列表: {', '.join(user.transfer_recipients)}")
            except Exception as e:
                log(f"  [自动转账] 无法获取管理员信息: {e}")
            
            # 判断是否需要转账（使用用户ID而不是手机号）
            if not transfer_config.should_transfer(user_id, balance, current_level=0):
                log(f"  [自动转账] 余额未达到转账条件")
                log(f"    - 起步金额: {transfer_config.min_transfer_amount:.2f} 元")
                log(f"    - 保留余额: {transfer_config.min_balance:.2f} 元")
                log(f"    - 需要余额 >= {transfer_config.min_transfer_amount + transfer_config.min_balance:.2f} 元")
                return
            
            # 获取收款人（使用增强版方法，支持用户管理和轮询选择）
            recipient = transfer_config.get_transfer_recipient_enhanced(
                phone=phone,
                user_id=user_id,
                current_level=0,
                selector=selector
            )
            
            if not recipient:
                log("  [自动转账] ❌ 未配置收款人")
                return
            
            # 显示选中的收款人和选择策略
            log(f"  [自动转账] ✓ 选中收款人: {recipient}")
            log(f"  [自动转账] 选择策略: {selector.strategy}")
            if selector.strategy == "rotation":
                # 显示轮询索引（下一次会使用的索引）
                try:
                    from .user_manager import UserManager
                    manager = UserManager()
                    user = manager.get_account_user(phone)
                    if user:
                        rotation_index = selector.get_rotation_index(user.user_id)
                        log(f"  [自动转账] 轮询索引: {rotation_index}")
                except:
                    pass
            
            log(f"  [自动转账] ✓ 满足转账条件，准备转账...")
            
            # 执行转账（使用重试机制）
            log(f"  [自动转账] 开始执行转账（支持重试）...")
            from .transfer_retry import get_transfer_retry
            retry_manager = get_transfer_retry(max_retries=3, retry_delay=5.0)
            
            transfer_result = await retry_manager.transfer_with_retry(
                transfer_func=self.balance_transfer.transfer_balance,
                device_id=device_id,
                recipient_id=recipient,
                log_callback=log
            )
            
            # 保存转账历史记录
            try:
                from .transfer_history import get_transfer_history
                transfer_history = get_transfer_history()
                
                # 获取收款人姓名
                recipient_name = recipient
                try:
                    from .user_manager import UserManager
                    manager = UserManager()
                    recipient_user = manager.get_account_user(recipient)
                    if recipient_user:
                        recipient_name = recipient_user.user_name
                except:
                    pass
                
                # 获取发送人姓名
                sender_name = phone
                owner_name = ""
                try:
                    from .user_manager import UserManager
                    manager = UserManager()
                    sender_user = manager.get_account_user(phone)
                    if sender_user:
                        sender_name = sender_user.user_name
                        owner_name = sender_user.user_id
                except:
                    pass
                
                # 保存记录
                transfer_history.save_transfer_record(
                    sender_phone=phone,
                    sender_user_id=user_id,
                    sender_name=sender_name,
                    recipient_phone=recipient,
                    recipient_name=recipient_name,
                    amount=transfer_result.get('amount', 0.0),
                    strategy=selector.strategy,
                    success=transfer_result['success'],
                    error_message=transfer_result.get('message', '') if not transfer_result['success'] else "",
                    owner=owner_name
                )
            except Exception as e:
                log(f"  [自动转账] ⚠️ 保存转账历史失败: {e}")
            
            if transfer_result['success']:
                log(f"  [自动转账] ✓ 转账成功")
                log(f"    - 收款人: {recipient}")
                if transfer_result.get('amount'):
                    log(f"    - 金额: {transfer_result['amount']:.2f} 元")
                log(f"    - 选择策略: {selector.strategy}")
            else:
                log(f"  [自动转账] ❌ 转账失败: {transfer_result.get('message', '未知错误')}")
                
        except Exception as e:
            import traceback
            log(f"  [自动转账] ❌ 异常: {e}")
            log(f"  [自动转账] 详细错误: {traceback.format_exc()}")
    
    async def navigate_to_draw(self, device_id: str) -> bool:
        """导航到抽奖页面
        
        Args:
            device_id: 设备 ID
            
        Returns:
            是否成功导航
        """
        for pattern in self.DRAW_PATTERNS:
            if await self.ui_automation.click_by_text(device_id, pattern, timeout=5):
                await asyncio.sleep(1)
                return True
        return False
    
    async def _get_remaining_draws(self, device_id: str) -> Optional[int]:
        """获取剩余抽奖次数（使用OCR增强器）
        
        Args:
            device_id: 设备 ID
            
        Returns:
            剩余次数，识别失败返回 None
        """
        try:
            # 获取整个屏幕的截图
            screenshot_data = await self.adb.screencap(device_id)
            if not screenshot_data:
                self._silent_log.info(f" 获取截图失败")
                return None
            
            # 使用PIL打开图片
            try:
                from PIL import Image
                from io import BytesIO
                image = Image.open(BytesIO(screenshot_data))
            except ImportError:
                self._silent_log.info(f" PIL未安装")
                return None
            
            # 使用OCR增强器识别数字
            if self._ocr_enhancer:
                result = await self._ocr_enhancer.recognize_number(
                    image,
                    pattern=r'剩余[：:\s]*(\d+)|(\d+)次|今日剩余[：:\s]*(\d+)'
                )
                
                if result.success:
                    # 提取数字
                    import re
                    numbers = re.findall(r'\d+', result.text)
                    if numbers:
                        count = int(numbers[0])
                        self._silent_log.info(f" ✓ 识别到剩余次数: {count}")
                        return count
            
            # 如果增强器不可用或识别失败，使用传统方法
            self._silent_log.info(f" 增强识别失败，使用传统方法...")
            try:
                from rapidocr import RapidOCR
                ocr = RapidOCR()
                result = ocr(image)
                
                if not result or not result.txts:
                    self._silent_log.info(f" OCR未识别到文字")
                    return None
                
                # 查找剩余次数文字
                import re
                for text in result.txts:
                    text_str = str(text)
                    
                    # 匹配格式: "剩余3次"、"剩余：3"、"剩余: 3"
                    match = re.search(r'剩余[：:\s]*(\d+)', text_str)
                    if match:
                        count = int(match.group(1))
                        self._silent_log.info(f" 识别到剩余次数: {count}")
                        return count
                    
                    # 匹配格式: "3次机会"、"3次"
                    match = re.search(r'(\d+)次', text_str)
                    if match:
                        count = int(match.group(1))
                        self._silent_log.info(f" 识别到次数: {count}")
                        return count
                    
                    # 匹配格式: "今日剩余: 3"
                    match = re.search(r'今日剩余[：:\s]*(\d+)', text_str)
                    if match:
                        count = int(match.group(1))
                        self._silent_log.info(f" 识别到今日剩余: {count}")
                        return count
                
                self._silent_log.info(f" 未找到次数信息")
                return None
                
            except ImportError:
                self._silent_log.info(f" RapidOCR未安装")
                return None
                
        except Exception as e:
            self._silent_log.info(f" 获取剩余次数失败: {e}")
            return None
            return None
    
    async def _check_no_draw_chances(self, device_id: str) -> bool:
        """检查是否没有抽奖次数（严格判断）
        
        Args:
            device_id: 设备 ID
            
        Returns:
            是否没有抽奖次数
        """
        # 方法1: 检查"次数已用完"等关键词
        no_chance_keywords = ["次数已用完", "明日再来", "抽奖次数不足", "今日次数已用完", "今日已用完"]
        
        for pattern in no_chance_keywords:
            location = await self.screen_capture.find_text_location(device_id, pattern)
            if location:
                self._silent_log.info(f" 检测到: {pattern}")
                return True
        
        # 方法2: 检查剩余次数是否为0
        remaining = await self._get_remaining_draws(device_id)
        if remaining is not None and remaining == 0:
            self._silent_log.info(f" 剩余次数: 0")
            return True
        
        self._silent_log.info(f" 仍有抽奖次数（剩余: {remaining if remaining is not None else '未知'}）")
        return False
    
    async def _extract_lottery_amount_from_region(self, device_id: str) -> Optional[float]:
        """从金额显示区域提取抽奖金额（使用OCR增强器）
        
        Args:
            device_id: 设备 ID
            
        Returns:
            抽奖金额，识别失败返回 None
        """
        try:
            # 定义金额显示区域（弹窗中央偏上）
            AMOUNT_REGION = (120, 300, 300, 200)
            
            # 获取整个屏幕的截图
            screenshot_data = await self.adb.screencap(device_id)
            if not screenshot_data:
                return None
            
            # 使用PIL打开图片
            try:
                from PIL import Image
                from io import BytesIO
                image = Image.open(BytesIO(screenshot_data))
            except ImportError:
                return None
            
            # 使用OCR增强器识别金额（多策略）
            if self._ocr_enhancer:
                # 方法1: 裁剪区域后识别
                from .image_processor import ImageProcessor
                cropped = ImageProcessor.crop_region(image, *AMOUNT_REGION)
                
                amount = await self._ocr_enhancer.recognize_amount(
                    cropped,
                    min_value=0.01,
                    max_value=100.0
                )
                
                if amount and amount > 0:
                    self._silent_log.info(f" ✓ 区域识别到金额: {amount:.2f} 元")
                    return amount
                
                # 方法2: 如果区域识别失败，尝试全屏
                self._silent_log.info(f" 区域识别失败，尝试全屏...")
                amount = await self._ocr_enhancer.recognize_amount(
                    image,
                    min_value=0.01,
                    max_value=100.0
                )
                
                if amount and amount > 0:
                    self._silent_log.info(f" ✓ 全屏识别到金额: {amount:.2f} 元")
                    return amount
            
            # 如果增强器不可用或识别失败，使用传统方法
            self._silent_log.info(f" 增强识别失败，使用传统方法...")
            x, y, w, h = AMOUNT_REGION
            region = image.crop((x, y, x + w, y + h))
            
            # 使用ImageProcessor增强文字区域
            try:
                from .image_processor import ImageProcessor
                enhanced = ImageProcessor.enhance_text_region(region)
            except Exception:
                enhanced = region
            
            # 使用RapidOCR识别
            try:
                from rapidocr import RapidOCR
                ocr = RapidOCR()
                result = ocr(enhanced)
                
                if not result or not result.txts:
                    return None
                
                # 正则表达式提取金额
                import re
                for text in result.txts:
                    text_str = str(text)
                    
                    # 匹配格式: "0.5元"、"1.50元"
                    match = re.search(r'(\d+\.?\d*)元', text_str)
                    if match:
                        amount = float(match.group(1))
                        if 0.01 <= amount <= 100:
                            return amount
                    
                    # 匹配格式: "¥1.50"、"￥1.50"
                    match = re.search(r'[¥￥](\d+\.?\d*)', text_str)
                    if match:
                        amount = float(match.group(1))
                        if 0.01 <= amount <= 100:
                            return amount
                    
                    # 匹配格式: "1.50"（纯数字）
                    match = re.search(r'^(\d+\.\d{2})$', text_str)
                    if match:
                        amount = float(match.group(1))
                        if 0.01 <= amount <= 100:
                            return amount
                
                return None
                
            except ImportError:
                return None
                
        except Exception as e:
            self._silent_log.info(f" 识别失败: {e}")
            return None
    
    async def _save_lottery_screenshot(self, device_id: str, phone: str, 
                                      draw_index: int, amount: Optional[float]) -> Optional[str]:
        """保存抽奖截图
        
        Args:
            device_id: 设备 ID
            phone: 手机号
            draw_index: 抽奖序号
            amount: 抽奖金额
            
        Returns:
            截图路径，保存失败返回 None
        """
        try:
            import os
            from datetime import datetime
            
            # 创建lottery_screenshots目录
            screenshot_dir = "lottery_screenshots"
            if not os.path.exists(screenshot_dir):
                os.makedirs(screenshot_dir)
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            amount_str = f"{amount:.2f}" if amount is not None else "unknown"
            filename = f"lottery_{phone}_{draw_index}_{amount_str}_{timestamp}.png"
            filepath = os.path.join(screenshot_dir, filename)
            
            # 获取截图
            screenshot_data = await self.adb.screencap(device_id)
            if not screenshot_data:
                return None
            
            # 保存截图
            with open(filepath, 'wb') as f:
                f.write(screenshot_data)
            
            print(f"  [截图] 已保存: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"  [截图] 保存失败: {e}")
            return None
    
    async def _do_single_draw(self, device_id: str, draw_index: int = 1, phone: str = "unknown") -> Optional[float]:
        """执行单次抽奖（带页面状态守卫）
        
        Args:
            device_id: 设备 ID
            draw_index: 抽奖序号（用于日志和截图）
            phone: 手机号（用于截图文件名）
            
        Returns:
            抽奖金额，识别失败返回 None（但不影响继续抽奖）
        """
        print(f"\n  [抽奖 {draw_index}] 开始...")
        
        from .retry_helper import retry_on_false
        
        # 1. 使用守卫执行点击抽奖按钮
        clicked = False
        for pattern in self.DRAW_PATTERNS:
            async def click_draw_button():
                return await self.ui_automation.click_by_text(device_id, pattern, timeout=3)
            
            # 使用守卫执行点击操作
            result = await self.guard.execute_with_guard(
                device_id,
                click_draw_button,
                None,  # 不验证特定页面状态
                f"点击抽奖按钮: {pattern}"
            )
            
            if result:
                clicked = True
                self._silent_log.info(f" ✓ 点击抽奖按钮: {pattern}")
                break
        
        if not clicked:
            self._silent_log.info(f" ✗ 未找到抽奖按钮")
            return None
        
        # 2. 等待抽奖动画（增加到4秒）
        self._silent_log.info(f" 等待抽奖结果...")
        await asyncio.sleep(4)
        
        # 3. 尝试提取抽奖金额（先区域识别，失败则全屏识别）
        amount = None
        
        # 方法1：区域识别（优先）
        self._silent_log.info(f" 尝试区域识别金额...")
        amount = await self._extract_lottery_amount_from_region(device_id)
        
        # 方法2：全屏识别（备用）
        if amount is None:
            self._silent_log.info(f" 区域识别失败，尝试全屏识别...")
            amount = await self.screen_capture.extract_balance(device_id)
        
        # 金额识别结果
        if amount is not None and amount > 0:
            self._silent_log.info(f" ✓ 识别到金额: {amount:.2f} 元")
        else:
            self._silent_log.info(f" ⚠️ 未能识别抽奖金额（继续执行）")
            # 注意：金额识别失败不返回None，继续执行
        
        # 4. 保存抽奖截图
        await self._save_lottery_screenshot(device_id, phone, draw_index, amount)
        
        # 5. 使用守卫执行关闭弹窗操作
        close_keywords = ["确定", "关闭", "知道了", "我知道了", "继续抽奖", "再抽一次"]
        closed = False
        
        for keyword in close_keywords:
            async def close_popup():
                return await self.ui_automation.click_by_text(device_id, keyword, timeout=2)
            
            result = await self.guard.execute_with_guard(
                device_id,
                close_popup,
                None,
                f"关闭弹窗: {keyword}"
            )
            
            if result:
                self._silent_log.info(f" ✓ 关闭弹窗: {keyword}")
                closed = True
                break
        
        # 如果关闭失败，使用备用方案（点击记录的关闭按钮位置）
        if not closed:
            self._silent_log.info(f" ⚠️ 未找到关闭按钮，点击记录位置...")
            
            async def click_close_position():
                await self.adb.tap(device_id, 283, 866)  # 用户记录的实际关闭按钮位置
            
            await self.guard.execute_with_guard(
                device_id,
                click_close_position,
                None,
                "点击关闭按钮位置"
            )
        
        await asyncio.sleep(1.5)
        
        self._silent_log.info(f" 完成\n")
        return amount

    async def lucky_draw(self, device_id: str, max_draws: int = 10, phone: str = "unknown") -> DrawResult:
        """执行抽奖（带页面状态守卫）
        
        Args:
            device_id: 设备 ID
            max_draws: 最大抽奖次数限制（防止无限循环）
            phone: 手机号（用于截图文件名）
            
        Returns:
            抽奖结果
        """
        print(f"\n{'='*60}")
        print(f"开始抽奖流程 - 手机号: {phone}")
        print(f"{'='*60}\n")
        
        try:
            # 1. 使用守卫导航到抽奖页面
            print(f"[步骤1] 导航到抽奖页面...")
            
            # 定义导航函数
            async def navigate_to_lottery_page():
                return await self.navigator.navigate_to_lottery(device_id)
            
            # 使用守卫确保到达抽奖页面
            success = await self.guard.safe_navigate(
                device_id,
                navigate_to_lottery_page,
                "抽奖页面",
                "导航到抽奖页面"
            )
            
            if not success:
                print(f"✗ 导航到抽奖页面失败\n")
                return DrawResult(
                    success=False,
                    draw_count=0,
                    total_amount=0.0,
                    amounts=[],
                    error_message="导航到抽奖页面失败"
                )
            print(f"✓ 成功到达抽奖页面\n")
            
            # 2. 获取初始剩余次数
            print(f"[步骤2] 检查剩余抽奖次数...")
            initial_remaining = await self._get_remaining_draws(device_id)
            if initial_remaining is not None:
                print(f"✓ 初始剩余次数: {initial_remaining}\n")
            else:
                print(f"⚠️ 无法读取剩余次数，将循环抽奖直到检测到次数用完\n")
            
            # 3. 检查是否有抽奖次数
            if await self._check_no_draw_chances(device_id):
                print(f"✗ 抽奖次数已用完\n")
                return DrawResult(
                    success=True,
                    draw_count=0,
                    total_amount=0.0,
                    amounts=[],
                    error_message="抽奖次数已用完"
                )
            
            # 4. 循环抽奖
            print(f"[步骤3] 开始循环抽奖...")
            print(f"{'='*60}\n")
            
            amounts = []
            success_count = 0
            failed_count = 0
            consecutive_failures = 0
            
            for i in range(max_draws):
                draw_index = i + 1
                
                # 每次抽奖前验证仍在抽奖页面
                self._silent_log.info(f" 验证页面状态...")
                current_state = await self.guard.get_current_page_state(device_id, f"抽奖 {draw_index} 前")
                
                # 如果不在抽奖页面,尝试处理
                if "抽奖" not in str(current_state):
                    self._silent_log.info(f" ⚠️ 不在抽奖页面: {current_state.value}")
                    handled = await self.guard._handle_unexpected_page(device_id, current_state, f"抽奖 {draw_index} 前")
                    if not handled:
                        self._silent_log.info(f" ❌ 无法处理异常页面,停止抽奖")
                        break
                
                # 执行单次抽奖
                amount = await self._do_single_draw(device_id, draw_index, phone)
                
                if amount is not None and amount > 0:
                    amounts.append(amount)
                    success_count += 1
                    consecutive_failures = 0
                    print(f"  ✓ 第 {draw_index} 次抽奖成功: {amount:.2f} 元")
                else:
                    failed_count += 1
                    consecutive_failures += 1
                    print(f"  ⚠️ 第 {draw_index} 次抽奖失败（金额识别失败或按钮未找到）")
                
                # 连续失败5次警告但继续
                if consecutive_failures >= 5:
                    print(f"\n  ⚠️ 警告：连续失败 {consecutive_failures} 次，可能出现问题")
                    print(f"  继续尝试...\n")
                
                # 等待一下，避免操作过快
                await asyncio.sleep(1.5)
                
                # 检查是否还有次数
                if await self._check_no_draw_chances(device_id):
                    print(f"\n  ✓ 检测到抽奖次数已用完，停止抽奖")
                    break
                
                # 如果达到最大次数限制，停止
                if draw_index >= max_draws:
                    print(f"\n  ⚠️ 已达到最大抽奖次数限制 ({max_draws})，停止抽奖")
                    break
            
            # 5. 返回结果
            total = sum(amounts)
            avg = total / len(amounts) if amounts else 0
            
            print(f"\n{'='*60}")
            print(f"抽奖流程完成")
            print(f"{'='*60}")
            print(f"  成功次数: {success_count}")
            print(f"  失败次数: {failed_count}")
            print(f"  总金额: {total:.2f} 元")
            print(f"  平均金额: {avg:.2f} 元")
            if amounts:
                print(f"  金额明细: {', '.join([f'{a:.2f}' for a in amounts])} 元")
            print(f"{'='*60}\n")
            
            return DrawResult(
                success=True,
                draw_count=len(amounts),
                total_amount=total,
                amounts=amounts
            )
            
        except Exception as e:
            print(f"\n✗ 抽奖过程出错: {str(e)}\n")
            return DrawResult(
                success=False,
                error_message=f"抽奖过程出错: {str(e)}"
            )
    
    async def run_full_workflow(self, device_id: str, account: Account, skip_login: bool = False) -> AccountResult:
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
            
        Returns:
            账号处理结果（包含完整数据）
        """
        import time
        
        # 记录工作流开始时间
        workflow_start = time.time()
        
        # 定义日志输出函数（优先使用回调，否则使用print）
        def log(msg):
            if self._log_callback:
                self._log_callback(msg)
            else:
                print(msg)
        
        log(f"\n{'='*60}")
        log(f"[时间记录] 工作流开始 - {time.strftime('%H:%M:%S')}")
        log(f"[时间记录] 账号: {account.phone}")
        log(f"{'='*60}\n")
        
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
            step1_start = time.time()
            log(f"{'='*60}")
            log(f"[时间记录] 步骤1: 登录账号 - {account.phone}")
            log(f"{'='*60}")
            
            # 如果 skip_login=True，说明缓存登录已验证，当前已在个人页
            if skip_login:
                log(f"✓ 缓存登录已验证，当前已在个人页，直接获取个人信息")
                step1_time = time.time() - step1_start
                log(f"[时间记录] 步骤1完成 - 耗时: {step1_time:.3f}秒（跳过登录）")
                log("")
                # 缓存登录不需要处理登录和积分页，直接跳到获取个人资料
            else:
                # 执行正常登录流程
                login_start = time.time()
                login_result = await self.auto_login.login(
                    device_id, account.phone, account.password
                )
                login_time = time.time() - login_start
                log(f"[时间记录] 登录操作耗时: {login_time:.3f}秒")
                
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
                    log(f"✗ 登录失败: {login_result.error_message}")
                    return result
                
                log(f"✓ 登录成功")
                # 登录后会跳转到积分页，需要返回到个人页
                log(f"登录后处理积分页跳转...")
                await asyncio.sleep(2)
                
                # 检测当前页面 - 使用整合检测器（GPU加速深度学习）
                from .page_detector import PageState
                page_result = await self.integrated_detector.detect_page(
                    device_id, use_cache=False, detect_elements=False
                )
                
                if page_result and page_result.state == PageState.POINTS_PAGE:
                    log(f"检测到积分页，需要按2次返回键到个人页...")
                    
                    # 第1次返回键
                    await self.adb.press_back(device_id)
                    await asyncio.sleep(1)
                    
                    # 第2次返回键
                    await self.adb.press_back(device_id)
                    await interruptible_sleep(3)  # 增加等待时间到3秒，让页面充分加载
                    
                    # 清理缓存（页面已改变）
                    self.hybrid_detector.clear_cache()
                    
                    # 按返回键后检测页面状态（最多重试3次）- 使用整合检测器（GPU加速）
                    for retry in range(3):
                        # 使用整合检测器进行快速检测
                        page_result = await self.integrated_detector.detect_page(
                            device_id, use_cache=False, detect_elements=False
                        )
                        
                        if page_result:
                            log(f"检测结果: {page_result.state.value} - {page_result.details}")
                            
                            if page_result.state == PageState.PROFILE_LOGGED:
                                log(f"✓ 已返回到个人页")
                                break
                            elif page_result.state == PageState.POINTS_PAGE:
                                # 仍然在积分页，再按一次返回键
                                log(f"⚠️ 仍在积分页，再按一次返回键...")
                                await self.adb.press_back(device_id)
                                await interruptible_sleep(3)
                                self.hybrid_detector.clear_cache()
                                # 不break，继续重试检测
                            elif page_result.state == PageState.LAUNCHER:
                                log(f"❌ 检测到桌面，应用已退出！尝试重启应用...")
                                # 强制停止应用
                                await self.adb.stop_app(device_id, "com.xmwl.shop")
                                await asyncio.sleep(1)
                                # 重新启动应用
                                await self.adb.start_app(device_id, "com.xmwl.shop")
                                await interruptible_sleep(5)
                                log(f"✓ 应用已重新启动")
                                
                                # 清理缓存（应用已重启）
                                self.hybrid_detector.clear_cache()
                                
                                # 重启后再次检测页面状态 - 使用整合检测器（GPU加速）
                                log(f"检测应用启动后的页面状态...")
                                page_result = await self.integrated_detector.detect_page(
                                    device_id, use_cache=False, detect_elements=False
                                )
                                if page_result:
                                    log(f"当前页面: {page_result.state.value}")
                                else:
                                    log(f"⚠️ 无法检测页面状态")
                                break
                            else:
                                # 其他状态，可能是个人页但模板没匹配上，用OCR再确认
                                if "个人" in page_result.details or "我的" in page_result.details or "余额" in page_result.details:
                                    log(f"✓ OCR确认已在个人页（{page_result.details}）")
                                    break
                                
                                if retry < 2:
                                    log(f"⚠️ 返回后页面状态: {page_result.state.value}，等待后重试...")
                                    await interruptible_sleep(3)  # 增加等待时间到3秒
                                else:
                                    log(f"⚠️ 返回后页面状态: {page_result.state.value}")
                        else:
                            if retry < 2:
                                log(f"⚠️ 无法检测页面状态，等待后重试...")
                                await asyncio.sleep(2)  # 等待更长时间
                            else:
                                log(f"⚠️ 无法检测页面状态")
                    else:
                        log(f"⚠️ 无法检测返回后的页面状态")
                else:
                    log(f"当前页面: {page_result.state.value if page_result else 'unknown'}")
                    await asyncio.sleep(2)
            
            # ==================== 步骤2: 获取初始个人资料 ====================
            log(f"{'='*60}")
            log(f"步骤2: 获取初始个人资料")
            log(f"{'='*60}")
            
            profile_success = False
            profile_data = None
            
            # 尝试最多3次获取个人资料
            for attempt in range(3):
                try:
                    if attempt > 0:
                        log(f"\n[尝试 {attempt + 1}/3] 重新获取个人资料...")
                    
                    # 如果是缓存登录，跳过导航（已经在个人页）
                    if skip_login:
                        cache_check_start = time.time()
                        log(f"[缓存登录] 验证当前页面状态...")
                        nav_success = True
                        
                        # 立即检测页面状态（不等待）- 使用整合检测器（GPU加速）
                        detect_start = time.time()
                        from .page_detector import PageState
                        page_result = await self.integrated_detector.detect_page(
                            device_id, use_cache=True, detect_elements=False
                        )
                        detect_time = time.time() - detect_start
                        log(f"[时间记录] 页面检测耗时: {detect_time:.3f}秒")
                        
                        if not page_result or not page_result.state:
                            log(f"  ⚠️ 无法检测当前页面状态")
                            if attempt < 2:
                                await asyncio.sleep(2)
                                continue
                            else:
                                result.error_message = "无法确认当前页面状态"
                                return result
                        
                        log(f"  当前页面: {page_result.state.value}（置信度{page_result.confidence:.2%}）")
                        
                        # 确认在个人页（已登录）
                        if page_result.state != PageState.PROFILE_LOGGED:
                            log(f"  ⚠️ 当前不在个人页（已登录），尝试导航...")
                            nav_start = time.time()
                            nav_success = await self.navigator.navigate_to_profile(device_id)
                            nav_time = time.time() - nav_start
                            log(f"[时间记录] 导航耗时: {nav_time:.3f}秒")
                            if not nav_success:
                                if attempt < 2:
                                    await asyncio.sleep(2)
                                    continue
                                else:
                                    result.error_message = "无法导航到个人页"
                                    return result
                        else:
                            log(f"  ✓ 确认在个人页（已登录）")
                            log(f"  ✓ 页面已就绪，直接获取个人资料")
                        
                        cache_check_time = time.time() - cache_check_start
                        log(f"[时间记录] 缓存登录验证总耗时: {cache_check_time:.3f}秒")
                    else:
                        # 导航到个人资料页面
                        # 注意：导航可能返回True但页面状态不确定，通过获取资料来验证
                        nav_start = time.time()
                        nav_success = await self.navigator.navigate_to_profile(device_id)
                        nav_time = time.time() - nav_start
                        log(f"[时间记录] 导航耗时: {nav_time:.3f}秒")
                        
                        if not nav_success:
                            log(f"⚠️ 导航到个人资料页面失败")
                            
                            # 检查当前页面状态 - 使用整合检测器（GPU加速）
                            from .page_detector import PageState
                            page_result = await self.integrated_detector.detect_page(
                                device_id, use_cache=True, detect_elements=False
                            )
                            
                            # 检查返回值是否有效
                            if not page_result or not page_result.state:
                                log(f"  ⚠️ 无法检测当前页面状态")
                                if attempt < 2:
                                    await asyncio.sleep(2)
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
                                await asyncio.sleep(2)
                                
                                # 重新导航
                                nav_success = await self.navigator.navigate_to_profile(device_id)
                                if not nav_success:
                                    if attempt < 2:
                                        log(f"  ⚠️ 导航仍然失败，等待2秒后重试...")
                                        await asyncio.sleep(2)
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
                    profile_read_start = time.time()
                    log(f"[时间记录] 开始获取个人资料...")
                    account_str = f"{account.phone}----{account.password}"
                    profile_data = await self.profile_reader.get_full_profile_with_retry(device_id, account=account_str)
                    profile_read_time = time.time() - profile_read_start
                    log(f"[时间记录] 获取个人资料耗时: {profile_read_time:.3f}秒")
                    
                    # 检查是否成功获取数据（必须获取到余额、昵称和用户ID）
                    has_balance = profile_data and profile_data.get('balance') is not None
                    has_nickname = profile_data and profile_data.get('nickname') is not None
                    has_user_id = profile_data and profile_data.get('user_id') is not None
                    
                    # ===== 核心逻辑：基于获取资料的结果判断是否成功 =====
                    if has_balance and has_nickname and has_user_id:
                        log(f"✓ 成功获取个人资料数据")
                        profile_success = True
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
                            log(f"  等待2秒后重试...")
                            await asyncio.sleep(2)
                        
                except Exception as e:
                    log(f"⚠️ 获取个人资料出错: {str(e)}")
                    if attempt < 2:
                        log(f"  等待2秒后重试...")
                        await asyncio.sleep(2)
                    else:
                        from .models.error_types import ErrorType
                        result.error_type = ErrorType.CANNOT_READ_PROFILE
                        result.error_message = f"获取个人资料失败: {str(e)}"
                        log(f"✗ 无法获取个人资料，终止流程\n")
                        return result
            
            # 如果3次尝试后仍未成功
            if not profile_success or not profile_data:
                from .models.error_types import ErrorType
                result.error_type = ErrorType.CANNOT_READ_PROFILE
                result.error_message = "获取个人资料失败（3次尝试后）"
                log(f"✗ 无法获取个人资料，终止流程\n")
                return result
            
            # ==================== 步骤3: 存储初始数据到 result ====================
            result.nickname = profile_data.get('nickname')  # 直接使用OCR结果，不使用历史数据
            result.user_id = profile_data.get('user_id')  # 直接使用OCR结果，不使用历史数据
            # phone 已经在初始化时设置
            result.balance_before = profile_data.get('balance')
            result.points = profile_data.get('points')
            result.vouchers = profile_data.get('vouchers')
            result.coupons = profile_data.get('coupons')
            
            # 显示收集到的账户信息
            self._log_profile_data(result)
            
            # 调试信息：显示缓存状态
            log(f"\n[调试] 缓存状态检查:")
            log(f"  - enable_cache: {self.auto_login.enable_cache}")
            log(f"  - cache_manager 存在: {self.auto_login.cache_manager is not None}")
            log(f"  - user_id: {result.user_id}")
            
            # ==================== 步骤3.5: 保存登录缓存（包含用户ID）====================
            # 优化：异步保存缓存，不阻塞主流程
            if self.auto_login.enable_cache and self.auto_login.cache_manager:
                if result.user_id:
                    log(f"\n异步保存登录缓存（包含用户ID: {result.user_id}）...")
                else:
                    log(f"\n异步保存登录缓存（未获取到用户ID）...")
                
                # 创建异步任务，不等待完成
                async def save_cache_async():
                    try:
                        cache_saved = await self.auto_login.cache_manager.save_login_cache(
                            device_id, 
                            account.phone, 
                            user_id=result.user_id
                        )
                        if cache_saved:
                            log(f"[后台] ✓ 登录缓存已保存")
                        else:
                            log(f"[后台] ⚠️ 登录缓存保存失败")
                    except Exception as e:
                        log(f"[后台] ⚠️ 保存登录缓存时出错: {str(e)}")
                
                # 启动后台任务，不等待完成
                asyncio.create_task(save_cache_async())
                log(f"✓ 缓存保存任务已启动（后台执行）")
            else:
                if not self.auto_login.enable_cache:
                    log(f"\n⚠️ 缓存功能未启用，跳过缓存保存")
                elif not self.auto_login.cache_manager:
                    log(f"\n⚠️ 缓存管理器未初始化，跳过缓存保存")
            
            # 优化：移除不必要的1秒等待
            # await asyncio.sleep(1)  # 已移除
            
            # ==================== 优化：跳过重复获取个人资料 ====================
            # 步骤2已经获取了完整的个人资料，不需要在步骤4重复获取
            # 直接使用步骤2的数据，节省时间
            log(f"\n{'='*60}")
            log(f"[优化] 使用步骤2已获取的个人资料数据")
            log(f"{'='*60}")
            log(f"  用户ID: {result.user_id}")
            log(f"  昵称: {result.nickname}")
            log(f"  余额: {result.balance_before:.2f} 元" if result.balance_before else "  余额: 未获取")
            log(f"  积分: {result.points}")
            log(f"  抵扣券: {result.vouchers}")
            log(f"  优惠券: {result.coupons}")
            log("")
            
            # ==================== 步骤3: 执行签到（仅在成功获取资料后执行）====================
            if not profile_success:
                log(f"{'='*60}")
                log(f"跳过签到流程")
                log(f"{'='*60}")
                log(f"  原因: 未能获取个人资料数据\n")
                # 标记为成功但跳过签到
                result.success = True
                return result
            
            log(f"{'='*60}")
            log(f"步骤3: 执行签到")
            log(f"{'='*60}")
            
            try:
                # 使用步骤2已获取的个人资料数据
                updated_profile_data = {
                    'balance': result.balance_before,  # 使用步骤2获取的余额
                    'points': result.points,
                    'vouchers': result.vouchers,
                    'coupons': result.coupons,
                    'nickname': result.nickname,
                    'user_id': result.user_id
                }
                
                # 直接调用 do_checkin，它会自动处理导航和返回首页
                checkin_result = await self.daily_checkin.do_checkin(
                    device_id, 
                    phone=account.phone,
                    password=account.password,
                    login_callback=None,  # 已经登录，不需要回调
                    log_callback=None,
                    profile_data=updated_profile_data  # 传递最新获取的个人信息（步骤4的数据）
                )
                
                # 记录签到结果
                if checkin_result['success']:
                    result.checkin_reward = checkin_result.get('reward_amount', 0.0)
                    result.checkin_total_times = checkin_result.get('total_times')
                    log(f"✓ 签到成功")
                    log(f"  签到次数: {checkin_result.get('checkin_count', 0)}")
                    log(f"  总奖励: {result.checkin_reward:.2f} 元\n")
                elif checkin_result.get('already_checked'):
                    # 即使已签到，也要获取总次数
                    result.checkin_total_times = checkin_result.get('total_times')
                    log(f"✓ 今日已签到")
                    if result.checkin_total_times:
                        log(f"  总签到次数: {result.checkin_total_times}\n")
                    else:
                        log(f"  (未获取到总签到次数)\n")
                else:
                    # 签到失败，设置错误类型
                    from .models.error_types import ErrorType
                    result.error_type = ErrorType.CHECKIN_FAILED
                    result.error_message = checkin_result.get('message', '未知错误')
                    log(f"⚠️ 签到失败: {result.error_message}\n")
                
            except Exception as e:
                # 签到异常，设置错误类型
                from .models.error_types import ErrorType
                result.error_type = ErrorType.CHECKIN_EXCEPTION
                result.error_message = str(e)
                log(f"⚠️ 签到流程出错: {str(e)}")
                log(f"⚠️ 跳过签到，继续执行后续流程\n")
            
            # 优化：移除签到后的1秒等待，直接进入下一步
            # await asyncio.sleep(1)  # 已移除
            
            # ==================== 步骤7: 再次获取完整个人资料（最终数据）====================
            log(f"{'='*60}")
            log(f"步骤7: 获取最终个人资料")
            log(f"{'='*60}")
            
            try:
                # 使用统一的导航方法（高频扫描，自动处理广告）
                log(f"  导航到个人页...")
                nav_success = await self._navigate_to_profile_with_ad_handling(device_id, log)
                
                if not nav_success:
                    log(f"  ⚠️ 导航到个人资料页面失败，跳过最终数据获取\n")
                else:
                    # 直接获取个人资料（不再需要检测和处理广告）
                    try:
                        # 获取完整个人资料（带超时）
                        # 使用并行处理方法提升性能
                        account_str = f"{account.phone}----{account.password}"
                        profile_task = self.profile_reader.get_full_profile_parallel(device_id, account=account_str)
                        try:
                            final_profile = await asyncio.wait_for(profile_task, timeout=10.0)
                            
                            # 更新最终数据（确保类型正确）
                            balance = final_profile.get('balance')
                            if balance is not None:
                                # 确保余额是float类型
                                if isinstance(balance, str):
                                    try:
                                        result.balance_after = float(balance)
                                    except ValueError:
                                        result.balance_after = None
                                else:
                                    result.balance_after = balance
                            else:
                                result.balance_after = None
                            
                            # 更新其他可能变化的字段
                            if final_profile.get('points') is not None:
                                result.points = final_profile.get('points')
                            if final_profile.get('vouchers') is not None:
                                result.vouchers = final_profile.get('vouchers')
                            if final_profile.get('coupons') is not None:
                                result.coupons = final_profile.get('coupons')
                            
                            if result.balance_after is not None:
                                log(f"  ✓ 最终余额: {result.balance_after:.2f} 元")
                            else:
                                log(f"  ⚠️ 未能获取最终余额")
                            
                            if result.points is not None:
                                log(f"  ✓ 最终积分: {result.points} 积分")
                            
                            if result.vouchers is not None:
                                log(f"  ✓ 最终抵扣券: {result.vouchers} 张")
                            
                            if result.coupons is not None:
                                log(f"  ✓ 最终优惠券: {result.coupons} 张")
                            
                            log("")  # 空行
                        except asyncio.TimeoutError:
                            log(f"  ⚠️ 获取最终数据超时\n")
                    except Exception as e:
                        log(f"  ⚠️ 获取最终数据出错: {str(e)}\n")
            except Exception as e:
                log(f"⚠️ 获取最终数据流程出错: {str(e)}\n")
            
            # 显示执行总结
            self._log_summary(result)
            
            # ==================== 步骤7.5: 自动转账（每次处理账号时重新读取配置）====================
            log(f"{'='*60}")
            log(f"步骤7.5: 检查自动转账")
            log(f"{'='*60}")
            
            try:
                # 每次处理账号时重新读取转账配置
                from .transfer_config import get_transfer_config
                transfer_config = get_transfer_config()
                should_auto_transfer = transfer_config.enabled
                
                if should_auto_transfer:
                    log(f"✓ 自动转账功能已启用，开始检查转账条件...")
                    
                    # 检查必要的数据
                    if not result.user_id:
                        log(f"  ⚠️ 无法获取用户ID，跳过转账")
                    elif result.balance_after is None:
                        log(f"  ⚠️ 无法获取余额，跳过转账")
                    else:
                        # 判断是否需要转账（使用最新的配置）
                        account_level = transfer_config.get_account_level(result.user_id)
                        
                        if transfer_config.should_transfer(result.user_id, result.balance_after, current_level=0):
                            recipient_id = transfer_config.get_transfer_recipient(result.user_id, current_level=0)
                            if recipient_id:
                                log(f"  ✓ 满足转账条件，准备转账到 ID: {recipient_id}")
                                try:
                                    # 执行转账（传入转账前余额用于验证）
                                    transfer_result = await self.balance_transfer.transfer_balance(
                                        device_id,
                                        recipient_id,
                                        initial_balance=result.balance_after,
                                        log_callback=log
                                    )
                                    
                                    if transfer_result['success']:
                                        log(f"  ✓ 转账成功")
                                        if transfer_result.get('amount'):
                                            log(f"    - 转账金额: {transfer_result['amount']:.2f} 元")
                                            # 更新余额（转账后余额会减少）
                                            if result.balance_after is not None:
                                                result.balance_after -= transfer_result['amount']
                                                log(f"    - 转账后余额: {result.balance_after:.2f} 元")
                                            # 保存转账信息到result对象
                                            result.transfer_amount = transfer_result.get('amount', 0.0)
                                            result.transfer_recipient = recipient_id
                                            log(f"    - 已保存转账信息: {result.transfer_amount:.2f} 元 → {result.transfer_recipient}")
                                    else:
                                        # 转账失败，设置错误类型
                                        from .models.error_types import ErrorType
                                        result.error_type = ErrorType.TRANSFER_FAILED
                                        result.error_message = transfer_result.get('message', '未知错误')
                                        log(f"  ❌ 转账失败: {result.error_message}")
                                        # 转账失败时，设置收款人为"失败"
                                        result.transfer_amount = 0.0
                                        result.transfer_recipient = "失败"
                                except Exception as e:
                                    # 转账异常，设置错误类型
                                    from .models.error_types import ErrorType
                                    result.error_type = ErrorType.TRANSFER_FAILED
                                    result.error_message = str(e)
                                    log(f"  ❌ 转账异常: {e}")
                                    import traceback
                                    log(f"  详细错误: {traceback.format_exc()}")
                                    # 转账异常时，设置收款人为"失败"
                                    result.transfer_amount = 0.0
                                    result.transfer_recipient = "失败"
                            else:
                                log(f"  ⚠️ 未配置收款人ID")
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
                                log(f"  ℹ️ 余额未达到转账条件")
                                log(f"    - 当前余额: {result.balance_after:.2f} 元")
                                log(f"    - 起步金额: {transfer_config.min_transfer_amount:.2f} 元")
                                log(f"    - 保留余额: {transfer_config.min_balance:.2f} 元")
                                log(f"    - 需要余额 >= {transfer_config.min_transfer_amount + transfer_config.min_balance:.2f} 元")
                else:
                    log(f"  ℹ️ 自动转账功能未启用，跳过转账")
                
                log("")  # 空行
            except Exception as e:
                log(f"  ❌ 转账检查出错: {str(e)}\n")
            
            # ==================== 步骤8: 退出登录 ====================
            log(f"{'='*60}")
            log(f"步骤8: 退出登录")
            log(f"{'='*60}")
            
            try:
                await self.auto_login.logout(device_id)
                log(f"✓ 已退出登录\n")
            except Exception as e:
                log(f"⚠️ 退出登录失败: {str(e)}\n")
            
            result.success = True
            return result
            
        except Exception as e:
            result.error_message = str(e)
            log(f"\n✗ 工作流程出错: {str(e)}\n")
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

    async def handle_startup_flow_optimized(self, device_id: str, log_callback=None, stop_check=None, 
                                           package_name: str = "com.ry.xmsc", activity_name: str = None,
                                           max_retries: int = 3,
                                           stuck_timeout: int = 15, max_wait_time: int = 60,
                                           enable_debug: bool = True) -> bool:
        """处理应用启动流程（优化版）- 使用性能监控和智能等待
        
        优化点：
        1. 使用模板匹配优先进行页面类型检测（不强制use_ocr=True）
        2. 使用SmartWaiter替代固定等待，提前检测到状态变化
        3. 集成PerformanceMonitor监控性能
        4. 使用DetectionCache避免重复检测
        
        流程：启动页(等待) -> 用户协议弹窗(关闭) -> 广告页(等待消失) -> 首页弹窗(关闭) -> 主页
        如果白屏卡死，清理缓存并重启应用
        
        Args:
            device_id: 设备 ID
            log_callback: 日志回调函数
            stop_check: 停止检查函数，返回 True 表示需要停止
            package_name: 应用包名
            activity_name: Activity名称（可选）
            max_retries: 最大重试次数（默认3次）
            stuck_timeout: 白屏卡住检测时间（秒，默认15秒）
            max_wait_time: 最大等待时间（秒，默认60秒）
            enable_debug: 是否启用调试日志（默认True）
            
        Returns:
            是否成功
        """
        from .performance.performance_monitor import PerformanceMonitor
        from .performance.smart_waiter import SmartWaiter
        from .performance.detection_cache import DetectionCache
        from .page_detector import PageState
        
        def log(msg):
            if log_callback:
                log_callback(msg)
        
        def should_stop():
            if stop_check:
                return stop_check()
            return False
        
        # 创建性能监控器
        monitor = PerformanceMonitor("启动流程优化版")
        monitor.start()
        
        # 创建智能等待器和缓存
        waiter = SmartWaiter()
        cache = DetectionCache(ttl=0.5)
        
        # 初始化调试日志
        debug_logger = None
        if enable_debug:
            from .debug_logger import DebugLogger
            debug_logger = DebugLogger()
            debug_logger.log_step("开始应用启动流程（优化版）", f"包名: {package_name}")
            log(f"调试日志已启用，保存到: {debug_logger.session_dir}")
        
        # 复用已有的混合检测器（避免重复加载模板）
        hybrid_detector = self.hybrid_detector
        
        for retry in range(max_retries):
            if should_stop():
                log("用户请求停止")
                if debug_logger:
                    debug_logger.log_warning("用户请求停止")
                    debug_logger.close()
                monitor.log_summary(log)
                return False
            
            if retry > 0:
                log(f"⚠️ 第 {retry + 1} 次尝试启动应用...")
                if debug_logger:
                    debug_logger.log_step(f"重试 {retry + 1}/{max_retries}", "白屏卡死，重新启动")
                
                # 清除缓存
                cache.clear(device_id)
                
                # 停止应用
                step_start = time.time()
                await self.adb.stop_app(device_id, package_name)
                await asyncio.sleep(1)
                
                # 清理缓存（只清理缓存，不清理数据）
                log("清理应用缓存（保留登录数据）...")
                # 方法1：尝试使用 pm clear-cache（Android 6.0+）
                result = await self.adb.shell(device_id, f"pm clear-cache {package_name}")
                if "Unknown" in result or "Error" in result:
                    # 方法2：如果 pm clear-cache 不支持，直接删除缓存目录
                    log("pm clear-cache 不支持，使用 rm 命令清理缓存...")
                    result = await self.adb.shell(device_id, f"rm -rf /data/data/{package_name}/cache/*")
                log(f"清理结果: {result.strip() if result.strip() else '成功'}")
                if debug_logger:
                    debug_logger.log_action("清理缓存", {"结果": result.strip() if result.strip() else '成功'})
                await asyncio.sleep(2)
                
                # 重新启动应用
                success = await self.adb.start_app(device_id, package_name, activity_name)
                log(f"启动{'成功' if success else '失败'}")
                if debug_logger:
                    debug_logger.log_action("启动应用", {"成功": success})
                
                step_duration = time.time() - step_start
                monitor.record_step(f"重试{retry + 1}: 清理并重启", step_duration, "")
                
                await asyncio.sleep(3)
            else:
                # 第一次启动，应用已经由调用者启动
                log("等待应用完全启动...")
                await asyncio.sleep(2)
            
            loading_count = 0  # 连续LOADING状态的次数
            stuck = False  # 是否卡死需要重试
            
            # 尝试检测页面状态
            for attempt in range(max_wait_time):
                if should_stop():
                    log("用户请求停止")
                    if debug_logger:
                        debug_logger.log_warning("用户请求停止")
                        debug_logger.close()
                    monitor.log_summary(log)
                    return False
                
                step_start = time.time()
                
                # 优化点1: 使用优先级模板检测（只匹配启动流程相关的页面）
                startup_templates = [
                    '加载卡死白屏.png',      # 最可能：加载页
                    '启动页服务弹窗.png',    # 用户协议弹窗
                    '广告.png',              # 广告页
                    '首页公告.png',          # 首页弹窗
                    '首页.png',              # 目标：首页
                    '登陆.png',              # 可能：登录页
                ]
                
                result = await hybrid_detector.detect_page_with_priority(
                    device_id,
                    startup_templates,
                    use_cache=True
                )
                
                detection_method = result.detection_method if hasattr(result, 'detection_method') else 'unknown'
                step_duration = time.time() - step_start
                
                log(f"[{attempt+1}/{max_wait_time}] {result.state.value}: {result.details} [{detection_method}]")
                
                # 记录检测步骤
                monitor.record_step(f"检测{attempt+1}: {result.state.value}", step_duration, detection_method)
                
                # 保存截图和OCR结果到调试日志
                if debug_logger and attempt % 5 == 0:
                    await debug_logger.save_screenshot(
                        self.adb, device_id, 
                        f"attempt_{attempt+1}",
                        f"状态: {result.state.value}"
                    )
                
                # 如果已经到达主页或我的页面，说明启动完成
                if result.state in [PageState.HOME, PageState.PROFILE, PageState.PROFILE_LOGGED]:
                    log("✓ 已进入主界面")
                    if debug_logger:
                        debug_logger.log_result(True, f"成功到达: {result.state.value}")
                        log_path = debug_logger.close()
                        log(f"调试日志已保存: {log_path}")
                    
                    # 输出性能摘要
                    monitor.end()
                    monitor.log_summary(log)
                    return True
                
                # 如果是登录页面，说明已经跳过广告
                if result.state == PageState.LOGIN:
                    log("✓ 已到达登录页面")
                    if debug_logger:
                        debug_logger.log_result(True, "成功到达登录页面")
                        log_path = debug_logger.close()
                        log(f"调试日志已保存: {log_path}")
                    
                    monitor.end()
                    monitor.log_summary(log)
                    return True
                
                # 如果是Android桌面，说明应用还没启动或已退出
                if result.state == PageState.LAUNCHER:
                    log("检测到Android桌面，尝试启动应用...")
                    if debug_logger:
                        debug_logger.log_step("检测到Android桌面", result.details)
                        await debug_logger.save_screenshot(
                            self.adb, device_id, "launcher", "Android桌面"
                        )
                    
                    step_start = time.time()
                    success = await self.adb.start_app(device_id, package_name, activity_name)
                    step_duration = time.time() - step_start
                    monitor.record_step("启动应用", step_duration, "")
                    
                    if success:
                        log("✓ 应用启动成功，等待加载...")
                    else:
                        log("⚠️ 应用启动失败")
                    loading_count = 0
                    await asyncio.sleep(3)
                    continue
                
                # 处理弹窗（用户协议、首页公告等）
                if result.state == PageState.POPUP:
                    log("检测到弹窗，使用OCR查找关闭按钮...")
                    if debug_logger:
                        debug_logger.log_step("处理弹窗", result.details)
                        await debug_logger.save_screenshot(
                            self.adb, device_id, "popup_before", "关闭前"
                        )
                    
                    step_start = time.time()
                    success = await hybrid_detector.close_popup(device_id)
                    step_duration = time.time() - step_start
                    monitor.record_step("关闭弹窗", step_duration, "ocr")
                    
                    if success:
                        log("✓ 成功关闭弹窗")
                    else:
                        log("⚠️ 关闭弹窗失败")
                    
                    if debug_logger:
                        debug_logger.log_result(success, "关闭弹窗")
                        await debug_logger.save_screenshot(
                            self.adb, device_id, "popup_after", "关闭后"
                        )
                    
                    loading_count = 0
                    
                    # 使用全局便捷函数等待页面变化
                    log("等待页面变化...")
                    wait_start = time.time()
                    from .performance.smart_waiter import wait_for_page
                    result_after = await wait_for_page(
                        device_id,
                        hybrid_detector,
                        [PageState.HOME, PageState.PROFILE, PageState.PROFILE_LOGGED, PageState.AD, PageState.LOGIN],
                        log_callback=log
                    )
                    wait_duration = time.time() - wait_start
                    monitor.record_step("等待页面变化", wait_duration, "")
                    
                    if result_after:
                        log(f"✓ 页面已变化到: {result_after.state.value}")
                    else:
                        log("⚠️ 等待超时，继续检测...")
                    
                    continue
                
                # 处理广告页（使用增强检测器）
                if result.state == PageState.AD:
                    log("检测到广告页，使用增强检测器验证...")
                    if debug_logger:
                        debug_logger.log_step("处理广告页", "使用增强检测器")
                        await debug_logger.save_screenshot(
                            self.adb, device_id, "ad_waiting", "等待中"
                        )
                    
                    # 使用增强广告检测器进行二次验证
                    from .ad_detection.enhanced_ad_detector import EnhancedAdDetector
                    enhanced_detector = EnhancedAdDetector(self.adb)
                    
                    verify_start = time.time()
                    ad_result = await enhanced_detector.detect_ad_page(device_id)
                    verify_duration = time.time() - verify_start
                    monitor.record_step("增强广告检测", verify_duration, ad_result.method)
                    
                    log(f"增强检测结果: {'是广告' if ad_result.is_ad else '非广告'}, 置信度: {ad_result.confidence:.2f}, 方法: {ad_result.method}")
                    
                    if debug_logger:
                        debug_logger.log_action("增强广告检测", {
                            "is_ad": ad_result.is_ad,
                            "confidence": ad_result.confidence,
                            "method": ad_result.method,
                            "details": ad_result.details
                        })
                    
                    # 如果确认是广告，等待自动消失（不点击，避免误触）
                    if ad_result.is_ad:
                        log("✓ 确认为广告页，等待广告自动消失（不点击，避免误触）...")
                        
                        # 使用全局便捷函数等待广告消失
                        log("使用轮询检测等待广告消失（最多30秒）...")
                        wait_start = time.time()
                        from .performance.smart_waiter import wait_for_page
                        result_after = await wait_for_page(
                            device_id,
                            hybrid_detector,
                            [PageState.HOME, PageState.PROFILE, PageState.PROFILE_LOGGED, PageState.POPUP, PageState.LOGIN],
                            log_callback=log
                        )
                        wait_duration = time.time() - wait_start
                        monitor.record_step("等待广告消失", wait_duration, "")
                        
                        if result_after:
                            log(f"✓ 广告已消失，当前页面: {result_after.state.value}")
                        else:
                            log("⚠️ 广告仍在显示或检测超时")
                        
                        if debug_logger:
                            debug_logger.log_result(result_after is not None, "广告消失")
                            await debug_logger.save_screenshot(
                                self.adb, device_id, "ad_after", "等待后"
                            )
                    else:
                        log("⚠️ 增强检测认为不是广告页，可能是误判，继续等待...")
                        # 即使增强检测认为不是广告，也等待一下以防万一
                        await asyncio.sleep(2)
                    
                    loading_count = 0
                    continue
                
                # 处理未知页面
                if result.state == PageState.UNKNOWN:
                    log(f"检测到未知页面: {result.details}")
                    if debug_logger:
                        debug_logger.log_step("处理未知页面", result.details)
                        await debug_logger.save_screenshot(
                            self.adb, device_id, "unknown_page", "未知页面"
                        )
                    
                    # 检查是否是异常页面（需要返回）
                    if any(keyword in result.details for keyword in ["异常页面", "商品列表", "商品详情", "活动页面", "文章列表"]):
                        log("检测到异常页面，按返回键...")
                        step_start = time.time()
                        await self.adb.press_back(device_id)
                        step_duration = time.time() - step_start
                        monitor.record_step("返回键", step_duration, "")
                        
                        await asyncio.sleep(2)
                        loading_count = 0
                        continue
                    
                    # 其他未知页面，等待一下看是否会变化
                    log("未知页面，等待页面加载...")
                    loading_count = 0
                    await asyncio.sleep(1)
                    continue
                
                # 如果是加载中状态
                if result.state == PageState.LOADING:
                    loading_count += 1
                    
                    # 如果连续LOADING超过 stuck_timeout 秒，标记为白屏卡死
                    if loading_count >= stuck_timeout:
                        log(f"⚠️ 检测到白屏卡死（连续{loading_count}秒LOADING）")
                        if debug_logger:
                            debug_logger.log_warning(f"白屏卡死（连续{loading_count}秒LOADING）")
                            await debug_logger.save_screenshot(
                                self.adb, device_id, "stuck_screen", "白屏卡死"
                            )
                        stuck = True
                        break
                    
                    if loading_count % 5 == 0:
                        log(f"仍在加载中... ({loading_count}秒)")
                    
                    await asyncio.sleep(1)
                    continue
                else:
                    # 不是LOADING状态，重置计数器
                    loading_count = 0
                
                # 其他未处理的状态，等待一下
                await asyncio.sleep(1)
            
            # 循环结束，检查最终状态
            if not stuck:
                # 再次检测当前页面状态
                step_start = time.time()
                final_result = await hybrid_detector.detect_page(device_id, use_ocr=True)
                step_duration = time.time() - step_start
                monitor.record_step("最终检测", step_duration, "ocr")
                
                log(f"循环结束，最终状态: {final_result.state.value}")
                
                # 如果已经到达目标页面，返回成功
                if final_result.state in [PageState.HOME, PageState.PROFILE, PageState.PROFILE_LOGGED, PageState.LOGIN]:
                    log("✓ 启动流程完成")
                    if debug_logger:
                        debug_logger.log_result(True, f"成功到达: {final_result.state.value}")
                        log_path = debug_logger.close()
                        log(f"调试日志已保存: {log_path}")
                    
                    monitor.end()
                    monitor.log_summary(log)
                    return True
                
                # 如果是弹窗或其他中间状态，多尝试几次
                elif final_result.state in [PageState.POPUP, PageState.SPLASH, PageState.LOADING]:
                    log(f"循环结束时状态为 {final_result.state.value}，尝试继续处理...")
                    
                    # 如果是弹窗，尝试关闭
                    if final_result.state == PageState.POPUP:
                        log("尝试关闭弹窗...")
                        step_start = time.time()
                        await hybrid_detector.close_popup(device_id)
                        step_duration = time.time() - step_start
                        monitor.record_step("额外关闭弹窗", step_duration, "ocr")
                        await asyncio.sleep(3)
                    else:
                        await asyncio.sleep(3)
                    
                    # 再次检测，最多尝试3次
                    for extra_attempt in range(3):
                        step_start = time.time()
                        final_result = await hybrid_detector.detect_page(device_id, use_ocr=True)
                        step_duration = time.time() - step_start
                        monitor.record_step(f"额外检测{extra_attempt+1}", step_duration, "ocr")
                        
                        log(f"额外检测 {extra_attempt+1}/3: {final_result.state.value}")
                        
                        if final_result.state in [PageState.HOME, PageState.PROFILE, PageState.PROFILE_LOGGED, PageState.LOGIN]:
                            log("✓ 启动流程完成")
                            if debug_logger:
                                debug_logger.log_result(True, f"成功到达: {final_result.state.value}")
                                log_path = debug_logger.close()
                                log(f"调试日志已保存: {log_path}")
                            
                            monitor.end()
                            monitor.log_summary(log)
                            return True
                        
                        # 如果还是弹窗，继续关闭
                        if final_result.state == PageState.POPUP:
                            log("仍有弹窗，继续关闭...")
                            step_start = time.time()
                            await hybrid_detector.close_popup(device_id)
                            step_duration = time.time() - step_start
                            monitor.record_step(f"额外关闭弹窗{extra_attempt+1}", step_duration, "ocr")
                        
                        await asyncio.sleep(2)
                    
                    # 3次额外尝试后仍未成功，标记为卡死
                    log(f"⚠️ 多次尝试后仍未到达目标页面: {final_result.state.value}")
                    stuck = True
                
                else:
                    # 其他未知状态，标记为卡死
                    log(f"⚠️ 超时未到达目标页面，当前状态: {final_result.state.value}")
                    if debug_logger:
                        debug_logger.log_warning(f"超时未到达目标页面: {final_result.state.value}")
                        await debug_logger.save_screenshot(
                            self.adb, device_id, "timeout", "超时"
                        )
                    stuck = True
            
            # 如果卡死了，继续下一次重试
            if stuck:
                log(f"⚠️ 启动失败，准备重试 ({retry + 1}/{max_retries})")
        
        # 所有重试都失败
        log(f"✗ 启动失败，已重试 {max_retries} 次，跳过此账号")
        if debug_logger:
            debug_logger.log_error(f"启动失败，已重试 {max_retries} 次")
            log_path = debug_logger.close()
            log(f"调试日志已保存: {log_path}")
        
        monitor.end()
        monitor.log_summary(log)
        return False

    async def do_balance_transfer(self, device_id: str, user_id: str, recipient_id: str, 
                                   balance: float, log_callback=None) -> Dict[str, any]:
        """执行余额转账
        
        Args:
            device_id: 设备ID
            user_id: 当前用户ID
            recipient_id: 收款用户ID
            balance: 当前余额
            log_callback: 日志回调函数
            
        Returns:
            dict: 转账结果
                - success: bool, 是否成功
                - message: str, 结果消息
                - amount: float, 转账金额
        """
        def log(msg):
            if log_callback:
                log_callback(msg)
            else:
                print(msg)
        
        try:
            # 检查转账配置
            from .transfer_config import get_transfer_config
            config = get_transfer_config()
            
            # 判断是否应该转账
            if not config.should_transfer(user_id, balance):
                if not config.enabled:
                    log("  [转账] 转账功能未启用，跳过")
                elif user_id in config.recipient_ids:
                    log("  [转账] 当前账号是收款账号，跳过转账")
                elif not config.recipient_ids:
                    log("  [转账] 未配置收款账号，跳过转账")
                elif balance <= config.min_balance:
                    log(f"  [转账] 余额 {balance:.2f} 元不足（最小保留 {config.min_balance:.2f} 元），跳过转账")
                return {
                    'success': True,
                    'message': '跳过转账',
                    'amount': 0.0,
                    'skipped': True
                }
            
            # 计算转账金额
            transfer_amount = config.get_transfer_amount(balance)
            log(f"  [转账] 准备转账 {transfer_amount:.2f} 元到用户 {recipient_id}")
            
            # 执行转账
            result = await self.balance_transfer.transfer_balance(
                device_id, 
                recipient_id, 
                log_callback=log
            )
            
            if result['success']:
                result['amount'] = transfer_amount
                log(f"  [转账] ✓ 转账成功: {transfer_amount:.2f} 元")
            else:
                log(f"  [转账] ❌ 转账失败: {result['message']}")
            
            return result
            
        except Exception as e:
            log(f"  [转账] ❌ 转账异常: {e}")
            return {
                'success': False,
                'message': f"转账异常: {e}",
                'amount': 0.0
            }

    async def handle_startup_flow_integrated(self, device_id: str, log_callback=None, stop_check=None,
                                            package_name: str = "com.ry.xmsc", activity_name: str = None,
                                            max_retries: int = 3) -> bool:
        """处理应用启动流程 - 使用整合检测器和智能等待器（GPU加速）
        
        优化点：
        1. 使用整合检测器（PyTorch GPU加速，2.24ms）
        2. 使用智能等待器（高频轮询0.3秒，即时响应）
        3. 替换所有固定等待为智能等待
        4. GPU加速页面分类，速度提升4.54倍
        5. 详细记录每个步骤的耗时，用于性能优化分析
        
        流程：启动页 -> 用户协议弹窗(关闭) -> 广告页(智能等待) -> 首页弹窗(关闭) -> 主页
        
        Args:
            device_id: 设备 ID
            log_callback: 日志回调函数
            stop_check: 停止检查函数
            package_name: 应用包名
            activity_name: Activity名称
            max_retries: 最大重试次数
            
        Returns:
            是否成功
        """
        import time
        
        def log(msg):
            if log_callback:
                log_callback(msg)
        
        def should_stop():
            if stop_check:
                return stop_check()
            return False
        
        # 记录总开始时间
        total_start_time = time.time()
        log(f"\n{'='*60}")
        log(f"[时间记录] 启动流程开始 - {time.strftime('%H:%M:%S')}")
        log(f"{'='*60}")
        
        # 使用已初始化的整合检测器（GPU加速深度学习）
        log("[GPU加速] 使用已初始化的深度学习检测器")
        detector = self.integrated_detector
        log("")
        
        for retry in range(max_retries):
            if should_stop():
                log("用户请求停止")
                return False
            
            if retry > 0:
                log(f"⚠️ 第 {retry + 1} 次尝试启动应用...")
                
                # 停止应用
                await self.adb.stop_app(device_id, package_name)
                await asyncio.sleep(1)
                
                # 清理缓存
                log("清理应用缓存...")
                result = await self.adb.shell(device_id, f"pm clear-cache {package_name}")
                if "Unknown" in result or "Error" in result:
                    result = await self.adb.shell(device_id, f"rm -rf /data/data/{package_name}/cache/*")
                log(f"清理结果: {result.strip() if result.strip() else '成功'}")
                
                # 重新启动应用
                await asyncio.sleep(1)
                success = await self.adb.start_app(device_id, package_name, activity_name)
                log(f"启动{'成功' if success else '失败'}")
                
                # 智能等待应用启动（替换固定等待3秒）
                log("[智能等待] 等待应用启动...")
                await asyncio.sleep(1)  # 最小等待，让应用开始启动
            else:
                # 第一次启动
                log("[GPU加速] 应用已启动，开始智能检测...")
            
            # 主循环：智能检测和处理页面
            max_wait_time = 60
            start_time = asyncio.get_event_loop().time()
            
            log(f"\n{'='*60}")
            log(f"开始智能启动流程检测（只检测5个启动相关页面）")
            log(f"{'='*60}")
            
            while asyncio.get_event_loop().time() - start_time < max_wait_time:
                if should_stop():
                    log("用户请求停止")
                    return False
                
                # 使用深度学习检测器检测当前页面（GPU加速，只检测启动相关页面）
                detect_start = time.time()
                
                # 定义启动流程需要检测的页面类别
                startup_pages = [
                    PageState.STARTUP_POPUP,  # 启动服务弹窗
                    PageState.AD,              # 广告页
                    PageState.LOADING,         # 加载页
                    PageState.HOME_NOTICE,     # 首页广告
                    PageState.HOME,            # 首页
                    PageState.LOGIN,           # 登录页（可能直接到登录）
                    PageState.PROFILE_LOGGED,  # 个人页（可能直接到个人页）
                    PageState.LAUNCHER         # Android桌面
                ]
                
                result = await detector.detect_page(device_id, use_cache=True, detect_elements=False)
                detect_time = time.time() - detect_start
                elapsed = asyncio.get_event_loop().time() - start_time
                log(f"[{elapsed:.1f}s] {result.state.value} (置信度: {result.confidence:.2%}, 检测耗时: {detect_time*1000:.2f}ms)")
                
                # 检查是否已到达目标页面
                if result.state in [PageState.HOME, PageState.PROFILE_LOGGED, PageState.PROFILE, PageState.LOGIN]:
                    total_time = time.time() - total_start_time
                    log(f"\n{'='*60}")
                    log(f"[时间记录] ✓ 启动流程完成")
                    log(f"[时间记录] 到达页面: {result.state.value}")
                    log(f"[时间记录] 总耗时: {total_time:.3f}秒")
                    log(f"[时间记录] 完成时间: {time.strftime('%H:%M:%S')}")
                    log(f"{'='*60}\n")
                    return True
                
                # 处理Android桌面
                if result.state == PageState.LAUNCHER:
                    step_start = time.time()
                    log("检测到Android桌面，启动应用...")
                    await self.adb.start_app(device_id, package_name, activity_name)
                    
                    # 智能等待应用启动（替换固定等待3秒）
                    log("[智能等待] 等待应用启动...")
                    wait_start = time.time()
                    from .performance.smart_waiter import wait_for_page
                    result = await wait_for_page(
                        device_id,
                        detector,
                        [PageState.LOADING, PageState.SPLASH, PageState.STARTUP_POPUP, PageState.AD],
                        log_callback=lambda msg: log(f"  [等待] {msg}")
                    )
                    wait_time = time.time() - wait_start
                    log(f"✓ 应用启动完成 (耗时: {time.time() - step_start:.2f}秒, 等待: {wait_time:.2f}秒)")
                    continue
                
                # 处理启动页服务弹窗
                if result.state == PageState.STARTUP_POPUP:
                    step_start = time.time()
                    log(f"[步骤] 检测到启动页服务弹窗")
                    
                    log("  [YOLO] 检测并点击'同意'按钮...")
                    
                    # 使用整合检测器点击元素（YOLO会在这里自动加载）
                    click_start = time.time()
                    success = await detector.click_element(device_id, "同意按钮")
                    click_time = time.time() - click_start
                    
                    if success:
                        log(f"  ✓ 成功点击'同意'按钮 (检测+点击耗时: {click_time:.3f}秒)")
                    else:
                        log(f"  ⚠️ 未找到'同意'按钮，使用固定坐标... (检测耗时: {click_time:.3f}秒)")
                        await self.adb.tap(device_id, 270, 600)
                    
                    # 优化：使用智能等待器等待页面变化（替换固定等待0.5秒）
                    log("  [智能等待] 等待页面变化...")
                    from .performance.smart_waiter import wait_for_page
                    wait_result = await wait_for_page(
                        device_id,
                        detector,
                        [PageState.HOME, PageState.AD, PageState.LOADING, PageState.HOME_NOTICE,
                         PageState.PROFILE_LOGGED, PageState.PROFILE, PageState.LOGIN],
                        log_callback=lambda msg: log(f"    [等待] {msg}")
                    )
                    
                    step_time = time.time() - step_start
                    log(f"[时间记录] 弹窗处理完成 - 总耗时: {step_time:.3f}秒")
                    log("")
                    continue
                
                # 处理首页公告弹窗
                if result.state == PageState.HOME_NOTICE:
                    step_start = time.time()
                    log(f"[步骤] 检测到首页公告弹窗")
                    
                    log("  [YOLO] 检测并点击'确认'按钮...")
                    
                    # 使用整合检测器点击元素（YOLO会在这里自动加载）
                    click_start = time.time()
                    success = await detector.click_element(device_id, "确认按钮")
                    click_time = time.time() - click_start
                    
                    if success:
                        log(f"  ✓ 成功点击'确认'按钮 (检测+点击耗时: {click_time:.3f}秒)")
                    else:
                        log(f"  ⚠️ 未找到'确认'按钮，使用固定坐标... (检测耗时: {click_time:.3f}秒)")
                        await self.adb.tap(device_id, 270, 690)
                    
                    # 优化：使用智能等待器等待页面变化（替换固定等待1.5秒）
                    log("  [智能等待] 等待页面变化...")
                    from .performance.smart_waiter import wait_for_page
                    wait_result = await wait_for_page(
                        device_id,
                        detector,
                        [PageState.HOME, PageState.PROFILE_LOGGED, PageState.PROFILE, PageState.LOGIN],
                        log_callback=lambda msg: log(f"    [等待] {msg}")
                    )
                    
                    step_time = time.time() - step_start
                    
                    if wait_result:
                        total_time = time.time() - total_start_time
                        log(f"[时间记录] 弹窗处理完成 - 总耗时: {step_time:.3f}秒")
                        log(f"\n{'='*60}")
                        log(f"[时间记录] ✓ 启动流程完成")
                        log(f"[时间记录] 到达页面: {wait_result.state.value}")
                        log(f"[时间记录] 总耗时: {total_time:.3f}秒")
                        log(f"[时间记录] 完成时间: {time.strftime('%H:%M:%S')}")
                        log(f"{'='*60}\n")
                        return True
                    
                    log(f"[时间记录] 弹窗处理完成 - 总耗时: {step_time:.3f}秒")
                    log("")
                    continue
                
                # 处理首页异常代码弹窗
                if result.state == PageState.HOME_ERROR_POPUP:
                    log("[YOLO] 检测到首页异常代码弹窗，点击'确认'按钮...")
                    
                    success = await detector.click_element(device_id, "确认按钮")
                    if success:
                        log("[YOLO] ✓ 成功点击'确认'按钮")
                    else:
                        log("[YOLO] ⚠️ 未找到'确认'按钮，使用固定坐标...")
                        await self.adb.tap(device_id, 270, 690)
                    
                    # 智能等待页面变化
                    log("[智能等待] 等待页面变化...")
                    from .performance.smart_waiter import wait_for_page
                    result = await wait_for_page(
                        device_id,
                        detector,
                        [PageState.HOME, PageState.PROFILE_LOGGED, PageState.PROFILE, PageState.LOGIN],
                        log_callback=lambda msg: log(f"  [等待] {msg}")
                    )
                    
                    if result:
                        log(f"✓ 启动流程完成，到达: {result.state.value}")
                        return True
                    continue
                
                # 处理广告页（使用智能等待器等待广告消失）
                if result.state == PageState.AD:
                    step_start = time.time()
                    log(f"[步骤] 检测到广告页，智能等待广告消失...")
                    
                    # 优化：使用智能等待器等待广告消失（替换固定轮询）
                    from .performance.smart_waiter import wait_for_page
                    wait_result = await wait_for_page(
                        device_id,
                        detector,
                        [PageState.HOME, PageState.PROFILE_LOGGED, PageState.PROFILE, PageState.LOGIN, 
                         PageState.HOME_NOTICE, PageState.STARTUP_POPUP],  # 广告后可能到达的页面
                        log_callback=lambda msg: log(f"  [智能等待] {msg}")
                    )
                    
                    step_time = time.time() - step_start
                    
                    if wait_result:
                        log(f"[时间记录] 广告已消失 - 当前页面: {wait_result.state.value}, 耗时: {step_time:.3f}秒")
                        log("")
                        
                        # 如果已到达目标页面，直接返回
                        if wait_result.state in [PageState.HOME, PageState.PROFILE_LOGGED, PageState.PROFILE, PageState.LOGIN]:
                            total_time = time.time() - total_start_time
                            log(f"\n{'='*60}")
                            log(f"[时间记录] ✓ 启动流程完成")
                            log(f"[时间记录] 到达页面: {wait_result.state.value}")
                            log(f"[时间记录] 总耗时: {total_time:.3f}秒")
                            log(f"[时间记录] 完成时间: {time.strftime('%H:%M:%S')}")
                            log(f"{'='*60}\n")
                            return True
                    else:
                        log(f"[时间记录] ⚠️ 广告等待超时，继续流程... (耗时: {step_time:.3f}秒)")
                        log("")
                    continue
                
                # 处理加载页
                if result.state == PageState.LOADING:
                    log("[智能等待] 页面加载中...")
                    
                    # 智能等待加载完成（替换固定等待）
                    from .performance.smart_waiter import wait_for_page
                    result = await wait_for_page(
                        device_id,
                        detector,
                        [PageState.HOME, PageState.AD, PageState.STARTUP_POPUP, PageState.HOME_NOTICE,
                         PageState.PROFILE_LOGGED, PageState.PROFILE, PageState.LOGIN],
                        log_callback=lambda msg: log(f"  [等待] {msg}")
                    )
                    
                    if result:
                        log(f"✓ 加载完成，当前页面: {result.state.value}")
                        
                        # 如果已到达目标页面，直接返回
                        if result.state in [PageState.HOME, PageState.PROFILE_LOGGED, PageState.PROFILE, PageState.LOGIN]:
                            log(f"✓ 启动流程完成")
                            return True
                    continue
                
                # 处理温馨提示弹窗
                if result.state == PageState.WARMTIP:
                    step_start = time.time()
                    log(f"[步骤] 检测到温馨提示弹窗")
                    
                    log("  [YOLO] 检测并点击'关闭'按钮...")
                    
                    click_start = time.time()
                    success = await detector.click_element(device_id, "关闭按钮")
                    click_time = time.time() - click_start
                    
                    if success:
                        log(f"  ✓ 成功点击'关闭'按钮 (检测+点击耗时: {click_time:.3f}秒)")
                    else:
                        log(f"  ⚠️ 未找到'关闭'按钮，按返回键... (检测耗时: {click_time:.3f}秒)")
                        await self.adb.press_back(device_id)
                    
                    # 优化：使用智能等待器等待页面变化（替换固定等待0.5秒）
                    log("  [智能等待] 等待页面变化...")
                    from .performance.smart_waiter import wait_for_page
                    wait_result = await wait_for_page(
                        device_id,
                        detector,
                        [PageState.HOME, PageState.PROFILE_LOGGED, PageState.PROFILE, PageState.LOGIN],
                        log_callback=lambda msg: log(f"    [等待] {msg}")
                    )
                    
                    step_time = time.time() - step_start
                    
                    if wait_result:
                        total_time = time.time() - total_start_time
                        log(f"[时间记录] 弹窗处理完成 - 总耗时: {step_time:.3f}秒")
                        log(f"\n{'='*60}")
                        log(f"[时间记录] ✓ 启动流程完成")
                        log(f"[时间记录] 到达页面: {wait_result.state.value}")
                        log(f"[时间记录] 总耗时: {total_time:.3f}秒")
                        log(f"[时间记录] 完成时间: {time.strftime('%H:%M:%S')}")
                        log(f"{'='*60}\n")
                        return True
                    
                    log(f"[时间记录] 弹窗处理完成 - 总耗时: {step_time:.3f}秒")
                    log("")
                    continue
                
                # 处理未知页面
                if result.state == PageState.UNKNOWN:
                    log(f"⚠️ 检测到未知页面，等待...")
                    await asyncio.sleep(0.5)  # 优化：从1秒改为0.5秒
                    continue
                
                # 其他状态，短暂等待（优化：从0.3秒改为0.2秒，更快响应）
                await asyncio.sleep(0.2)
            
            # 超时，检查最终状态
            log("⚠️ 启动流程超时，检查最终状态...")
            final_result = await detector.detect_page(device_id, use_cache=False, detect_elements=False)
            log(f"最终状态: {final_result.state.value}")
            
            if final_result.state in [PageState.HOME, PageState.PROFILE_LOGGED, PageState.PROFILE, PageState.LOGIN]:
                log(f"✓ 启动流程完成（超时后检测）")
                return True
            
            # 如果不是最后一次重试，继续
            if retry < max_retries - 1:
                log(f"⚠️ 启动流程失败，准备重试...")
                continue
        
        # 所有重试都失败
        log("✗ 启动流程失败，已达到最大重试次数")
        return False
