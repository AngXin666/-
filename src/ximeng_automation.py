"""
溪盟商城业务自动化模块
Ximeng Mall Business Automation Module

代码清理记录：
- 2026-02-02: 删除废弃的启动流程函数（handle_startup_flow, handle_startup_flow_optimized）
- 保留实际使用的函数（handle_startup_flow_integrated）
- 文件从 3531 行减少到 520 行（减少 85%）
- 备份文件：ximeng_automation_backup_20260202.py
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
    
    async def handle_startup_flow_integrated(self, device_id: str, log_callback=None, stop_check=None,
                                            package_name: str = "com.ry.xmsc", activity_name: str = None,
                                            max_retries: int = 3, file_logger=None) -> bool:
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
            file_logger: 文件日志记录器（用于详细技术日志）
            
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
        if file_logger:
            file_logger.info(f"启动流程开始 - {time.strftime('%H:%M:%S')}")
        
        # 简洁日志：步骤1 - 启动应用
        concise.step(1, "启动应用")
        
        # 使用已初始化的整合检测器（GPU加速深度学习）
        detector = self.integrated_detector
        if file_logger:
            file_logger.debug("使用已初始化的深度学习检测器（GPU加速）")
        
        for retry in range(max_retries):
            if should_stop():
                if file_logger:
                    file_logger.info("用户请求停止")
                return False
            
            if retry > 0:
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
                    file_logger.debug("应用已启动，开始智能检测")
            
            # 主循环：智能检测和处理页面
            max_wait_time = 60
            start_time = asyncio.get_event_loop().time()
            
            if file_logger:
                file_logger.debug("开始智能启动流程检测")
            
            while asyncio.get_event_loop().time() - start_time < max_wait_time:
                if should_stop():
                    if file_logger:
                        file_logger.info("用户请求停止")
                    return False
                
                # 使用深度学习检测器检测当前页面（GPU加速）
                detect_start = time.time()
                
                result = await detector.detect_page(device_id, use_cache=True, detect_elements=False)
                detect_time = time.time() - detect_start
                elapsed = asyncio.get_event_loop().time() - start_time
                
                # 详细日志：记录检测结果（仅文件日志）
                if file_logger:
                    file_logger.debug(
                        f"[{elapsed:.1f}s] {result.state.value} "
                        f"(置信度: {result.confidence:.2%}, 检测耗时: {detect_time*1000:.2f}ms)"
                    )
                
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
                    return True
                
                # 如果到达登录页或个人页，说明启动流程异常（应该到首页）
                if result.state in [PageState.PROFILE_LOGGED, PageState.PROFILE, PageState.LOGIN]:
                    if file_logger:
                        file_logger.warning(
                            f"启动流程异常：到达 {result.state.value}，预期应该到达首页，继续等待"
                        )
                    await asyncio.sleep(0.5)
                    continue
                
                # 处理Android桌面
                if result.state == PageState.LAUNCHER:
                    step_start = time.time()
                    if file_logger:
                        file_logger.debug("检测到Android桌面，启动应用")
                    
                    await self.adb.start_app(device_id, package_name, activity_name)
                    
                    # 智能等待应用启动
                    wait_start = time.time()
                    from .performance.smart_waiter import wait_for_page
                    result = await wait_for_page(
                        device_id,
                        detector,
                        [PageState.LOADING, PageState.SPLASH, PageState.STARTUP_POPUP, PageState.AD],
                        log_callback=None  # 不传递日志回调，避免显示技术细节
                    )
                    wait_time = time.time() - wait_start
                    
                    if file_logger:
                        file_logger.debug(
                            f"应用启动完成 (耗时: {int(time.time() - step_start)}秒, 等待: {int(wait_time)}秒)"
                        )
                    continue
                
                # 处理启动页服务弹窗
                if result.state == PageState.STARTUP_POPUP:
                    step_start = time.time()
                    
                    # 简洁日志：关闭服务弹窗
                    concise.action("关闭服务弹窗")
                    
                    # 使用整合检测器点击元素（YOLO会在这里自动加载）
                    click_start = time.time()
                    success = await detector.click_element(device_id, "同意按钮")
                    click_time = time.time() - click_start
                    
                    # 详细日志（仅文件日志）
                    if file_logger:
                        if success:
                            file_logger.debug(f"成功点击'同意'按钮 (检测+点击耗时: {int(click_time)}秒)")
                        else:
                            file_logger.debug(f"未找到'同意'按钮，使用固定坐标 (检测耗时: {int(click_time)}秒)")
                    
                    if not success:
                        await self.adb.tap(device_id, 270, 600)
                    
                    # 使用智能等待器等待页面变化
                    from .performance.smart_waiter import wait_for_page
                    wait_result = await wait_for_page(
                        device_id,
                        detector,
                        [PageState.HOME, PageState.AD, PageState.LOADING, PageState.HOME_NOTICE,
                         PageState.PROFILE_LOGGED, PageState.PROFILE, PageState.LOGIN],
                        log_callback=None  # 不传递日志回调
                    )
                    
                    step_time = time.time() - step_start
                    if file_logger:
                        file_logger.debug(f"弹窗处理完成 - 总耗时: {int(step_time)}秒")
                    continue
                
                # 处理首页公告弹窗
                if result.state == PageState.HOME_NOTICE:
                    step_start = time.time()
                    
                    # 简洁日志：关闭首页广告
                    concise.action("关闭首页广告")
                    
                    # 点击弹窗外面的上方空白背景区域（避开搜索框）
                    await self.adb.tap(device_id, 270, 200)
                    
                    # 使用智能等待器等待页面变化
                    from .performance.smart_waiter import wait_for_page
                    wait_result = await wait_for_page(
                        device_id,
                        detector,
                        [PageState.HOME],  # 启动流程只等待首页
                        log_callback=None  # 不传递日志回调
                    )
                    
                    step_time = time.time() - step_start
                    
                    if wait_result:
                        total_time = time.time() - total_start_time
                        
                        # 简洁日志：到达首页
                        concise.success("到达首页")
                        
                        # 详细日志（仅文件日志）
                        if file_logger:
                            file_logger.info(
                                f"弹窗处理完成 - 总耗时: {int(step_time)}秒"
                            )
                            file_logger.info(
                                f"启动流程完成 - 到达页面: {wait_result.state.value}, "
                                f"总耗时: {int(total_time)}秒, 完成时间: {time.strftime('%H:%M:%S')}"
                            )
                        return True
                    
                    if file_logger:
                        file_logger.debug(f"弹窗处理完成 - 总耗时: {int(step_time)}秒")
                    continue
                
                # 处理首页异常代码弹窗
                if result.state == PageState.HOME_ERROR_POPUP:
                    if file_logger:
                        file_logger.debug("检测到首页异常代码弹窗，点击'确认'按钮")
                    
                    success = await detector.click_element(device_id, "确认按钮")
                    
                    if not success:
                        # 修正坐标：根据标注数据,确认按钮中心约在 (265, 637)
                        await self.adb.tap(device_id, 265, 637)
                        if file_logger:
                            file_logger.debug("未找到'确认'按钮，使用固定坐标")
                    
                    # 智能等待页面变化
                    from .performance.smart_waiter import wait_for_page
                    result = await wait_for_page(
                        device_id,
                        detector,
                        [PageState.HOME, PageState.PROFILE_LOGGED, PageState.PROFILE, PageState.LOGIN],
                        log_callback=None  # 不传递日志回调
                    )
                    
                    if result:
                        if file_logger:
                            file_logger.info(f"启动流程完成，到达: {result.state.value}")
                        return True
                    continue
                
                # 处理广告页（使用智能等待器等待广告消失）
                if result.state == PageState.AD:
                    step_start = time.time()
                    
                    # 简洁日志：等待广告
                    concise.action("等待广告")
                    
                    # 使用智能等待器等待广告消失
                    from .performance.smart_waiter import wait_for_page
                    wait_result = await wait_for_page(
                        device_id,
                        detector,
                        [PageState.HOME, PageState.PROFILE_LOGGED, PageState.PROFILE, PageState.LOGIN, 
                         PageState.HOME_NOTICE, PageState.STARTUP_POPUP],  # 广告后可能到达的页面
                        log_callback=None  # 不传递日志回调
                    )
                    
                    step_time = time.time() - step_start
                    
                    if wait_result:
                        # 详细日志（仅文件日志）
                        if file_logger:
                            file_logger.debug(
                                f"广告已消失 - 当前页面: {wait_result.state.value}, 耗时: {int(step_time)}秒"
                            )
                        
                        # 如果已到达目标页面，直接返回
                        if wait_result.state in [PageState.HOME, PageState.PROFILE_LOGGED, PageState.PROFILE, PageState.LOGIN]:
                            total_time = time.time() - total_start_time
                            
                            # 如果到达首页，添加简洁日志
                            if wait_result.state == PageState.HOME:
                                concise.success("到达首页")
                            
                            # 详细日志（仅文件日志）
                            if file_logger:
                                file_logger.info(
                                    f"启动流程完成 - 到达页面: {wait_result.state.value}, "
                                    f"总耗时: {int(total_time)}秒, 完成时间: {time.strftime('%H:%M:%S')}"
                                )
                            return True
                    else:
                        if file_logger:
                            file_logger.warning(f"广告等待超时，继续流程 (耗时: {int(step_time)}秒)")
                    continue
                
                # 处理加载页
                if result.state == PageState.LOADING:
                    # 智能等待加载完成
                    from .performance.smart_waiter import wait_for_page
                    result = await wait_for_page(
                        device_id,
                        detector,
                        [PageState.HOME, PageState.AD, PageState.STARTUP_POPUP, PageState.HOME_NOTICE,
                         PageState.PROFILE_LOGGED, PageState.PROFILE, PageState.LOGIN],
                        log_callback=None  # 不传递日志回调
                    )
                    
                    if result:
                        # 详细日志（仅文件日志）
                        if file_logger:
                            file_logger.debug(f"加载完成，当前页面: {result.state.value}")
                        
                        # 如果已到达首页，启动流程完成
                        if result.state == PageState.HOME:
                            concise.success("到达首页")
                            if file_logger:
                                file_logger.info("启动流程完成")
                            return True
                    continue
                
                # 处理温馨提示弹窗
                if result.state == PageState.WARMTIP:
                    step_start = time.time()
                    
                    click_start = time.time()
                    success = await detector.click_element(device_id, "关闭按钮")
                    click_time = time.time() - click_start
                    
                    # 详细日志（仅文件日志）
                    if file_logger:
                        if success:
                            file_logger.debug(f"成功点击'关闭'按钮 (检测+点击耗时: {int(click_time)}秒)")
                        else:
                            file_logger.debug(f"未找到'关闭'按钮，按返回键 (检测耗时: {int(click_time)}秒)")
                    
                    if not success:
                        await self.adb.press_back(device_id)
                    
                    # 使用智能等待器等待页面变化
                    from .performance.smart_waiter import wait_for_page
                    wait_result = await wait_for_page(
                        device_id,
                        detector,
                        [PageState.HOME, PageState.PROFILE_LOGGED, PageState.PROFILE, PageState.LOGIN],
                        log_callback=None  # 不传递日志回调
                    )
                    
                    step_time = time.time() - step_start
                    
                    if wait_result:
                        total_time = time.time() - total_start_time
                        
                        # 如果到达首页，添加简洁日志
                        if wait_result.state == PageState.HOME:
                            concise.success("到达首页")
                        
                        # 详细日志（仅文件日志）
                        if file_logger:
                            file_logger.debug(f"弹窗处理完成 - 总耗时: {int(step_time)}秒")
                            file_logger.info(
                                f"启动流程完成 - 到达页面: {wait_result.state.value}, "
                                f"总耗时: {int(total_time)}秒, 完成时间: {time.strftime('%H:%M:%S')}"
                            )
                        return True
                    
                    if file_logger:
                        file_logger.debug(f"弹窗处理完成 - 总耗时: {int(step_time)}秒")
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
            
            final_result = await detector.detect_page(device_id, use_cache=False, detect_elements=False)
            
            if file_logger:
                file_logger.info(f"最终状态: {final_result.state.value}")
            
            if final_result.state == PageState.HOME:
                concise.success("到达首页")
                if file_logger:
                    file_logger.info("启动流程完成（超时后检测）")
                return True
            
            # 如果不是最后一次重试，继续
            if retry < max_retries - 1:
                if file_logger:
                    file_logger.warning("启动流程失败，准备重试")
                continue
        
        # 所有重试都失败
        if file_logger:
            file_logger.error("启动流程失败，已达到最大重试次数")
        return False
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
        
        # 定义一个辅助函数：尝试检测并点击按钮（YOLO + OCR）
        async def try_detect_and_tap(button_name: str, model_name: str = '首页') -> bool:
            """尝试使用YOLO和OCR检测并点击按钮
            
            Returns:
                bool: 是否成功检测并点击
            """
            # 1. 尝试YOLO（直接使用按钮名称，不添加后缀）
            button_pos = await self.integrated_detector.find_button_yolo(
                device_id, 
                model_name,
                button_name,  # 直接使用按钮名称
                conf_threshold=0.3
            )
            
            if button_pos:
                log(f"  YOLO检测到'{button_name}'按钮: {button_pos}")
                await self.adb.tap(device_id, button_pos[0], button_pos[1])
                return True
            
            # 2. 降级到OCR
            log(f"  YOLO未检测到，尝试OCR...")
            ocr_pos = await self.screen_capture.find_text_location(device_id, button_name)
            
            if ocr_pos:
                log(f"  OCR检测到'{button_name}'按钮: {ocr_pos}")
                await self.adb.tap(device_id, ocr_pos[0], ocr_pos[1])
                return True
            
            return False
        
        # 尝试检测并点击"我的"按钮
        success = await try_detect_and_tap("我的")
        
        if not success:
            log(f"  未检测到'我的'按钮，可能不在首页，尝试先导航到首页...")
            
            # 尝试点击首页按钮
            home_success = await try_detect_and_tap("首页")
            
            if home_success:
                log(f"  已点击首页，等待页面加载...")
                await asyncio.sleep(0.5)
                
                # 再次尝试点击"我的"按钮
                success = await try_detect_and_tap("我的")
            
            # 如果所有智能检测都失败，返回失败
            if not success:
                log(f"  ❌ 所有检测方法都失败，无法导航到个人页")
                return False
        
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
                log(f"  ✓ 到达个人页（耗时: {int(elapsed)}秒，关闭广告: {ad_closed_count}次）")
                return True
            
            # 检测到广告 → 立即关闭
            elif current_state == PageState.PROFILE_AD:
                log(f"  ⚠️ 检测到个人页广告，立即关闭...")
                
                # 方法1: 使用YOLO检测关闭按钮
                close_button_pos = await self.integrated_detector.find_button_yolo(
                    device_id, 
                    '个人页广告',
                    '确认按钮',
                    conf_threshold=0.5
                )
                
                if close_button_pos:
                    log(f"  YOLO检测到关闭按钮: {close_button_pos}")
                    await self.adb.tap(device_id, close_button_pos[0], close_button_pos[1])
                else:
                    # 方法2: 使用返回键关闭（更可靠）
                    log(f"  YOLO未检测到按钮，使用返回键关闭")
                    await self.adb.press_back(device_id)
                
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
        log(f"  ❌ 导航到个人页超时（耗时: {int(elapsed)}秒，关闭广告: {ad_closed_count}次）")
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
        
        # 创建简洁日志记录器（GUI日志 + 文件日志）
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
            file_logger=file_logger  # ← 添加文件日志记录器
        )
        
        # 输出工作流开始信息（仅文件日志）
        file_logger.info(f"工作流开始 - {time.strftime('%H:%M:%S')}")
        file_logger.info(f"账号: {account.phone}")
        
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
            # 检查是否启用获取资料流程
            if not workflow_config.get('enable_profile', True):
                file_logger.info("="*60)
                file_logger.info("快速签到模式：仅获取余额，跳过昵称和用户ID")
                file_logger.info("="*60)
                
                # 快速签到模式：只获取余额，不获取昵称和用户ID
                step_number += 1
                concise.step(step_number, "获取余额")
                
                try:
                    # 使用 BalanceReader 快速获取余额
                    balance = await self.balance_reader.get_balance(device_id)
                    
                    if balance is not None:
                        result.balance_before = balance
                        file_logger.info(f"登录余额: {balance:.2f} 元")
                        concise.success(f"余额: {balance:.2f}元")
                    else:
                        file_logger.warning("未能获取余额")
                        concise.error("未能获取余额")
                except Exception as e:
                    file_logger.error(f"获取余额出错: {str(e)}")
                    concise.error(f"获取余额出错")
                
                # 标记为跳过完整资料获取
                profile_success = False
                profile_data = None
            else:
                step_number += 1
                
                # 添加简洁日志：步骤2 - 获取资料
                concise.step(step_number, "获取资料")
                
                # 文件日志记录详细信息
                file_logger.info(f"步骤{step_number}: 获取初始个人资料")
                
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
                            
                            # 立即检测页面状态（不等待）- 使用整合检测器（GPU加速）
                            detect_start = time.time()
                            from .page_detector import PageState
                            page_result = await self.integrated_detector.detect_page(
                                device_id, use_cache=True, detect_elements=False
                            )
                            detect_time = time.time() - detect_start
                            file_logger.info(f"页面检测耗时: {int(detect_time)}秒")
                            
                            if not page_result or not page_result.state:
                                file_logger.warning("无法检测当前页面状态")
                                if attempt < 2:
                                    await asyncio.sleep(2)
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
                                        await asyncio.sleep(2)
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
                            concise.action("进入个人页")
                            nav_start = time.time()
                            nav_success = await self._navigate_to_profile_with_ad_handling(device_id, log)
                            nav_time = time.time() - nav_start
                            file_logger.info(f"导航耗时: {int(nav_time)}秒")
                            
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
                        concise.action("获取详细资料")
                        profile_read_start = time.time()
                        file_logger.info("开始获取个人资料...")
                        account_str = f"{account.phone}----{account.password}"
                        profile_data = await self.profile_reader.get_full_profile_with_retry(
                            device_id, 
                            account=account_str,
                            gui_logger=gui_logger_obj,
                            step_number=step_number
                        )
                        profile_read_time = time.time() - profile_read_start
                        file_logger.info(f"获取个人资料耗时: {int(profile_read_time)}秒")
                        
                        # 检查是否成功获取数据（必须获取到余额、昵称和用户ID）
                        has_balance = profile_data and profile_data.get('balance') is not None
                        has_nickname = profile_data and profile_data.get('nickname') is not None
                        has_user_id = profile_data and profile_data.get('user_id') is not None
                        
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
            
            # 如果3次尝试后仍未成功（但快速签到模式除外）
            # 快速签到模式下，允许跳过获取资料，在签到时获取
            if not workflow_config.get('enable_profile', True):
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
                # ==================== 步骤3: 存储初始数据到 result ====================
                result.nickname = profile_data.get('nickname')  # 直接使用OCR结果，不使用历史数据
                result.user_id = profile_data.get('user_id')  # 直接使用OCR结果，不使用历史数据
                # phone 已经在初始化时设置
                result.balance_before = profile_data.get('balance')
                result.points = profile_data.get('points')
                result.vouchers = profile_data.get('vouchers')
                result.coupons = profile_data.get('coupons')
                
                # 显示收集到的账户信息（仅文件日志）
                file_logger.info("="*60)
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
                file_logger.info("="*60)
                
                # 调试信息：显示缓存状态（仅文件日志）
                file_logger.debug("缓存状态检查:")
                file_logger.debug(f"enable_cache: {self.auto_login.enable_cache}")
                file_logger.debug(f"cache_manager 存在: {self.auto_login.cache_manager is not None}")
                file_logger.debug(f"user_id: {result.user_id}")
                
                # ==================== 步骤3.5: 保存登录缓存（包含用户ID）====================
                # 优化：异步保存缓存，不阻塞主流程
                if self.auto_login.enable_cache and self.auto_login.cache_manager:
                    if result.user_id:
                        file_logger.info(f"异步保存登录缓存（包含用户ID: {result.user_id}）...")
                    else:
                        file_logger.info("异步保存登录缓存（未获取到用户ID）...")
                    
                    # 创建异步任务，不等待完成
                    async def save_cache_async():
                        try:
                            cache_saved = await self.auto_login.cache_manager.save_login_cache(
                                device_id, 
                                account.phone, 
                                user_id=result.user_id
                            )
                            if cache_saved:
                                file_logger.info("登录缓存已保存")
                            else:
                                file_logger.warning("登录缓存保存失败")
                        except Exception as e:
                            file_logger.error(f"保存登录缓存时出错: {str(e)}")
                    
                    # 启动后台任务，不等待完成
                    asyncio.create_task(save_cache_async())
                    file_logger.info("缓存保存任务已启动（后台执行）")
                else:
                    if not self.auto_login.enable_cache:
                        file_logger.info("缓存功能未启用，跳过缓存保存")
                    elif not self.auto_login.cache_manager:
                        file_logger.warning("缓存管理器未初始化，跳过缓存保存")
                
                # 优化：移除不必要的1秒等待
                # await asyncio.sleep(1)  # 已移除
                
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
                file_logger.info(f"优惠券: {result.coupons}")
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
                
                # 添加简洁日志：步骤3 - 签到
                concise.step(step_number, "签到")
                
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
                            'coupons': result.coupons,
                            'nickname': result.nickname,
                            'user_id': result.user_id
                        }
                    else:
                        # 快速签到模式：从数据库历史记录中获取昵称和用户ID
                        file_logger.info("快速签到模式：从数据库历史记录中获取用户信息")
                        from .local_db import LocalDatabase
                        db = LocalDatabase()
                        latest_record = db.get_latest_record_by_phone(account.phone)
                        
                        if latest_record:
                            # 从历史记录中提取信息
                            result.nickname = latest_record.get('nickname', '未知')
                            result.user_id = latest_record.get('user_id', '未知')
                            file_logger.info(f"从历史记录获取: 昵称={result.nickname}, 用户ID={result.user_id}")
                        else:
                            file_logger.warning("未找到历史记录，昵称和用户ID将在签到后获取")
                        
                        updated_profile_data = None
                    
                    # 直接调用 do_checkin，它会自动处理导航和返回首页
                    checkin_result = await self.daily_checkin.do_checkin(
                        device_id, 
                        phone=account.phone,
                        password=account.password,
                        login_callback=None,  # 已经登录，不需要回调
                        log_callback=None,
                        profile_data=updated_profile_data,  # 传递个人信息（可能为None）
                        allow_skip_profile=not profile_success  # 快速签到模式下允许跳过
                    )
                    
                    # 记录签到结果
                    if checkin_result['success']:
                        # 保存签到后余额（用于计算签到奖励）
                        checkin_balance_after = checkin_result.get('checkin_balance_after')
                        if checkin_balance_after is not None:
                            result.checkin_balance_after = checkin_balance_after
                        
                        # 计算签到奖励：签到后余额 - 余额前
                        if result.balance_before is not None and checkin_balance_after is not None:
                            result.checkin_reward = checkin_balance_after - result.balance_before
                        else:
                            # 如果无法计算，使用返回的reward_amount（可能不准确）
                            result.checkin_reward = checkin_result.get('reward_amount', 0.0)
                        
                        result.checkin_total_times = checkin_result.get('total_times')
                        
                        # 如果是快速签到模式（没有获取资料），从签到结果中提取资料
                        if not profile_success:
                            # 提取昵称、用户ID等信息
                            if checkin_result.get('nickname'):
                                result.nickname = checkin_result.get('nickname')
                            if checkin_result.get('user_id'):
                                result.user_id = checkin_result.get('user_id')
                            if checkin_result.get('balance_before') is not None:
                                result.balance_before = checkin_result.get('balance_before')
                            if checkin_result.get('points') is not None:
                                result.points = checkin_result.get('points')
                            if checkin_result.get('vouchers') is not None:
                                result.vouchers = checkin_result.get('vouchers')
                            if checkin_result.get('coupons') is not None:
                                result.coupons = checkin_result.get('coupons')
                        
                        # 注意：这里不要使用checkin_result的balance_after，因为那是签到后余额
                        # 最终余额应该在转账后更新
                        file_logger.info("签到成功")
                        file_logger.info(f"签到次数: {checkin_result.get('checkin_count', 0)}")
                        file_logger.info(f"签到奖励: {result.checkin_reward:.2f} 元")
                        
                        # 添加简洁日志：签到完成
                        concise.success("签到完成")
                        
                    elif checkin_result.get('already_checked'):
                        # 即使已签到，也要获取总次数
                        result.checkin_total_times = checkin_result.get('total_times')
                        
                        # 保存签到后余额
                        checkin_balance_after = checkin_result.get('checkin_balance_after')
                        if checkin_balance_after is not None:
                            result.checkin_balance_after = checkin_balance_after
                        
                        # 如果是快速签到模式（没有获取资料），从签到结果中提取资料
                        if not profile_success:
                            # 提取昵称、用户ID等信息
                            if checkin_result.get('nickname'):
                                result.nickname = checkin_result.get('nickname')
                            if checkin_result.get('user_id'):
                                result.user_id = checkin_result.get('user_id')
                            if checkin_result.get('balance_before') is not None:
                                result.balance_before = checkin_result.get('balance_before')
                            if checkin_result.get('points') is not None:
                                result.points = checkin_result.get('points')
                            if checkin_result.get('vouchers') is not None:
                                result.vouchers = checkin_result.get('vouchers')
                            if checkin_result.get('coupons') is not None:
                                result.coupons = checkin_result.get('coupons')
                        
                        file_logger.info("今日已签到")
                        if result.checkin_total_times:
                            file_logger.info(f"总签到次数: {result.checkin_total_times}")
                        else:
                            file_logger.info("(未获取到总签到次数)")
                        
                        # 添加简洁日志：今日已签到
                        concise.success("今日已签到")
                        
                    else:
                        # 签到失败，设置错误类型
                        from .models.error_types import ErrorType
                        result.error_type = ErrorType.CHECKIN_FAILED
                        result.error_message = checkin_result.get('message', '未知错误')
                        file_logger.error(f"签到失败: {result.error_message}")
                        
                        # 添加简洁日志：签到失败
                        concise.error(f"签到失败: {result.error_message}")
                    
                except Exception as e:
                    # 签到异常，设置错误类型
                    from .models.error_types import ErrorType
                    result.error_type = ErrorType.CHECKIN_EXCEPTION
                    result.error_message = str(e)
                    file_logger.error(f"签到流程出错: {str(e)}")
                    file_logger.info("跳过签到，继续执行后续流程")
            
            # 优化：移除签到后的1秒等待，直接进入下一步
            # await asyncio.sleep(1)  # 已移除
            
            # ==================== 步骤7: 获取最终余额（仅在签到启用且未返回余额时）====================
            # 只有在启用签到流程时才获取最终余额
            if workflow_config.get('enable_checkin', True):
                # 优化：如果签到流程已经返回了余额，跳过重复获取
                if result.balance_after is not None:
                    step_number += 1
                    
                    # 添加简洁日志：使用签到返回的余额
                    concise.step(step_number, "获取最终余额")
                    concise.success(f"最终余额: {result.balance_after:.2f}元（使用签到数据）")
                    
                    file_logger.info("="*60)
                    file_logger.info(f"步骤{step_number}: 使用签到流程返回的最终余额")
                    file_logger.info(f"最终余额: {result.balance_after:.2f} 元")
                    file_logger.info("(跳过重复获取，节省时间)")
                    file_logger.info("="*60)
                elif result.checkin_balance_after is not None:
                    # 如果有签到后余额，使用它作为最终余额（没有转账的情况）
                    step_number += 1
                    
                    concise.step(step_number, "获取最终余额")
                    
                    result.balance_after = result.checkin_balance_after
                    file_logger.info("="*60)
                    file_logger.info(f"步骤{step_number}: 使用签到后余额作为最终余额")
                    file_logger.info(f"最终余额: {result.balance_after:.2f} 元")
                    file_logger.info("="*60)
                    concise.success(f"最终余额: {result.balance_after:.2f}元（使用签到后余额）")
                elif not profile_success:
                    # 快速签到模式：使用登录时获取的余额作为最终余额
                    step_number += 1
                    
                    concise.step(step_number, "获取最终余额")
                    
                    file_logger.info("="*60)
                    file_logger.info(f"步骤{step_number}: 快速签到模式 - 使用登录余额作为最终余额")
                    file_logger.info("="*60)
                    
                    # 使用登录时获取的余额
                    if result.balance_before is not None:
                        result.balance_after = result.balance_before
                        file_logger.info(f"最终余额: {result.balance_after:.2f} 元（使用登录数据）")
                        concise.success(f"最终余额: {result.balance_after:.2f}元（使用登录数据）")
                    else:
                        file_logger.warning("登录时未获取到余额，无法设置最终余额")
                        concise.error("未获取到余额")
                else:
                    step_number += 1
                    
                    # 添加简洁日志：获取最终余额
                    concise.step(step_number, "获取最终余额")
                    
                    file_logger.info("="*60)
                    file_logger.info(f"步骤{step_number}: 获取最终个人资料")
                    file_logger.info("(签到流程未返回余额，需要重新获取)")
                    file_logger.info("="*60)
                
                    try:
                        # 判断是否执行了转账
                        has_transfer = result.transfer_amount is not None and result.transfer_amount > 0
                        
                        if has_transfer:
                            # 情况2：转账后，程序可能在钱包页或个人页
                            file_logger.info("检测到已执行转账，检查当前页面状态")
                            
                            # 检测当前页面
                            page_result = await self.detector.detect_page(device_id, use_cache=False, detect_elements=False)
                            
                            if page_result and page_result.state == PageState.WALLET:
                                # 在钱包页，按返回键回到个人页
                                concise.action("从钱包页返回个人页")
                                file_logger.info("当前在钱包页，按返回键回到个人页...")
                                await self.adb.press_back(device_id)
                                await asyncio.sleep(1.0)  # 等待页面切换
                                file_logger.info("✓ 已返回个人页")
                            elif page_result and page_result.state in [PageState.PROFILE, PageState.PROFILE_LOGGED]:
                                # 已经在个人页
                                file_logger.info("当前已在个人页")
                            else:
                                # 其他页面，尝试导航到个人页
                                file_logger.info(f"当前在其他页面（{page_result.state.value if page_result else 'unknown'}），导航到个人页")
                                concise.action("导航到个人页")
                                nav_success = await self.navigator.navigate_to_profile(device_id)
                                if not nav_success:
                                    file_logger.warning("导航到个人资料页面失败，跳过最终数据获取")
                                    concise.error("导航失败")
                                    # 继续执行，不要中断流程
                        else:
                            # 情况1：不转账，直接导航到个人页
                            concise.action("导航到个人页")
                            file_logger.info("导航到个人页...")
                            nav_success = await self.navigator.navigate_to_profile(device_id)
                            
                            if not nav_success:
                                file_logger.warning("导航到个人资料页面失败，跳过最终数据获取")
                                concise.error("导航失败")
                        
                        # 直接获取个人资料（不再需要检测和处理广告）
                            try:
                                # 获取完整个人资料（带超时）
                                # 使用并行处理方法提升性能
                                concise.action("获取最终资料")
                                account_str = f"{account.phone}----{account.password}"
                                profile_task = self.profile_reader.get_full_profile_parallel(device_id, account=account_str)
                                try:
                                    final_profile = await asyncio.wait_for(profile_task, timeout=15.0)  # 统一为15秒
                                    
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
                                        file_logger.info(f"最终余额: {result.balance_after:.2f} 元")
                                        # 添加简洁日志：最终余额
                                        concise.success(f"最终余额: {result.balance_after:.2f}元")
                                    else:
                                        file_logger.warning("未能获取最终余额")
                                        concise.error("未能获取最终余额")
                                    
                                    if result.points is not None:
                                        file_logger.info(f"最终积分: {result.points} 积分")
                                    
                                    if result.vouchers is not None:
                                        file_logger.info(f"最终抵扣券: {result.vouchers} 张")
                                    
                                    if result.coupons is not None:
                                        file_logger.info(f"最终优惠券: {result.coupons} 张")
                                    
                                except asyncio.TimeoutError:
                                    file_logger.error("获取最终数据超时")
                                    concise.error("获取超时")
                            except Exception as e:
                                file_logger.error(f"获取最终数据出错: {str(e)}")
                                concise.error(f"获取出错: {str(e)}")
                    except Exception as e:
                        file_logger.error(f"获取最终数据流程出错: {str(e)}")
                        concise.error(f"流程出错: {str(e)}")
            else:
                file_logger.info("="*60)
                file_logger.info("跳过获取最终余额")
                file_logger.info("原因: 流程控制已禁用签到，无需获取最终余额")
                file_logger.info("="*60)
                
                # 只登录模式：使用初始余额作为最终余额
                file_logger.info(f"[调试] 只登录模式 - balance_before = {result.balance_before}")
                if result.balance_before is not None:
                    result.balance_after = result.balance_before
                    file_logger.info(f"使用初始余额作为最终余额: {result.balance_after:.2f} 元")
                    concise.success(f"最终余额: {result.balance_after:.2f}元（使用登录数据）")
                else:
                    file_logger.warning("[调试] balance_before 为 None，无法设置 balance_after")
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
            
            # ==================== 步骤7.5: 自动转账（每次处理账号时重新读取配置）====================
            file_logger.info("="*60)
            file_logger.info("步骤7.5: 检查自动转账")
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
                    from .transfer_lock import get_transfer_lock
                    from .transfer_history import get_transfer_history
                    
                    transfer_config = get_transfer_config()
                    transfer_lock = get_transfer_lock()
                    transfer_history = get_transfer_history()
                    
                    should_auto_transfer = transfer_config.enabled
                    
                    if should_auto_transfer:
                        log(f"✓ 自动转账功能已启用，开始检查转账条件...")
                        
                        # 检查必要的数据
                        if not result.user_id:
                            log(f"  ⚠️ 无法获取用户ID，跳过转账")
                        elif result.balance_after is None:
                            log(f"  ⚠️ 无法获取余额，跳过转账")
                        else:
                            # 【安全检查1】检查是否有转账锁
                            if transfer_lock.is_locked(account.phone):
                                lock_info = transfer_lock.get_lock_info(account.phone)
                                log(f"  ⚠️ 转账进行中，跳过（已锁定 {lock_info['elapsed']:.1f}秒）")
                            else:
                                # 【安全检查2】检查最近5分钟是否已转账
                                recent_transfer = transfer_history.get_recent_transfer(
                                    sender_phone=account.phone,
                                    minutes=5
                                )
                                
                                if recent_transfer:
                                    log(f"  ⚠️ 最近已有转账记录，跳过")
                                    log(f"    - 时间: {recent_transfer.timestamp}")
                                    log(f"    - 金额: {recent_transfer.amount:.2f} 元")
                                    log(f"    - 收款人: {recent_transfer.recipient_phone}")
                                    log(f"    - 状态: {'成功' if recent_transfer.success else '失败'}")
                                else:
                                    # 判断是否需要转账（使用最新的配置）
                                    account_level = transfer_config.get_account_level(result.user_id)
                                    
                                    if transfer_config.should_transfer(result.user_id, result.balance_after, current_level=0):
                                        recipient_id = transfer_config.get_transfer_recipient(result.user_id, current_level=0)
                                        if recipient_id:
                                            log(f"  ✓ 满足转账条件，准备转账到 ID: {recipient_id}")
                                            
                                            # 【安全机制】获取转账锁
                                            if not transfer_lock.acquire_lock(account.phone):
                                                log(f"  ⚠️ 无法获取转账锁，跳过")
                                            else:
                                                try:
                                                    log(f"  ✓ 已获取转账锁")
                                                    
                                                    # 转账重试机制（最多重试3次）
                                                    max_transfer_retries = 3
                                                    transfer_result = None
                                                    
                                                    for transfer_attempt in range(max_transfer_retries):
                                                        if transfer_attempt > 0:
                                                            log(f"  ⚠️ 第 {transfer_attempt + 1} 次尝试转账...")
                                                            await asyncio.sleep(3)  # 重试前等待3秒
                                                            
                                                            # 重试前重新导航到个人页面
                                                            log(f"  重新导航到个人页面...")
                                                            nav_success = await self.navigator.navigate_to_profile(device_id)
                                                            if not nav_success:
                                                                log(f"  ⚠️ 导航失败，继续尝试转账...")
                                                            await asyncio.sleep(2)
                                                        
                                                        # 执行转账（传入转账前余额用于验证）
                                                        transfer_result = await self.balance_transfer.transfer_balance(
                                                            device_id,
                                                            recipient_id,
                                                            initial_balance=result.balance_after,
                                                            log_callback=log
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
                                                                owner_name = ""
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
                                                        # 转账失败，设置错误类型
                                                        from .models.error_types import ErrorType
                                                        result.error_type = ErrorType.TRANSFER_FAILED
                                                        result.error_message = transfer_result.get('message', '未知错误')
                                                        log(f"  ❌ 转账失败: {result.error_message}")
                                                        # 转账失败时，设置收款人为"失败"
                                                        result.transfer_amount = 0.0
                                                        result.transfer_recipient = "失败"
                                                        
                                                        # 保存失败记录到数据库
                                                        try:
                                                            recipient_name = transfer_result.get('recipient_name', '')
                                                            recipient_phone = recipient_id
                                                            
                                                            # 获取转账策略描述
                                                            if transfer_config.multi_level_enabled:
                                                                strategy = f"多级转账(最多{transfer_config.max_transfer_level}级)"
                                                            else:
                                                                strategy = "单级转账"
                                                            
                                                            # 获取账号的管理员信息
                                                            owner_name = ""
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
                                                    
                                                    # 保存异常记录到数据库
                                                    try:
                                                        strategy = transfer_config.strategy
                                                        
                                                        # 获取账号的管理员信息
                                                        owner_name = ""
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
                                                finally:
                                                    # 【安全机制】释放转账锁
                                                    transfer_lock.release_lock(account.phone)
                                                    log(f"  ✓ 已释放转账锁")
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
                                            log(f"  ℹ️ 余额未达标")
                    else:
                        log(f"  ℹ️ 自动转账功能未启用，跳过转账")
                    
                    log("")  # 空行
                except Exception as e:
                    log(f"  ❌ 转账检查出错: {str(e)}\n")
            
            # ==================== 步骤8: 退出登录 ====================
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
            
            return result
            
        except Exception as e:
            result.error_message = str(e)
            file_logger.error(f"工作流程出错: {str(e)}")
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


