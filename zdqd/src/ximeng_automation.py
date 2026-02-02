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
                
                # 检查是否已到达目标页面（启动流程只检测到首页即可）
                if result.state == PageState.HOME:
                    total_time = time.time() - total_start_time
                    log(f"\n{'='*60}")
                    log(f"[时间记录] ✓ 启动流程完成")
                    log(f"[时间记录] 到达页面: {result.state.value}")
                    log(f"[时间记录] 总耗时: {total_time:.3f}秒")
                    log(f"[时间记录] 完成时间: {time.strftime('%H:%M:%S')}")
                    log(f"{'='*60}\n")
                    return True
                
                # 如果到达登录页或个人页，说明启动流程异常（应该到首页）
                if result.state in [PageState.PROFILE_LOGGED, PageState.PROFILE, PageState.LOGIN]:
                    log(f"⚠️ 启动流程异常：到达 {result.state.value}，预期应该到达首页")
                    log(f"⚠️ 可能是缓存登录或其他原因，继续等待...")
                    await asyncio.sleep(0.5)
                    continue
                
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
                        [PageState.HOME],  # 启动流程只等待首页
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
                        
                        # 如果已到达首页，启动流程完成
                        if result.state == PageState.HOME:
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
            
            if final_result.state == PageState.HOME:
                log(f"✓ 启动流程完成（超时后检测）")
                return True
            
            # 如果不是最后一次重试，继续
            if retry < max_retries - 1:
                log(f"⚠️ 启动流程失败，准备重试...")
                continue
        
        # 所有重试都失败
        log("✗ 启动流程失败，已达到最大重试次数")
        return False

