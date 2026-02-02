"""
自动登录模块 - 使用坐标方式 + 页面检测
Auto Login Module - Coordinate-based with page detection
"""

import asyncio
from typing import Optional
from dataclasses import dataclass

from .screen_capture import ScreenCapture
from .ui_automation import UIAutomation
from .page_detector import PageDetector, PageState
from .login_cache_manager import LoginCacheManager
from .wait_helper import wait_for_page_state, wait_after_action, WaitResult
from .models.error_types import ErrorType


@dataclass
class LoginResult:
    """登录结果"""
    success: bool
    error_message: Optional[str] = None
    error_type: Optional[ErrorType] = None  # 错误类型枚举
    used_cache: bool = False  # 是否使用了缓存登录


class AutoLogin:
    """自动登录器 - 基于坐标 + 页面检测"""
    
    # MuMu模拟器坐标（手动验证 v1.7.0）
    COORDS_MUMU = {
        'TAB_HOME': (90, 920),
        'TAB_MY': (450, 920),
        'LOGIN_ENTRY': (270, 200),
        'PHONE_INPUT': [(111, 466), (105, 466), (117, 466)],  # 手动验证 3次
        'PASSWORD_INPUT': [(97, 532), (91, 532), (103, 532)],  # 手动验证 3次
        'AGREEMENT_CHECK': [(89, 591), (93, 589), (85, 593)],  # 已验证可用
        'LOGIN_BUTTON': [(271, 687), (221, 692), (321, 692)],  # 自动检测
        'ERROR_CONFIRM_BUTTON': (430, 588),
    }
    
    # 默认坐标
    TAB_HOME = COORDS_MUMU['TAB_HOME']
    TAB_MY = COORDS_MUMU['TAB_MY']
    LOGIN_ENTRY = COORDS_MUMU['LOGIN_ENTRY']
    PHONE_INPUT = COORDS_MUMU['PHONE_INPUT']
    PASSWORD_INPUT = COORDS_MUMU['PASSWORD_INPUT']
    AGREEMENT_CHECK = COORDS_MUMU['AGREEMENT_CHECK']
    LOGIN_BUTTON = COORDS_MUMU['LOGIN_BUTTON']
    ERROR_CONFIRM_BUTTON = COORDS_MUMU['ERROR_CONFIRM_BUTTON']
    
    def __init__(self, ui_automation: UIAutomation, screen_capture: ScreenCapture, 
                 adb_bridge=None, enable_cache: bool = True, emulator_type: str = "mumu",
                 integrated_detector=None):
        """初始化自动登录器
        
        Args:
            ui_automation: UI 自动化器实例
            screen_capture: 屏幕捕获器实例
            adb_bridge: ADB 桥接器实例
            enable_cache: 是否启用登录缓存
            emulator_type: 模拟器类型 (默认 "mumu")
            integrated_detector: 整合检测器实例（可选，用于性能优化）
        """
        self.ui_automation = ui_automation
        self.screen_capture = screen_capture
        self.adb = adb_bridge or ui_automation.adb_bridge
        self.page_detector = PageDetector(self.adb)
        self.enable_cache = enable_cache
        self.cache_manager = LoginCacheManager(self.adb) if enable_cache else None
        
        # 从ModelManager获取共享的检测器实例（不再自己创建）
        from .model_manager import ModelManager
        from .page_state_guard import PageStateGuard
        from .page_detector_hybrid_optimized import PageState
        
        model_manager = ModelManager.get_instance()
        
        # 优先使用整合检测器（深度学习），如果没有则使用混合检测器
        if integrated_detector:
            self.detector = integrated_detector
            print("[AutoLogin] 使用整合检测器（深度学习）")
        else:
            self.detector = model_manager.get_page_detector_hybrid()
            print("[AutoLogin] 使用混合检测器（模板匹配）")
        
        # 保持兼容性
        self.hybrid_detector = self.detector
        self.integrated_detector = integrated_detector
        
        self.guard = PageStateGuard(self.adb, self.detector)
        
        # 根据模拟器类型设置坐标
        self.set_emulator_type(emulator_type)
    
    def set_emulator_type(self, emulator_type: str):
        """设置模拟器类型并更新坐标
        
        Args:
            emulator_type: 模拟器类型 (默认 "mumu")
        """
        coords = self.COORDS_MUMU
        print(f"[AutoLogin v1.7.1] 使用 MuMu 模拟器坐标（手动验证）")
        print(f"[AutoLogin v1.7.1] 手机号输入框: {coords['PHONE_INPUT']}")
        print(f"[AutoLogin v1.7.1] 密码输入框: {coords['PASSWORD_INPUT']}")
        print(f"[AutoLogin v1.7.1] 协议勾选框: {coords['AGREEMENT_CHECK']}")
        
        self.TAB_HOME = coords['TAB_HOME']
        self.TAB_MY = coords['TAB_MY']
        self.LOGIN_ENTRY = coords['LOGIN_ENTRY']
        self.PHONE_INPUT = coords['PHONE_INPUT']
        self.PASSWORD_INPUT = coords['PASSWORD_INPUT']
        self.AGREEMENT_CHECK = coords['AGREEMENT_CHECK']
        self.LOGIN_BUTTON = coords['LOGIN_BUTTON']
        self.ERROR_CONFIRM_BUTTON = coords['ERROR_CONFIRM_BUTTON']
    
    async def _tap_with_fallback(self, device_id: str, coord, log_callback=None):
        """点击坐标，支持备用坐标
        
        Args:
            device_id: 设备ID
            coord: 坐标，可以是 (x, y) 或 [(x1, y1), (x2, y2), ...]
            log_callback: 日志回调
        
        Returns:
            是否成功点击
        """
        def log(msg):
            if log_callback:
                log_callback(msg)
        
        # 如果是单个坐标，转换为列表
        if isinstance(coord, tuple):
            coords = [coord]
        else:
            coords = coord
        
        log(f"  尝试点击，共有 {len(coords)} 个备用坐标")
        
        # 尝试每个坐标
        for i, (x, y) in enumerate(coords):
            try:
                log(f"  尝试坐标 {i+1}/{len(coords)}: ({x}, {y})")
                await self.adb.tap(device_id, x, y)
                log(f"  ✓ 点击成功: ({x}, {y})")
                return True
            except Exception as e:
                log(f"  ✗ 坐标 ({x}, {y}) 失败: {e}")
                if i < len(coords) - 1:
                    log(f"  等待0.2秒后尝试下一个坐标...")
                    await asyncio.sleep(0.2)
                else:
                    log(f"  ✗ 所有 {len(coords)} 个坐标都失败")
                    return False
        
        return False
    
    async def _find_agreement_checkbox_by_ocr(self, device_id: str, log_callback=None):
        """通过OCR识别文字位置，智能定位协议勾选框
        
        Args:
            device_id: 设备ID
            log_callback: 日志回调
        
        Returns:
            协议勾选框坐标 (x, y)，如果找不到返回默认坐标
        """
        def log(msg):
            if log_callback:
                log_callback(msg)
        
        try:
            # 截取屏幕
            img = await self.screen_capture.capture(device_id)
            if img is None:
                log("  ⚠️  截图失败，使用默认坐标")
                return (63, 590)
            
            # 转换为PIL图像
            from PIL import Image
            import cv2
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            # 使用OCR识别文字
            from .ocr_thread_pool import get_ocr_pool
            ocr_pool = get_ocr_pool()
            ocr_result = await ocr_pool.recognize(pil_img, timeout=5.0)
            
            if not ocr_result.texts:
                log("  ⚠️  OCR未识别到文字，使用默认坐标")
                return (63, 590)
            
            # 查找"我已阅读"或"用户协议"等关键词
            keywords = ["我已阅读", "用户协议", "隐私政策", "已阅读"]
            
            for i, (text, box) in enumerate(zip(ocr_result.texts, ocr_result.boxes)):
                for keyword in keywords:
                    if keyword in text:
                        # 找到文字，计算勾选框位置
                        x_coords = [p[0] for p in box]
                        y_coords = [p[1] for p in box]
                        
                        text_x = int(min(x_coords))
                        text_y = int(min(y_coords))
                        text_h = int(max(y_coords) - min(y_coords))
                        
                        # 勾选框在文字左侧约15像素
                        checkbox_x = text_x - 15
                        checkbox_y = text_y + text_h // 2
                        
                        log(f"  ✓ 通过OCR定位: 文字'{text}'在({text_x}, {text_y})")
                        log(f"  ✓ 计算勾选框位置: ({checkbox_x}, {checkbox_y})")
                        
                        return (checkbox_x, checkbox_y)
            
            log("  ⚠️  未找到协议文字，使用默认坐标")
            return (63, 590)
            
        except Exception as e:
            log(f"  ⚠️  OCR定位失败: {e}，使用默认坐标")
            return (65, 627)
    
    async def _click_agreement_with_retry(self, device_id: str, log_callback=None, max_retries: int = 1) -> bool:
        """智能点击协议勾选框，使用OCR定位
        
        Args:
            device_id: 设备ID
            log_callback: 日志回调
            max_retries: 最大重试次数（默认1次就够）
        
        Returns:
            是否成功勾选
        """
        def log(msg):
            if log_callback:
                log_callback(msg)
        
        log(f"智能勾选协议框（使用OCR定位）")
        
        # 通过OCR定位协议勾选框
        checkbox_x, checkbox_y = await self._find_agreement_checkbox_by_ocr(device_id, log_callback)
        
        log(f"点击勾选框: ({checkbox_x}, {checkbox_y})")
        
        # 直接点击，不需要多次重试
        try:
            await self.adb.tap(device_id, checkbox_x, checkbox_y)
            await asyncio.sleep(0.3)  # 短暂等待UI响应
            log(f"✓ 已点击勾选框")
        except Exception as e:
            log(f"✗ 点击失败: {e}")
        
        return True

    async def login(self, device_id: str, phone: str, password: str, 
                    log_callback=None, use_cache: bool = True) -> LoginResult:
        """执行登录操作（坐标方式 + 页面检测 + OCR + 缓存）
        
        Args:
            device_id: 设备 ID
            phone: 手机号
            password: 密码
            log_callback: 日志回调函数
            use_cache: 是否使用缓存登录（False表示直接执行正常登录）
            
        Returns:
            登录结果
        """
        def log(msg):
            if log_callback:
                log_callback(msg)
        
        try:
            from .page_detector_hybrid import PageState
            
            # 如果 use_cache=False，说明已经确定需要正常登录，直接导航到登录页面
            if not use_cache:
                log("直接执行正常登录流程...")
                
                # 检测当前页面
                page_result = await self.hybrid_detector.detect_page(device_id, use_ocr=True)
                if not page_result or not page_result.state:
                    log("❌ 无法检测当前页面状态")
                    return LoginResult(success=False, error_message="无法检测页面状态")
                
                current_state = page_result.state
                log(f"当前页面: {current_state.value}")
                
                # 如果不在登录页面，导航到登录页面
                if current_state != PageState.LOGIN:
                    log("导航到登录页面...")
                    if not await self.navigate_to_login(device_id, log_callback):
                        return LoginResult(success=False, error_message="无法导航到登录页面")
                
                # 执行正常登录
                result = await self._do_normal_login(device_id, phone, password, log_callback)
                return result
            
            # 以下是 use_cache=True 的逻辑（缓存登录验证）
            
            # P1修复: 统一缓存清除策略 - 在关键节点清除缓存
            # 清理页面检测缓存，确保使用最新的页面状态
            # 避免使用旧的缓存导致误判（比如之前检测到的积分页）
            if hasattr(self.detector, 'clear_cache'):
                self.detector.clear_cache()
            
            # 1. 先检测当前页面状态（使用深度学习检测器）
            log("检测当前登录状态...")
            page_result = await self.detector.detect_page(device_id, use_cache=False)
            if not page_result or not page_result.state:
                log("❌ 无法检测当前页面状态")
                return LoginResult(success=False, error_message="无法检测页面状态")
            
            current_state = page_result.state
            log(f"当前页面: {current_state.value}")
            if hasattr(page_result, 'details') and page_result.details:
                log(f"  检测详情: {page_result.details}")
            
            # 2. 如果已经登录，直接返回成功
            if current_state == PageState.PROFILE_LOGGED:
                log("✓ 已经登录，无需重复登录")
                return LoginResult(success=True, used_cache=True)
            
            # 3. 如果在首页或其他非登录页面，先验证是否已登录
            if current_state in [PageState.HOME, PageState.PROFILE]:
                log(f"检测到在应用内（{current_state.value}），验证登录状态...")
                
                # 导航到个人页面验证登录状态
                from .navigator import Navigator
                navigator = Navigator(self.adb, self.detector)
                
                nav_success = await navigator.navigate_to_profile(device_id)
                if nav_success:
                    # 清除缓存，强制重新检测
                    if hasattr(self.detector, 'clear_cache'):
                        self.detector.clear_cache()
                    
                    # 重新检测页面状态
                    page_result = await self.detector.detect_page(device_id, use_cache=False)
                    current_state = page_result.state
                    log(f"个人页面状态: {current_state.value}")
                    if hasattr(page_result, 'details') and page_result.details:
                        log(f"  检测详情: {page_result.details}")
                    
                    # 如果已登录，直接返回成功
                    if current_state == PageState.PROFILE_LOGGED:
                        log("✓ 已经登录，无需重复登录")
                        return LoginResult(success=True, used_cache=True)
                    
                    # 如果未登录，继续到登录页面
                    if current_state == PageState.PROFILE:
                        log("检测到未登录（个人页_未登录），导航到登录页面...")
                        
                        # P1修复: 导航前清除缓存，确保使用最新的页面状态
                        if hasattr(self.detector, 'clear_cache'):
                            self.detector.clear_cache()
                        
                        # 点击登录入口
                        log(f"点击登录入口坐标: {self.LOGIN_ENTRY}")
                        await self.adb.tap(device_id, self.LOGIN_ENTRY[0], self.LOGIN_ENTRY[1])
                        await asyncio.sleep(2)  # 等待页面跳转
                        
                        # 重新检测页面状态（最多尝试5次，每次等待1秒）
                        for attempt in range(5):
                            # 清除缓存，强制重新检测
                            if hasattr(self.detector, 'clear_cache'):
                                self.detector.clear_cache()
                            
                            # 使用检测器检测页面
                            page_result = await self.detector.detect_page(device_id, use_cache=False)
                            current_state = page_result.state
                            log(f"尝试 {attempt + 1}/5: 当前页面 {current_state.value}")
                            if hasattr(page_result, 'details') and page_result.details:
                                log(f"  检测详情: {page_result.details}")
                            
                            if current_state == PageState.LOGIN:
                                log("✓ 成功进入登录页面")
                                break
                            
                            # 如果检测到弹窗，尝试关闭
                            if current_state == PageState.POPUP:
                                log("检测到弹窗，尝试关闭...")
                                if hasattr(self.detector, 'close_popup'):
                                    await self.detector.close_popup(device_id)
                                await asyncio.sleep(1)
                                continue
                            
                            if attempt < 4:
                                log("未检测到登录页面，等待1秒后重试...")
                                await asyncio.sleep(1)
                        
                        # 如果仍然不在登录页面，尝试再次点击登录入口
                        if current_state != PageState.LOGIN:
                            log("⚠️ 未能进入登录页面，尝试再次点击登录入口...")
                            await self.adb.tap(device_id, self.LOGIN_ENTRY[0], self.LOGIN_ENTRY[1])
                            await asyncio.sleep(2)
                            
                            # 最后一次检测
                            if hasattr(self.detector, 'clear_cache'):
                                self.detector.clear_cache()
                            page_result = await self.detector.detect_page(device_id, use_cache=False)
                            current_state = page_result.state
                            log(f"再次尝试后的页面: {current_state.value}")
                            if hasattr(page_result, 'details') and page_result.details:
                                log(f"  检测详情: {page_result.details}")
            
            # 4. 如果在登录页面，说明未登录或缓存失效，直接执行正常登录
            if current_state == PageState.LOGIN:
                log("当前在登录页面，执行正常登录流程...")
                result = await self._do_normal_login(device_id, phone, password, log_callback)
                
                # 注意：缓存保存已移到GUI主流程中，在获取用户ID后统一保存
                
                return result
            
            # 5. 其他未知状态，返回错误
            log(f"❌ 无法确定登录状态，当前页面: {current_state.value}")
            log(f"提示：请检查是否需要手动处理弹窗或其他干扰")
            return LoginResult(success=False, error_message=f"无法确定登录状态: {current_state.value}")
            
        except Exception as e:
            log(f"登录异常: {e}")
            return LoginResult(success=False, error_message=str(e))
    
    async def _do_normal_login(self, device_id: str, phone: str, password: str,
                               log_callback=None) -> LoginResult:
        """执行正常登录流程（使用混合检测器）
        
        Args:
            device_id: 设备 ID
            phone: 手机号
            password: 密码
            log_callback: 日志回调函数
            
        Returns:
            登录结果
        """
        def log(msg):
            if log_callback:
                log_callback(msg)
        
        try:
            from .page_detector_hybrid import PageState
            
            # 1. 验证当前在登录页面（使用深度学习检测器）
            log("验证当前页面...")
            page_result = await self.detector.detect_page(device_id, use_cache=False)
            if not page_result or not page_result.state:
                log("❌ 无法检测当前页面状态")
                return LoginResult(success=False, error_message="无法检测页面状态")
            
            current_state = page_result.state
            log(f"当前页面: {current_state.value}")
            
            # 如果已经登录，直接跳过登录步骤
            if current_state == PageState.PROFILE_LOGGED:
                log("✓ 检测到已登录状态，跳过登录步骤")
                return LoginResult(success=True, used_cache=True)
            
            # 如果不在登录页面，尝试导航
            if current_state != PageState.LOGIN and current_state != PageState.PROFILE_LOGGED:
                log("不在登录页面，尝试导航...")
                nav_result = await self.navigate_to_login(device_id, log_callback)
                
                if not nav_result:
                    return LoginResult(success=False, error_message="无法导航到登录页面")
                
                # 重新验证（使用深度学习检测器）
                page_result = await self.detector.detect_page(device_id, use_cache=False)
                if not page_result or not page_result.state:
                    log("❌ 导航后无法检测页面状态")
                    return LoginResult(success=False, error_message="无法检测页面状态")
                
                current_state = page_result.state
                log(f"导航后页面状态: {current_state.value}")
                
                # 如果导航后检测到已登录，直接返回成功（不需要再登录）
                if current_state == PageState.PROFILE_LOGGED:
                    log("✓ 导航后检测到已登录状态，跳过登录流程")
                    return LoginResult(success=True, used_cache=True)
                
                # 如果不是登录页面也不是已登录状态，继续尝试（可能是其他页面）
                if current_state != PageState.LOGIN:
                    log(f"⚠ 导航后不在登录页面: {current_state.value}，继续尝试登录流程")
                    # 不要直接返回错误，继续执行登录流程
            
            # 2. 输入账号密码
            # 点击手机号输入框并激活
            log(f"点击手机号输入框...")
            await self._tap_with_fallback(device_id, self.PHONE_INPUT, log_callback)
            await wait_after_action(min_wait=0.2, max_wait=0.5)
            await self._tap_with_fallback(device_id, self.PHONE_INPUT, log_callback)
            await wait_after_action(min_wait=0.3, max_wait=0.8)
            
            # 清空手机号输入框
            log("清空手机号输入框...")
            for _ in range(15):
                await self.adb.key_event(device_id, 67)
                await asyncio.sleep(0.03)
            
            # 输入手机号
            log(f"输入手机号: {phone}")
            await self.adb.input_text(device_id, phone)
            await wait_after_action(min_wait=0.5, max_wait=1.5)
            
            # 点击密码输入框并激活
            log(f"点击密码输入框...")
            await self._tap_with_fallback(device_id, self.PASSWORD_INPUT, log_callback)
            await wait_after_action(min_wait=0.2, max_wait=0.5)
            await self._tap_with_fallback(device_id, self.PASSWORD_INPUT, log_callback)
            await wait_after_action(min_wait=0.3, max_wait=0.8)
            
            # 清空密码输入框
            log("清空密码输入框...")
            for _ in range(15):
                await self.adb.key_event(device_id, 67)
                await asyncio.sleep(0.03)
            
            # 输入密码
            log(f"输入密码: {'*' * len(password)}")
            await self.adb.input_text(device_id, password)
            await wait_after_action(min_wait=0.5, max_wait=1.5)
            
            # 3. 勾选协议并点击登录按钮
            log("勾选协议并点击登录按钮...")
            
            # 使用智能勾选方法（带重试和多坐标）
            log(f"智能勾选协议框...")
            await self._click_agreement_with_retry(device_id, log_callback, max_retries=3)
            await wait_after_action(min_wait=0.5, max_wait=1.0)
            
            # 点击登录按钮
            log(f"点击登录按钮...")
            await self._tap_with_fallback(device_id, self.LOGIN_BUTTON, log_callback)
            await wait_after_action(min_wait=0.5, max_wait=1.5)
            
            # 4. 等待登录完成，使用像素检测（不用OCR）
            log("等待登录完成...")
            
            # 使用快速检测，每0.5秒检测一次，最多10秒
            for i in range(20):  # 20次 * 0.5秒 = 10秒
                await asyncio.sleep(0.5)
                
                # 使用像素检测当前页面状态
                page_result = await self.page_detector.detect_page(device_id)
                current_state = page_result.state
                
                if i % 2 == 0:  # 每1秒打印一次日志
                    log(f"检测页面: {current_state.value} ({i//2 + 1}秒)")
                
                # 检测到错误弹窗（像素检测）
                if current_state == PageState.LOGIN_ERROR:
                    log("检测到登录错误弹窗")
                    
                    # 获取错误信息
                    error_msg = page_result.details if hasattr(page_result, 'details') and page_result.details else "未知错误"
                    log(f"错误信息: {error_msg}")
                    
                    # 关闭错误弹窗
                    if hasattr(self.detector, 'close_popup'):
                        await self.detector.close_popup(device_id)
                    await wait_after_action(min_wait=0.3, max_wait=1.0)
                    
                    # 根据错误信息返回具体错误类型
                    if "手机号不存在" in error_msg:
                        return LoginResult(
                            success=False, 
                            error_message="登录失败：手机号不存在",
                            error_type=ErrorType.LOGIN_PHONE_NOT_EXIST
                        )
                    elif "密码错误" in error_msg:
                        return LoginResult(
                            success=False, 
                            error_message="登录失败：密码错误",
                            error_type=ErrorType.LOGIN_PASSWORD_ERROR
                        )
                    else:
                        return LoginResult(
                            success=False, 
                            error_message=f"登录失败：{error_msg}",
                            error_type=None
                        )
                
                # 处理登录后可能出现的弹窗
                if current_state == PageState.POPUP:
                    log("检测到弹窗，尝试关闭...")
                    if hasattr(self.detector, 'close_popup'):
                        await self.detector.close_popup(device_id)
                    await wait_after_action(min_wait=0.5, max_wait=1.5)
                    continue
                
                # 如果不再是登录页面，说明登录成功
                if current_state != PageState.LOGIN:
                    if current_state in [PageState.HOME, PageState.PROFILE, PageState.PROFILE_LOGGED]:
                        log("✓ 登录成功！")
                        return LoginResult(success=True)
                    
                    # 可能跳转到其他页面，尝试处理
                    if current_state == PageState.UNKNOWN:
                        log("登录后跳转到未知页面，尝试返回...")
                        await self.adb.press_back(device_id)
                        await wait_after_action(min_wait=0.5, max_wait=1.5)
                        continue
                    
                    # 其他已知页面状态，认为登录成功
                    log(f"✓ 登录成功，当前页面: {current_state.value}")
                    return LoginResult(success=True)
            
            # 5. 超时后再次检测最终状态
            log("登录等待超时，检测最终状态...")
            page_result = await self.detector.detect_page(device_id, use_cache=False)
            current_state = page_result.state
            
            if current_state in [PageState.HOME, PageState.PROFILE, PageState.PROFILE_LOGGED]:
                log("✓ 登录成功！")
                return LoginResult(success=True)
            elif current_state == PageState.LOGIN_ERROR:
                # 处理错误弹窗
                error_msg = page_result.details if hasattr(page_result, 'details') and page_result.details else "未知错误"
                log(f"检测到登录错误: {error_msg}")
                if hasattr(self.detector, 'close_popup'):
                    await self.detector.close_popup(device_id)
                return LoginResult(
                    success=False, 
                    error_message=f"登录失败：{error_msg}",
                    error_type=None  # 未知错误，不设置特定类型
                )
            elif current_state == PageState.LOGIN:
                log("仍在登录页面，登录可能失败")
                return LoginResult(success=False, error_message="登录超时，仍在登录页面")
            else:
                # 不在登录页面，认为登录成功
                log(f"✓ 登录完成，当前页面: {current_state.value}")
                return LoginResult(success=True)
            
        except Exception as e:
            log(f"登录异常: {e}")
            return LoginResult(success=False, error_message=str(e))

    async def navigate_to_login(self, device_id: str, log_callback=None) -> bool:
        """导航到登录页面（使用深度学习检测器）
        
        Args:
            device_id: 设备 ID
            log_callback: 日志回调函数
            
        Returns:
            是否成功
        """
        def log(msg):
            if log_callback:
                log_callback(msg)
        
        try:
            from .page_detector_hybrid import PageState
            
            # 1. 先用深度学习检测器验证当前页面
            log("检测当前页面...")
            page_result = await self.detector.detect_page(device_id, use_cache=False)
            current_state = page_result.state
            log(f"当前页面: {current_state.value}")
            
            # 如果已经在登录页面
            if current_state == PageState.LOGIN:
                log("已在登录页面")
                return True
            
            # 2. 处理弹窗（如果有）
            if current_state == PageState.POPUP:
                log("检测到弹窗，尝试关闭...")
                if hasattr(self.detector, 'close_popup'):
                    await self.detector.close_popup(device_id)
                await wait_after_action(min_wait=0.5, max_wait=1.5)
            
            # 3. 点击"我的"标签
            log("点击'我的'标签...")
            await self.adb.tap(device_id, self.TAB_MY[0], self.TAB_MY[1])
            await wait_after_action(min_wait=0.5, max_wait=2.0)
            
            # 4. 点击登录入口
            log("点击登录入口...")
            await self.adb.tap(device_id, self.LOGIN_ENTRY[0], self.LOGIN_ENTRY[1])
            await wait_after_action(min_wait=1.5, max_wait=3.0)
            
            # 5. 验证是否到达登录页面（使用深度学习检测器）
            for attempt in range(8):  # 尝试8次，每次间隔1秒 = 最多8秒等待
                page_result = await self.detector.detect_page(device_id, use_cache=False)
                current_state = page_result.state
                log(f"检测登录页面: {current_state.value} (第{attempt+1}次)")
                
                # 显示详细信息
                if hasattr(page_result, 'details') and page_result.details:
                    log(f"  详情: {page_result.details}")
                
                # 检测到登录页面
                if current_state == PageState.LOGIN:
                    log("✓ 成功进入登录页面")
                    return True
                
                # 如果检测到已登录个人页，说明账号已经登录，不需要再进入登录页面
                if current_state == PageState.PROFILE_LOGGED:
                    log("✓ 检测到已登录状态，无需进入登录页面")
                    return True
                
                # 如果检测到模拟器桌面，说明应用崩溃了，立即返回失败
                if current_state == PageState.LAUNCHER:
                    log(f"❌ 检测到模拟器桌面，应用可能已崩溃")
                    if hasattr(page_result, 'details') and page_result.details:
                        log(f"  详情: {page_result.details}")
                    return False
                
                # 如果是加载中状态（白屏/灰屏），继续等待
                if current_state == PageState.LOADING:
                    if attempt < 7:
                        log(f"  页面加载中，继续等待...")
                        await asyncio.sleep(1)
                        continue
                
                # 如果是其他状态，也等待后重试
                if attempt < 7:
                    log(f"  等待1秒后重试...")
                    await asyncio.sleep(1)
            
            # 最后一次检测仍然失败
            log(f"❌ 未能进入登录页面，当前: {current_state.value}")
            if hasattr(page_result, 'details') and page_result.details:
                log(f"  最后检测详情: {page_result.details}")
            return False
            
        except Exception as e:
            log(f"导航失败: {e}")
            return False

    async def logout(self, device_id: str, log_callback=None) -> bool:
        """退出登录
        
        Args:
            device_id: 设备 ID
            log_callback: 日志回调函数
            
        Returns:
            是否成功退出
        """
        # 简单实现：强制停止应用即可
        return True
    
    async def is_logged_in(self, device_id: str) -> bool:
        """检测是否已登录
        
        Args:
            device_id: 设备 ID
            
        Returns:
            是否已登录
        """
        result = await self.page_detector.detect_page(device_id)
        return result.state == PageState.PROFILE_LOGGED
