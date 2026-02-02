"""
页面导航模块 - 处理应用内页面跳转
Navigator Module - Handle in-app page navigation
"""

import asyncio
import time
from typing import Optional
from .adb_bridge import ADBBridge
from .page_detector_hybrid import PageDetectorHybrid, PageState
from .logger import get_silent_logger
from .timeouts_config import TimeoutsConfig


class Navigator:
    """页面导航器"""
    
    # 底部导航栏坐标 (540x960) - MuMu模拟器手动验证坐标（v1.7.1）
    TAB_HOME = (90, 920)          # "首页" 标签（MuMu验证）
    TAB_CATEGORY = (200, 920)     # "分类" 标签
    TAB_CART = (330, 920)         # "购物车" 标签
    TAB_MY = (450, 920)           # "我的" 标签（MuMu验证）
    
    def __init__(self, adb: ADBBridge, detector=None):
        """初始化导航器
        
        Args:
            adb: ADB桥接对象
            detector: 页面检测器（可选，如果不提供则从ModelManager获取）
        """
        self.adb = adb
        
        # 如果没有提供检测器，从ModelManager获取共享的整合检测器
        if detector is None:
            from .model_manager import ModelManager
            model_manager = ModelManager.get_instance()
            self.detector = model_manager.get_page_detector_integrated()
            print(f"[Navigator] 从ModelManager获取共享的整合检测器")
        else:
            # 使用传入的检测器（应该是从ModelManager获取的共享实例）
            self.detector = detector
            print(f"[Navigator] 使用传入的检测器: {type(detector).__name__}")
        
        # 初始化静默日志记录器
        self._silent_log = get_silent_logger()
        
        # 初始化页面状态守卫
        from .page_state_guard import PageStateGuard
        self.guard = PageStateGuard(adb, self.detector)
        
        # 初始化页面检测缓存管理器
        from .page_detector_cache import PageDetectorCache
        self._page_cache = PageDetectorCache(
            default_ttl=1.0,  # 导航流程中页面变化较慢，使用1秒缓存
            max_cache_size=50
        )
        
        # 从ModelManager获取OCR线程池
        from .model_manager import ModelManager
        model_manager = ModelManager.get_instance()
        self.ocr_pool = model_manager.get_ocr_thread_pool()
        
        # 初始化屏幕截图
        from .screen_capture import ScreenCapture
        self.screen_capture = ScreenCapture(adb)
        
        # 不再需要创建SmartWaiter实例，直接使用全局函数
    
    async def _detect_page_cached(self, device_id: str, use_cache: bool = True,
                                  detect_elements: bool = False,
                                  cache_key: str = "default",
                                  ttl: Optional[float] = None,
                                  use_ocr: bool = False,
                                  use_template: bool = True) -> Optional[any]:
        """使用缓存的页面检测
        
        这是一个便捷方法，封装了页面检测缓存的使用逻辑
        
        Args:
            device_id: 设备ID
            use_cache: 是否使用缓存
            detect_elements: 是否检测元素
            cache_key: 缓存键（用于区分不同类型的检测）
            ttl: 缓存生存时间（秒），None表示使用默认值
            use_ocr: 是否使用OCR
            use_template: 是否使用模板匹配
            
        Returns:
            页面检测结果
        """
        # 如果不使用缓存，直接检测并失效旧缓存
        if not use_cache:
            self._page_cache.invalidate(device_id, cache_key)
            result = await self.detector.detect_page(
                device_id, 
                use_cache=False, 
                detect_elements=detect_elements,
                use_ocr=use_ocr,
                use_template=use_template
            )
            return result
        
        # 尝试从缓存获取
        cached_result = self._page_cache.get(device_id, cache_key)
        if cached_result is not None:
            return cached_result
        
        # 缓存未命中，执行检测
        result = await self.detector.detect_page(
            device_id, 
            use_cache=False, 
            detect_elements=detect_elements,
            use_ocr=use_ocr,
            use_template=use_template
        )
        
        # 更新缓存
        if result is not None:
            self._page_cache.set(device_id, result, cache_key, ttl)
        
        return result
    
    async def _find_my_button_by_ocr(self, device_id: str) -> Optional[tuple]:
        """使用 OCR 识别"我的"按钮位置
        
        Args:
            device_id: 设备ID
            
        Returns:
            Optional[tuple]: 按钮中心坐标 (x, y)，失败返回 None
        """
        try:
            # 截图
            screenshot_np = await self.screen_capture.capture(device_id)
            if screenshot_np is None:
                print(f"  [OCR识别] ❌ 截图失败")
                return None
            
            # 转换为 PIL Image
            from PIL import Image
            screenshot = Image.fromarray(screenshot_np)
            print(f"  [OCR识别] ✓ 截图成功: {screenshot.width}x{screenshot.height}")
            
            # OCR 识别
            ocr_result = await self.ocr_pool.recognize(screenshot, timeout=5.0)
            
            if not ocr_result.texts:
                print(f"  [OCR识别] ❌ OCR 未识别到任何文本")
                return None
            
            print(f"  [OCR识别] ✓ OCR 识别到 {len(ocr_result.texts)} 个文本")
            
            # 查找"我的"按钮（底部导航栏区域 y > 850）
            found_my_buttons = []
            for text, box, confidence in zip(ocr_result.texts, ocr_result.boxes, ocr_result.scores):
                if '我的' in text:
                    # 计算中心点
                    x_coords = [point[0] for point in box]
                    y_coords = [point[1] for point in box]
                    center_x = int(sum(x_coords) / len(x_coords))
                    center_y = int(sum(y_coords) / len(y_coords))
                    
                    print(f"  [OCR识别] 找到'我的'文本: ({center_x}, {center_y}), 置信度: {confidence:.2f}")
                    
                    # 确保在底部导航栏区域
                    if center_y > 850:
                        print(f"  [OCR识别] ✓ 位置在底部导航栏区域，使用此坐标")
                        return (center_x, center_y)
                    else:
                        print(f"  [OCR识别] ⚠️ 位置不在底部导航栏区域 (y={center_y} <= 850)，跳过")
                        found_my_buttons.append((center_x, center_y, confidence))
            
            if found_my_buttons:
                print(f"  [OCR识别] ⚠️ 找到 {len(found_my_buttons)} 个'我的'文本，但都不在底部导航栏区域")
                for x, y, conf in found_my_buttons:
                    print(f"  [OCR识别]   - ({x}, {y}), 置信度: {conf:.2f}")
            else:
                print(f"  [OCR识别] ❌ 未找到'我的'文本")
            
            return None
            
        except Exception as e:
            print(f"  [OCR识别] ❌ 识别'我的'按钮失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def navigate_to_home(self, device_id: str, max_attempts: int = 3) -> bool:
        """导航到首页（优化版：使用优先级检测）
        
        优化点：
        1. 减少最大尝试次数从5次到3次
        2. 使用优先级模板检测
        3. 优化智能等待器参数
        
        Args:
            device_id: 设备ID
            max_attempts: 最大尝试次数（默认3次）
            
        Returns:
            bool: 是否成功到达首页
        """
        # 定义导航到首页过程中最可能出现的页面
        expected_pages = [
            '首页.png',              # 最可能
            '签到页.png',            # 可能需要返回
            '已登陆个人页.png',      # 可能
        ]
        
        for attempt in range(max_attempts):
            # 使用优先级模板检测
            page_result = await self.detector.detect_page_with_priority(
                device_id,
                expected_pages,
                use_cache=True
            )
            if not page_result or not page_result.state:
                self._silent_log.info(f"[导航到首页] ⚠️ 无法检测页面状态，重试...")
                await asyncio.sleep(0.5)
                continue
            
            current_state = page_result.state
            
            # 已经在首页
            if current_state == PageState.HOME:
                return True
            
            # 如果是个人页广告，使用YOLO关闭广告
            if current_state == PageState.PROFILE_AD:
                # 方法1: 使用YOLO检测"确认按钮"
                self._silent_log.log(f"[导航到首页] 检测到个人页广告，使用YOLO检测确认按钮...")
                close_button_pos = await self.detector.find_button_yolo(
                    device_id, 
                    '个人页广告',
                    '确认按钮',
                    conf_threshold=0.5
                )
                
                if close_button_pos:
                    self._silent_log.log(f"[导航到首页] YOLO检测到确认按钮: {close_button_pos}，点击关闭")
                    await self.adb.tap(device_id, close_button_pos[0], close_button_pos[1])
                    
                    # 使用智能等待器等待页面变化
                    from .performance.smart_waiter import wait_for_page
                    wait_result = await wait_for_page(
                        device_id,
                        self.detector,
                        [PageState.HOME, PageState.PROFILE, PageState.PROFILE_LOGGED],
                        timeout=TimeoutsConfig.SMART_WAIT_TIMEOUT,
                        log_callback=lambda msg: self._silent_log.log(f"  [智能等待] {msg}")
                    )
                    
                    # 清除缓存，重新检测
                    self.detector.clear_cache()
                    continue
                else:
                    # 方法2: 使用返回键关闭（更可靠）
                    self._silent_log.log(f"[导航到首页] YOLO未检测到按钮，使用返回键关闭")
                    await self.adb.press_back(device_id)
                    
                    # 使用智能等待器等待页面变化
                    from .performance.smart_waiter import wait_for_page
                    wait_result = await wait_for_page(
                        device_id,
                        self.detector,
                        [PageState.HOME, PageState.PROFILE, PageState.PROFILE_LOGGED],
                        timeout=TimeoutsConfig.SMART_WAIT_TIMEOUT,
                        log_callback=lambda msg: self._silent_log.log(f"  [智能等待] {msg}")
                    )
                    
                    # 清除缓存，重新检测
                    self.detector.clear_cache()
                    if success:
                        # 使用智能等待器等待页面变化（替换固定等待1秒）
                        from .performance.smart_waiter import wait_for_page
                        wait_result = await wait_for_page(
                            device_id,
                            self.detector,
                            [PageState.HOME, PageState.PROFILE, PageState.PROFILE_LOGGED],
                            timeout=TimeoutsConfig.SMART_WAIT_TIMEOUT,
                            log_callback=lambda msg: self._silent_log.log(f"  [智能等待] {msg}")
                        )
                        
                        if wait_result:
                            self._silent_log.info(f"[导航到首页] ✓ 广告已关闭")
                        
                        self.detector.clear_cache()
                        continue
                    else:
                        self._silent_log.info(f"[导航到首页] ⚠️ 无法关闭广告")
                        continue
            
            # 如果在广告页，等待自动跳过
            if current_state == PageState.AD:
                self._silent_log.info(f"[导航到首页] 当前在广告页，等待自动跳过...")
                # 使用全局便捷函数等待页面变化
                from .performance.smart_waiter import wait_for_page
                result = await wait_for_page(
                    device_id,
                    self.detector,
                    [PageState.HOME],  # 期望到达首页
                    log_callback=lambda msg: print(f"  [智能等待] {msg}")
                )
                if result and result.state == PageState.HOME:
                    self._silent_log.info(f"[导航到首页] ✓ 广告已跳过，到达首页")
                    return True
                else:
                    self._silent_log.info(f"[导航到首页] ⚠️ 广告跳过超时或到达其他页面")
                    # 清除缓存，重新检测
                    self.detector.clear_cache()
                    continue
            
            # 如果在签到页面，按返回键
            if current_state == PageState.CHECKIN:
                self._silent_log.info(f"[导航到首页] 当前在签到页面，按返回键...")
                await self.adb.press_back(device_id)
                # 使用全局便捷函数等待页面变化
                from .performance.smart_waiter import wait_for_page
                result = await wait_for_page(
                    device_id,
                    self.detector,
                    [PageState.HOME],
                    log_callback=lambda msg: print(f"  [智能等待] {msg}")
                )
                if not result:
                    self._silent_log.info(f"[导航到首页] ⚠️ 等待返回首页超时")
                continue
            
            # 使用守卫处理异常页面
            if current_state in [PageState.POPUP, PageState.UNKNOWN]:
                handled = await self.guard._handle_unexpected_page(device_id, current_state, PageState.HOME, "导航到首页")
                if handled:
                    await asyncio.sleep(1)
                    continue
            
            # 在其他已知页面,点击首页标签
            # 优先使用YOLO检测"首页"按钮位置（更准确）
            self._silent_log.log(f"[导航到首页] 使用YOLO检测'首页'按钮位置...")
            
            # 根据当前页面选择合适的模型
            if current_state in [PageState.PROFILE, PageState.PROFILE_LOGGED]:
                # 从个人页跳转，使用avatar_homepage模型
                home_button_pos = await self.detector.find_button_yolo(
                    device_id, 
                    'avatar_homepage',  # 个人页模型有"首页按钮"
                    '首页按钮',
                    conf_threshold=0.5
                )
            else:
                # 从其他页面跳转，尝试分类页模型
                home_button_pos = await self.detector.find_button_yolo(
                    device_id, 
                    '分类页',  # 分类页模型有"首页按钮"
                    '首页按钮',
                    conf_threshold=0.5
                )
            
            # 如果YOLO检测成功，使用检测到的坐标；否则使用固定坐标
            if home_button_pos:
                self._silent_log.log(f"[导航到首页] YOLO检测到'首页'按钮: {home_button_pos}")
                await self.adb.tap(device_id, home_button_pos[0], home_button_pos[1])
            else:
                self._silent_log.log(f"[导航到首页] YOLO未检测到按钮，使用固定坐标: {self.TAB_HOME}")
                await self.adb.tap(device_id, self.TAB_HOME[0], self.TAB_HOME[1])
            
            # 使用全局便捷函数等待页面变化
            from .performance.smart_waiter import wait_for_page
            result = await wait_for_page(
                device_id,
                self.detector,
                [PageState.HOME],
                log_callback=lambda msg: print(f"  [智能等待] {msg}")
            )
            if result:
                self._silent_log.info(f"[导航到首页] ✓ 成功到达首页")
                return True
        
        # 最后确认
        page_result = await self.detector.detect_page_with_priority(
            device_id,
            ['首页.png'],
            use_cache=False
        )
        if not page_result or not page_result.state:
            return False
        return page_result.state == PageState.HOME
    
    async def navigate_to_profile(self, device_id: str, max_attempts: int = 3) -> bool:
        """导航到我的页面（重写版：高频检测，不使用智能等待器）
        
        核心优化：
        1. 点击"我的"按钮后，立即开始高频检测（每0.05秒）
        2. 一旦检测到广告，立即关闭（不等待应用完全加载）
        3. 不使用智能等待器（避免等待5-7秒）
        
        Args:
            device_id: 设备ID
            max_attempts: 最大尝试次数（默认3次）
            
        Returns:
            bool: 是否成功到达我的页面
        """
        self._silent_log.info(f"[导航到我的页面] 开始导航，最多尝试 {max_attempts} 次")
        
        for attempt in range(max_attempts):
            self._silent_log.info(f"[导航到我的页面] 尝试 {attempt + 1}/{max_attempts}")
            
            # 快速检测当前页面状态（使用缓存）
            page_result = await self._detect_page_cached(
                device_id, 
                use_cache=True, 
                detect_elements=False,
                cache_key=f"profile_nav_{attempt}",
                ttl=0.5  # 导航过程中页面变化较快，使用较短的缓存时间
            )
            
            if not page_result or not page_result.state:
                self._silent_log.info(f"[导航到我的页面] ⚠️ 无法检测页面状态，重试...")
                await asyncio.sleep(0.5)
                continue
            
            current_state = page_result.state
            self._silent_log.info(f"[导航到我的页面] 当前页面: {current_state.value}")
            
            # 已经在我的页面
            if current_state in [PageState.PROFILE, PageState.PROFILE_LOGGED]:
                self._silent_log.info(f"[导航到我的页面] ✓ 已在我的页面")
                return True
            
            # 如果是个人页广告，立即关闭
            if current_state == PageState.PROFILE_AD:
                self._silent_log.info(f"[导航到我的页面] ⚠️ 检测到个人页广告，立即关闭...")
                
                # 方法1: 使用YOLO检测关闭按钮
                close_button_pos = await self.detector.find_button_yolo(
                    device_id, 
                    '个人页广告',
                    '确认按钮',
                    conf_threshold=0.5
                )
                
                if close_button_pos:
                    self._silent_log.info(f"[导航到我的页面] YOLO检测到关闭按钮: {close_button_pos}")
                    await self.adb.tap(device_id, close_button_pos[0], close_button_pos[1])
                    await asyncio.sleep(TimeoutsConfig.WAIT_SHORT)
                else:
                    # 方法2: 使用返回键关闭（更可靠）
                    self._silent_log.info(f"[导航到我的页面] YOLO未检测到按钮，使用返回键关闭")
                    await self.adb.press_back(device_id)
                    await asyncio.sleep(TimeoutsConfig.WAIT_SHORT)
                
                self.detector.clear_cache()
                
                # 验证广告是否已关闭
                self._page_cache.invalidate(device_id, f"profile_nav_{attempt}")
                page_result = await self._detect_page_cached(
                    device_id, 
                    use_cache=False, 
                    detect_elements=False,
                    cache_key=f"after_ad_close_{attempt}"
                )
                if page_result and page_result.state in [PageState.PROFILE, PageState.PROFILE_LOGGED]:
                    self._silent_log.info(f"[导航到我的页面] ✓ 广告已关闭，到达个人页")
                    return True
                else:
                    self._silent_log.info(f"[导航到我的页面] ⚠️ 广告关闭后状态: {page_result.state.value if page_result else 'unknown'}，继续尝试...")
                    continue
            
            # 如果在签到页面，先返回首页
            if current_state == PageState.CHECKIN:
                self._silent_log.info(f"[导航到我的页面] 当前在签到页面，先返回首页...")
                await self.adb.press_back(device_id)
                # 使用全局便捷函数等待页面变化
                from .performance.smart_waiter import wait_for_page
                result = await wait_for_page(
                    device_id,
                    self.detector,
                    [PageState.HOME],
                    log_callback=lambda msg: print(f"  [智能等待] {msg}")
                )
                if not result:
                    self._silent_log.info(f"[导航到我的页面] ⚠️ 等待返回首页超时")
                continue
            
            # 如果在积分页，需要按2次返回键到个人页（登录后常见情况）
            if current_state == PageState.POINTS_PAGE:
                self._silent_log.info(f"[导航到我的页面] 当前在积分页，需要按2次返回键...")
                
                # 第1次返回键
                await self.adb.press_back(device_id)
                
                # 使用智能等待器等待页面变化
                from .performance.smart_waiter import wait_for_page
                result = await wait_for_page(
                    device_id,
                    self.detector,
                    [PageState.PROFILE_LOGGED, PageState.PROFILE, PageState.POINTS_PAGE],
                    log_callback=lambda msg: self._silent_log.log(f"  [智能等待] {msg}")
                )
                
                # 第2次返回键
                await self.adb.press_back(device_id)
                
                # 使用智能等待器等待页面变化到个人页
                result = await wait_for_page(
                    device_id,
                    self.detector,
                    [PageState.PROFILE_LOGGED, PageState.PROFILE],
                    log_callback=lambda msg: self._silent_log.log(f"  [智能等待] {msg}")
                )
                
                # 清理缓存（页面已改变）
                self.detector.clear_cache()
                
                # 按返回键后检测页面状态（最多重试3次）
                # 重要：必须包含积分页模板，否则会误判！
                check_templates = ['已登陆个人页.png', '未登陆个人页.png', '积分页.png', '模拟器桌面.png']
                
                success = False
                for retry in range(3):
                    # 使用完整检测（模板+OCR），更准确判断页面状态（使用缓存）
                    result = await self._detect_page_cached(
                        device_id, 
                        use_cache=True,
                        cache_key=f"points_back_{retry}",
                        ttl=0.5,
                        use_ocr=True,
                        use_template=True
                    )
                    
                    if result:
                        self._silent_log.info(f"[导航到我的页面] 检测结果: {result.state.value} - {result.details}")
                        
                        if result.state == PageState.PROFILE_LOGGED:
                            self._silent_log.info(f"[导航到我的页面] ✓ 已返回到个人页")
                            success = True
                            break
                        elif result.state == PageState.POINTS_PAGE:
                            # 仍然在积分页，说明页面还没加载完或者需要更多返回键
                            self._silent_log.info(f"[导航到我的页面] ⚠️ 仍在积分页，再按一次返回键...")
                            await self.adb.press_back(device_id)
                            # 使用智能等待器等待页面变化
                            from .performance.smart_waiter import wait_for_page
                            wait_result = await wait_for_page(
                                device_id,
                                self.detector,
                                [PageState.PROFILE_LOGGED, PageState.PROFILE],
                                log_callback=lambda msg: self._silent_log.log(f"  [智能等待] {msg}")
                            )
                            self.detector.clear_cache()
                            # 不break，继续重试检测
                        elif result.state == PageState.LAUNCHER:
                            self._silent_log.info(f"[导航到我的页面] ❌ 检测到桌面，应用已退出！尝试重启应用...")
                            # 强制停止应用
                            await self.adb.stop_app(device_id, "com.xmwl.shop")
                            await asyncio.sleep(1)
                            # 重新启动应用
                            await self.adb.start_app(device_id, "com.xmwl.shop")
                            await asyncio.sleep(5)
                            self._silent_log.info(f"[导航到我的页面] ✓ 应用已重新启动")
                            # 清理缓存（应用已重启）
                            self.detector.clear_cache()
                            # 重启后跳出，让外层循环重新检测
                            break
                        else:
                            # 其他状态，可能是个人页但模板没匹配上，用OCR再确认
                            if "个人" in result.details or "我的" in result.details or "余额" in result.details:
                                self._silent_log.info(f"[导航到我的页面] ✓ OCR确认已在个人页（{result.details}）")
                                success = True
                                break
                            
                            if retry < 2:
                                self._silent_log.info(f"[导航到我的页面] ⚠️ 返回后页面状态: {result.state.value}，等待后重试...")
                                await asyncio.sleep(1)  # 缩短等待时间从3秒到1秒
                            else:
                                self._silent_log.info(f"[导航到我的页面] ⚠️ 返回后页面状态: {result.state.value}")
                    else:
                        if retry < 2:
                            self._silent_log.info(f"[导航到我的页面] ⚠️ 无法检测页面状态，等待后重试...")
                            await asyncio.sleep(1)  # 缩短等待时间从3秒到1秒
                        else:
                            self._silent_log.info(f"[导航到我的页面] ⚠️ 无法检测页面状态")
                
                # 如果成功返回到个人页，直接返回成功
                if success:
                    return True
                
                # 否则继续外层循环（重新检测页面状态）
                continue
            
            # 使用守卫处理异常页面（弹窗、未知页面等）
            if current_state in [PageState.POPUP, PageState.UNKNOWN, PageState.LOADING]:
                self._silent_log.info(f"[导航到我的页面] 检测到异常页面: {current_state.value}，尝试处理...")
                
                # 如果是弹窗，关闭它
                if current_state == PageState.POPUP:
                    self._silent_log.info(f"[导航到我的页面] 关闭弹窗...")
                    await self.detector.close_popup(device_id)
                    await asyncio.sleep(0.5)
                    continue
                
                # 如果是加载页面，等待加载完成
                if current_state == PageState.LOADING:
                    self._silent_log.info(f"[导航到我的页面] 页面加载中，智能等待...")
                    # 使用全局便捷函数等待页面变化
                    from .performance.smart_waiter import wait_for_page
                    result = await wait_for_page(
                        device_id,
                        self.detector,
                        [PageState.HOME, PageState.PROFILE, PageState.PROFILE_LOGGED],
                        log_callback=lambda msg: print(f"  [智能等待] {msg}")
                    )
                    if not result:
                        self._silent_log.info(f"[导航到我的页面] ⚠️ 等待加载完成超时")
                    continue
                
                # 如果是未知页面，尝试按返回键回到首页
                if current_state == PageState.UNKNOWN:
                    # 检查是否是需要返回的异常页面
                    if any(keyword in page_result.details for keyword in ["异常页面", "商品", "活动", "文章"]):
                        self._silent_log.info(f"[导航到我的页面] 检测到异常页面，按返回键回到首页...")
                        await self.adb.press_back(device_id)
                        # 使用全局便捷函数等待页面变化
                        from .performance.smart_waiter import wait_for_page
                        result = await wait_for_page(
                            device_id,
                            self.detector,
                            [PageState.HOME],
                            log_callback=lambda msg: print(f"  [智能等待] {msg}")
                        )
                        continue
                    else:
                        self._silent_log.info(f"[导航到我的页面] 未知页面类型: {page_result.details}，尝试返回首页...")
                        await self.adb.press_back(device_id)
                        # 使用全局便捷函数等待页面变化
                        from .performance.smart_waiter import wait_for_page
                        result = await wait_for_page(
                            device_id,
                            self.detector,
                            [PageState.HOME],
                            log_callback=lambda msg: print(f"  [智能等待] {msg}")
                        )
                        continue
            
            # 确保在首页或有底部导航栏的页面
            if current_state not in [PageState.HOME]:
                self._silent_log.info(f"[导航到我的页面] 当前不在首页，先返回首页...")
                # 尝试返回首页（减少尝试次数）
                success = await self.navigate_to_home(device_id, max_attempts=2)
                if not success:
                    self._silent_log.info(f"[导航到我的页面] ⚠️ navigate_to_home 返回失败，再次检测页面状态...")
                    # 再次检测，可能实际上已经在首页了
                    recheck_result = await self.detector.detect_page_with_priority(
                        device_id,
                        ['首页.png'],
                        use_cache=False
                    )
                    if recheck_result and recheck_result.state == PageState.HOME:
                        self._silent_log.info(f"[导航到我的页面] ✓ 实际上已经在首页，继续...")
                    else:
                        self._silent_log.info(f"[导航到我的页面] ✗ 确实不在首页: {recheck_result.state.value if recheck_result else 'unknown'}")
                        continue
                await asyncio.sleep(0.5)
            
            # 优先使用YOLO检测"我的"按钮位置（更准确）
            self._silent_log.log(f"[导航到我的页面] 使用YOLO检测'我的'按钮位置...")
            my_button_pos = await self.detector.find_button_yolo(
                device_id, 
                'homepage',  # 首页模型
                '我的按钮',
                conf_threshold=0.5
            )
            
            # 如果YOLO检测成功，使用检测到的坐标；否则使用固定坐标
            if my_button_pos:
                self._silent_log.log(f"[导航到我的页面] YOLO检测到'我的'按钮: {my_button_pos}")
                await self.adb.tap(device_id, my_button_pos[0], my_button_pos[1])
            else:
                self._silent_log.log(f"[导航到我的页面] YOLO未检测到按钮，使用固定坐标: {self.TAB_MY}")
                await self.adb.tap(device_id, self.TAB_MY[0], self.TAB_MY[1])
            
            # 点击后立即清除缓存
            self.detector.clear_cache(device_id)
            
            # ===== 优化：点击后等待页面加载，然后立即返回让调用者验证 =====
            # 等待0.5秒让页面开始加载
            await asyncio.sleep(0.5)
            
            # 快速检测一次当前页面状态
            page_result = await self.detector.detect_page(device_id, use_cache=False, detect_elements=False)
            
            if page_result:
                self._silent_log.info(f"[导航到我的页面] 点击后页面状态: {page_result.state.value}")
                
                # 如果检测到个人页广告，立即关闭
                if page_result.state == PageState.PROFILE_AD:
                    self._silent_log.info(f"[导航到我的页面] ⚠️ 检测到个人页广告，立即关闭...")
                    
                    # 方法1: 使用YOLO检测关闭按钮
                    close_button_pos = await self.detector.find_button_yolo(
                        device_id, 
                        '个人页广告',
                        '确认按钮',
                        conf_threshold=0.5
                    )
                    
                    if close_button_pos:
                        self._silent_log.info(f"[导航到我的页面] YOLO检测到关闭按钮: {close_button_pos}")
                        await self.adb.tap(device_id, close_button_pos[0], close_button_pos[1])
                        await asyncio.sleep(0.5)
                    else:
                        # 方法2: 使用返回键关闭（更可靠）
                        self._silent_log.info(f"[导航到我的页面] YOLO未检测到按钮，使用返回键关闭")
                        await self.adb.press_back(device_id)
                        await asyncio.sleep(0.5)
                    
                    self.detector.clear_cache(device_id)
                    
                    # 再次检测页面状态
                    page_result = await self.detector.detect_page(device_id, use_cache=False, detect_elements=False)
                    if page_result:
                        self._silent_log.info(f"[导航到我的页面] 广告关闭后页面状态: {page_result.state.value}")
                
                # 如果已经在个人页，直接返回成功
                if page_result.state in [PageState.PROFILE, PageState.PROFILE_LOGGED]:
                    self._silent_log.info(f"[导航到我的页面] ✓ 已到达个人页")
                    return True
                
                # 如果在其他页面，返回失败让外层重试
                elif page_result.state in [PageState.HOME, PageState.CHECKIN]:
                    self._silent_log.info(f"[导航到我的页面] ⚠️ 当前在{page_result.state.value}，导航失败")
                    continue
            
            # 无法确定页面状态，返回True让调用者通过获取资料来验证
            # 如果获取资料失败，调用者会重试导航
            self._silent_log.info(f"[导航到我的页面] ⚠️ 无法确定页面状态，返回True让调用者验证")
        
        # 最后确认
        self._silent_log.info(f"[导航到我的页面] 所有尝试完成，最后确认...")
        page_result = await self.detector.detect_page_with_priority(
            device_id,
            ['已登陆个人页.png', '未登陆个人页.png'],
            use_cache=False
        )
        if page_result and page_result.state:
            self._silent_log.info(f"[导航到我的页面] 最终页面: {page_result.state.value} - {page_result.details}")
            success = page_result.state in [PageState.PROFILE, PageState.PROFILE_LOGGED]  # 不包含个人页广告
            if success:
                self._silent_log.info(f"[导航到我的页面] ✓ 导航成功")
            else:
                self._silent_log.info(f"[导航到我的页面] ✗ 导航失败")
            return success
        else:
            self._silent_log.info(f"[导航到我的页面] ✗ 无法检测最终页面状态")
            return False
    
    async def navigate_to_cart(self, device_id: str, max_attempts: int = 5) -> bool:
        """导航到购物车页面
        
        Args:
            device_id: 设备ID
            max_attempts: 最大尝试次数
            
        Returns:
            bool: 是否成功到达购物车页面
        """
        for attempt in range(max_attempts):
            # 检测当前页面
            result = await self.detector.detect_page(device_id, use_ocr=True)
            
            # 处理弹窗
            if result.state == PageState.POPUP:
                await self.detector.close_popup(device_id)
                await asyncio.sleep(1)
                continue
            
            # 处理活动页面
            if result.state == PageState.UNKNOWN and "活动" in result.details:
                await self.adb.press_back(device_id)
                await asyncio.sleep(1)
                continue
            
            # 点击购物车标签
            await self.adb.tap(device_id, self.TAB_CART[0], self.TAB_CART[1])
            await asyncio.sleep(2)
            
            # 检查是否到达（通过OCR识别"购物车"关键词）
            result = await self.detector.detect_page(device_id, use_ocr=True)
            if "购物车" in result.details:
                return True
        
        return False
    
    async def go_back(self, device_id: str, times: int = 1) -> bool:
        """按返回键
        
        Args:
            device_id: 设备ID
            times: 按返回键的次数
            
        Returns:
            bool: 是否成功
        """
        for _ in range(times):
            await self.adb.press_back(device_id)
            await asyncio.sleep(1)
        return True
    
    async def handle_popup_and_activity(self, device_id: str, max_attempts: int = 3) -> bool:
        """处理弹窗和活动页面
        
        Args:
            device_id: 设备ID
            max_attempts: 最大尝试次数
            
        Returns:
            bool: 是否成功处理
        """
        for _ in range(max_attempts):
            result = await self.detector.detect_page(device_id, use_ocr=True)
            
            # 没有弹窗或活动页面
            if result.state not in [PageState.POPUP, PageState.UNKNOWN]:
                return True
            
            # 处理弹窗
            if result.state == PageState.POPUP:
                await self.detector.close_popup(device_id)
                await asyncio.sleep(1)
                continue
            
            # 处理活动页面
            if result.state == PageState.UNKNOWN and "活动" in result.details:
                await self.adb.press_back(device_id)
                await asyncio.sleep(1)
                continue
            
            # 其他未知状态，尝试返回
            await self.adb.press_back(device_id)
            await asyncio.sleep(1)
        
        return False
    
    async def safe_return_to_home(self, device_id: str, max_attempts: int = 15) -> bool:
        """安全返回首页（处理异常页面）
        
        当检测到不是预期的业务页面时，通过不断按返回键直到到达首页
        如果检测到卡死（连续5次相同页面），则直接点击首页标签
        如果点击首页标签也无效，则需要重启应用
        
        Args:
            device_id: 设备ID
            max_attempts: 最大尝试次数（默认15次）
            
        Returns:
            bool: 是否成功返回首页
        """
        print(f"  [safe_return_to_home] 开始返回首页，最多尝试 {max_attempts} 次")
        
        # 记录页面路径
        page_path = []
        stuck_count = 0  # 卡死计数器
        last_page_info = None
        stuck_threshold = 5  # 卡死阈值：连续5次相同页面
        
        for attempt in range(max_attempts):
            # 检测当前页面
            result = await self.detector.detect_page(device_id, use_ocr=True)
            
            # 记录页面信息
            page_info = f"{result.state.value} - {result.details}"
            page_path.append(page_info)
            
            print(f"  [尝试 {attempt+1}/{max_attempts}] 当前: {page_info}")
            
            # 已经在首页，成功返回
            if result.state == PageState.HOME:
                print(f"  ✅ 成功到达首页")
                print(f"\n  📍 返回路径:")
                for i, page in enumerate(page_path, 1):
                    print(f"     {i}. {page}")
                return True
            
            # 检测是否卡死（连续N次相同页面）
            if page_info == last_page_info:
                stuck_count += 1
                if stuck_count >= stuck_threshold:
                    print(f"  ⚠️  检测到页面卡死（连续{stuck_count}次相同页面）")
                    break
            else:
                stuck_count = 0
            
            last_page_info = page_info
            
            # 处理弹窗（关闭后继续检测）
            if result.state == PageState.POPUP:
                print(f"  → 检测到弹窗，关闭...")
                await self.detector.close_popup(device_id)
                await asyncio.sleep(1.5)
                continue
            
            # 如果是有导航栏的页面（如分类页、购物车页、我的页面），直接点击首页标签
            if ("有导航栏" in result.details or 
                "分类页" in result.details or 
                "购物车" in result.details or 
                result.state in [PageState.PROFILE, PageState.PROFILE_LOGGED]):
                print(f"  → 检测到有导航栏的页面，点击首页标签...")
                await self.adb.tap(device_id, self.TAB_HOME[0], self.TAB_HOME[1])
                await asyncio.sleep(2.0)
                continue
            
            # 不在首页，按返回键
            print(f"  → 按返回键...")
            await self.adb.press_back(device_id)
            await asyncio.sleep(2.0)  # 增加等待时间到2秒，让页面有充足时间切换
        
        # 如果按返回键多次后还没到首页，尝试点击首页标签
        print(f"  ⚠️  按返回键后仍未到达首页")
        print(f"\n  📍 已尝试的页面路径:")
        for i, page in enumerate(page_path, 1):
            print(f"     {i}. {page}")
        
        result = await self.detector.detect_page(device_id, use_ocr=True)
        if result.state != PageState.HOME:
            print(f"  → 尝试点击首页标签...")
            await self.adb.tap(device_id, self.TAB_HOME[0], self.TAB_HOME[1])
            await asyncio.sleep(2)
            
            # 处理可能的弹窗
            result = await self.detector.detect_page(device_id, use_ocr=True)
            if result.state == PageState.POPUP:
                print(f"  → 关闭弹窗...")
                await self.detector.close_popup(device_id)
                await asyncio.sleep(1)
            
            # 再次确认
            result = await self.detector.detect_page(device_id, use_ocr=True)
            print(f"  最终状态: {result.state.value}")
            page_path.append(f"{result.state.value} - {result.details} (点击首页标签后)")
            
            # 如果点击首页标签也无效，说明应用卡死，需要重启
            if result.state != PageState.HOME:
                print(f"  ❌ 点击首页标签无效，应用可能卡死")
                print(f"  ⚠️  建议：需要重启应用")
                return False
        
        return result.state == PageState.HOME

    async def navigate_to_profile_optimized(
        self, 
        device_id: str, 
        cache=None,
        max_attempts: int = 3,
        log_callback=None
    ) -> bool:
        """优化后的导航到个人页面
        
        优化点：
        1. 使用缓存检测当前页面状态，避免重复检测
        2. 已在目标页面时立即返回
        3. 点击后使用 SmartWaiter 等待页面切换
        4. 优化导航路径，减少不必要的返回操作
        5. 添加性能监控
        
        Args:
            device_id: 设备ID
            cache: 检测缓存（DetectionCache实例）
            max_attempts: 最大尝试次数
            log_callback: 日志回调函数（可选）
            
        Returns:
            bool: 是否成功到达个人页面
        """
        from .performance.performance_monitor import PerformanceMonitor
        from .performance.smart_waiter import SmartWaiter
        
        # 创建性能监控器
        monitor = PerformanceMonitor("导航到个人页")
        monitor.start()
        
        # 定义日志函数
        def log(msg):
            if log_callback:
                log_callback(msg)
            else:
                print(msg)
        
        log(f"  [导航优化] 开始导航到个人页面，最多尝试 {max_attempts} 次")
        
        # 创建智能等待器
        waiter = SmartWaiter()
        
        for attempt in range(max_attempts):
            log(f"  [导航优化] 尝试 {attempt + 1}/{max_attempts}")
            
            # 步骤1：检测当前页面（使用优先级模板）
            step_start = time.time()
            
            # 如果有缓存，先尝试从缓存获取
            page_result = None
            if cache:
                page_result = cache.get(device_id)
                if page_result:
                    log(f"  [导航优化] ✓ 使用缓存结果: {page_result.state.value}")
            
            # 缓存未命中，使用优先级检测（只匹配导航相关的页面）
            if not page_result:
                # 导航过程中最可能出现的页面（按优先级排序）
                expected_pages = [
                    '首页.png',  # 最可能
                    '已登陆个人页.png',  # 次可能
                    '未登陆个人页.png',
                    '加载卡死白屏.png',  # 加载中
                    '签到页.png',  # 可能需要返回
                    '模拟器桌面.png',  # 应用崩溃
                ]
                
                page_result = await self.detector.detect_page_with_priority(
                    device_id,
                    expected_pages,
                    use_cache=False  # 我们自己管理缓存
                )
                
                if cache and page_result:
                    cache.set(device_id, page_result)
            
            detection_time = time.time() - step_start
            monitor.record_step(
                "检测当前页面", 
                detection_time, 
                page_result.detection_method if page_result else "unknown"
            )
            
            if not page_result or not page_result.state:
                log(f"  [导航优化] ⚠️ 无法检测页面状态，重试...")
                await asyncio.sleep(0.5)
                continue
            
            current_state = page_result.state
            log(f"  [导航优化] 当前页面: {current_state.value} - {page_result.details}")
            
            # 步骤2：已在目标页面，立即返回
            if current_state in [PageState.PROFILE, PageState.PROFILE_LOGGED]:
                log(f"  [导航优化] ✓ 已在个人页面，立即返回")
                monitor.end()
                monitor.log_summary(log)
                return True
            
            # 步骤3：处理异常页面
            
            # 3.1：检测到模拟器桌面，说明应用已退出，需要重启
            if current_state == PageState.LAUNCHER:
                log(f"  [导航优化] ❌ 检测到模拟器桌面，应用已退出")
                log(f"  [导航优化] 导航失败，需要重启应用")
                monitor.end()
                monitor.log_summary(log)
                return False
            
            # 3.2：检测到弹窗，关闭
            if current_state == PageState.POPUP:
                step_start = time.time()
                log(f"  [导航优化] 检测到弹窗，关闭...")
                await self.detector.close_popup(device_id)
                
                # 等待一小段时间让页面稳定
                await asyncio.sleep(0.3)
                
                # 清除缓存，因为页面状态已改变
                if cache:
                    cache.clear(device_id)
                
                # 重新检测页面（期望是首页或个人页）
                expected_pages = ['首页.png', '已登陆个人页.png', '未登陆个人页.png', '模拟器桌面.png']
                result = await self.detector.detect_page_with_priority(
                    device_id,
                    expected_pages,
                    use_cache=False
                )
                
                popup_time = time.time() - step_start
                monitor.record_step("关闭弹窗", popup_time, "popup_handling")
                
                # 检测到模拟器桌面，应用已退出
                if result and result.state == PageState.LAUNCHER:
                    log(f"  [导航优化] ❌ 关闭弹窗后检测到模拟器桌面，应用已退出")
                    log(f"  [导航优化] 导航失败，需要重启应用")
                    monitor.end()
                    monitor.log_summary(log)
                    return False
                
                if result and cache:
                    cache.set(device_id, result)
                continue
            
            # 步骤4：如果在签到页面，返回
            if current_state == PageState.CHECKIN:
                step_start = time.time()
                log(f"  [导航优化] 当前在签到页面，按返回键...")
                await self.adb.press_back(device_id)
                
                # 等待一小段时间让页面稳定
                await asyncio.sleep(0.5)
                
                # 清除缓存
                if cache:
                    cache.clear(device_id)
                
                # 重新检测页面（期望是首页）
                expected_pages = ['首页.png', '模拟器桌面.png']
                result = await self.detector.detect_page_with_priority(
                    device_id,
                    expected_pages,
                    use_cache=False
                )
                
                back_time = time.time() - step_start
                monitor.record_step("从签到页返回", back_time, "navigation")
                
                # 检测到模拟器桌面，应用已退出
                if result and result.state == PageState.LAUNCHER:
                    log(f"  [导航优化] ❌ 从签到页返回后检测到模拟器桌面，应用已退出")
                    log(f"  [导航优化] 导航失败，需要重启应用")
                    monitor.end()
                    monitor.log_summary(log)
                    return False
                
                if result and cache:
                    cache.set(device_id, result)
                continue
            
            # 步骤5：确保在首页（优化路径）
            if current_state != PageState.HOME:
                step_start = time.time()
                log(f"  [导航优化] 当前不在首页，返回首页...")
                
                # 如果在有导航栏的页面，直接点击首页标签
                if "有导航栏" in page_result.details or current_state in [PageState.PROFILE, PageState.PROFILE_LOGGED]:
                    log(f"  [导航优化] 检测到有导航栏，直接点击首页标签")
                    await self.adb.tap(device_id, self.TAB_HOME[0], self.TAB_HOME[1])
                else:
                    # 否则按返回键
                    await self.adb.press_back(device_id)
                
                # 等待一小段时间让页面稳定
                await asyncio.sleep(0.5)
                
                # 清除缓存
                if cache:
                    cache.clear(device_id)
                
                # 重新检测页面（期望是首页）
                expected_pages = ['首页.png', '模拟器桌面.png']  # 添加模拟器桌面检测
                result = await self.detector.detect_page_with_priority(
                    device_id,
                    expected_pages,
                    use_cache=False
                )
                
                home_time = time.time() - step_start
                monitor.record_step("返回首页", home_time, "navigation")
                
                # 检测到模拟器桌面，应用已退出
                if result and result.state == PageState.LAUNCHER:
                    log(f"  [导航优化] ❌ 检测到模拟器桌面，应用已退出")
                    log(f"  [导航优化] 导航失败，需要重启应用")
                    monitor.end()
                    monitor.log_summary(log)
                    return False
                
                if not result or result.state != PageState.HOME:
                    log(f"  [导航优化] ⚠️ 未能返回首页")
                    continue
                
                if cache:
                    cache.set(device_id, result)
            
            # 步骤6：点击"我的"按钮
            step_start = time.time()
            log(f"  [导航优化] 点击'我的'按钮: {self.TAB_MY}")
            await self.adb.tap(device_id, self.TAB_MY[0], self.TAB_MY[1])
            
            # 等待一小段时间让页面稳定
            await asyncio.sleep(0.5)
            
            # 清除缓存
            if cache:
                cache.clear(device_id)
            
            # 重新检测页面（期望是个人页或加载页）
            expected_pages = ['已登陆个人页.png', '未登陆个人页.png', '加载卡死白屏.png', '模拟器桌面.png']
            result = await self.detector.detect_page_with_priority(
                device_id,
                expected_pages,
                use_cache=False
            )
            
            click_time = time.time() - step_start
            monitor.record_step("点击我的按钮", click_time, "navigation")
            
            # 检测到模拟器桌面，应用已退出
            if result and result.state == PageState.LAUNCHER:
                log(f"  [导航优化] ❌ 点击我的按钮后检测到模拟器桌面，应用已退出")
                log(f"  [导航优化] 导航失败，需要重启应用")
                monitor.end()
                monitor.log_summary(log)
                return False
            
            if result and result.state in [PageState.PROFILE, PageState.PROFILE_LOGGED]:
                log(f"  [导航优化] ✓ 成功到达个人页面")
                if cache:
                    cache.set(device_id, result)
                
                monitor.end()
                monitor.log_summary(log)
                return True
            elif result and result.state == PageState.LAUNCHER:
                # 再次检查：如果是模拟器桌面，立即返回失败
                log(f"  [导航优化] ❌ 检测到模拟器桌面，应用已退出")
                log(f"  [导航优化] 导航失败，需要重启应用")
                monitor.end()
                monitor.log_summary(log)
                return False
            else:
                log(f"  [导航优化] ⚠️ 点击后未到达个人页面，继续尝试...")
        
        # 最后确认
        log(f"  [导航优化] 所有尝试完成，最后确认...")
        page_result = await self.detector.detect_page(device_id, use_ocr=False)
        if page_result and page_result.state:
            log(f"  [导航优化] 最终页面: {page_result.state.value} - {page_result.details}")
            success = page_result.state in [PageState.PROFILE, PageState.PROFILE_LOGGED]
            
            monitor.end()
            monitor.log_summary(log)
            
            if success:
                log(f"  [导航优化] ✓ 导航成功")
            else:
                log(f"  [导航优化] ✗ 导航失败")
            return success
        else:
            log(f"  [导航优化] ✗ 无法检测最终页面状态")
            monitor.end()
            monitor.log_summary(log)
            return False

    async def navigate_to_lottery(self, device_id: str, max_attempts: int = 5) -> bool:
        """导航到抽奖页面（带页面状态守卫）
        
        Args:
            device_id: 设备ID
            max_attempts: 最大尝试次数
            
        Returns:
            bool: 是否成功到达抽奖页面
        """
        print(f"\n[导航] 开始导航到抽奖页面...")
        
        for attempt in range(max_attempts):
            print(f"[导航] 尝试 {attempt+1}/{max_attempts}")
            
            # 1. 使用守卫检测当前页面
            current_state = await self.guard.get_current_page_state(device_id, f"导航到抽奖页面 尝试{attempt+1}")
            print(f"[导航] 当前页面: {current_state.value}")
            
            # 2. 如果已在抽奖页面,返回成功
            # 注意: 需要根据实际的页面状态枚举来判断
            result = await self.detector.detect_page(device_id, use_ocr=True)
            if "抽奖页面" in result.details:
                print(f"[导航] ✓ 已在抽奖页面")
                return True
            
            # 3. 使用守卫处理异常页面
            if current_state == PageState.POPUP:
                print(f"[导航] 检测到弹窗,尝试处理...")
                handled = await self.guard._handle_unexpected_page(device_id, current_state, "导航到抽奖页面")
                if handled:
                    await asyncio.sleep(2)
                    continue
            
            # 4. 如果不在首页,先返回首页
            if current_state != PageState.HOME:
                print(f"[导航] 当前不在首页,先返回首页")
                if not await self.navigate_to_home(device_id):
                    print(f"[导航] ✗ 返回首页失败")
                    continue
                print(f"[导航] ✓ 已返回首页")
            
            # 5. 在首页查找抽奖入口（使用OCR）
            print(f"[导航] 在首页查找抽奖入口...")
            lottery_keywords = ["抽奖", "幸运抽奖", "立即抽奖", "免费抽奖"]
            
            # 导入UI自动化模块
            from .ui_automation import UIAutomation
            ui_automation = UIAutomation(self.adb, None)
            
            clicked = False
            for keyword in lottery_keywords:
                # 使用守卫执行点击操作
                async def click_lottery_entry():
                    return await ui_automation.click_by_text(device_id, keyword, timeout=5)
                
                try:
                    result = await self.guard.execute_with_guard(
                        device_id,
                        click_lottery_entry,
                        PageState.HOME,
                        f"点击抽奖入口: {keyword}"
                    )
                    
                    if result:
                        print(f"[导航] ✓ 点击了: {keyword}")
                        clicked = True
                        await asyncio.sleep(3)
                        break
                except Exception as e:
                    print(f"[导航] 点击 {keyword} 失败: {e}")
                    continue
            
            if not clicked:
                print(f"[导航] ✗ 未找到抽奖入口")
                continue
            
            # 6. 验证是否到达抽奖页面
            result = await self.detector.detect_page(device_id, use_ocr=True)
            print(f"[导航] 点击后页面: {result.state.value} - {result.details}")
            
            if "抽奖页面" in result.details:
                print(f"[导航] ✓ 成功到达抽奖页面\n")
                return True
            else:
                print(f"[导航] ✗ 点击后未到达抽奖页面")
        
        print(f"[导航] ✗ 导航失败,已尝试 {max_attempts} 次\n")
        return False
