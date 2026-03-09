"""
页面导航模块 - 处理应用内页面跳转
Navigator Module - Handle in-app page navigation
"""

import asyncio
import time
from typing import Optional, Tuple
from .adb_bridge import ADBBridge
from .page_detector import PageState
from .logger import get_silent_logger
from .timeouts_config import TimeoutsConfig


class Navigator:
    """页面导航器"""
    
    # 底部导航栏坐标 (540x960) - MuMu模拟器手动验证坐标（v1.7.1）
    # [2026-03-01] 修正：根据可视化结果调整坐标
    TAB_HOME = (70, 920)          # "首页" 标签（向左移动20px）
    TAB_CATEGORY = (200, 920)     # "分类" 标签（位置正确）
    TAB_CART = (330, 920)         # "购物车" 标签（应用无此功能）
    TAB_MY = (480, 920)           # "我的" 标签（位置正确）
    
    def __init__(self, adb: ADBBridge, detector=None):
        """初始化导航器
        
        Args:
            adb: ADB桥接对象
            detector: 页面检测器（可选，如果不提供则从ModelManager获取）
        """
        self.adb = adb
        
        # [2026-03-05] 修复原因：获取通用分类器用于页面状态检测
        # page_detector_integrated 只做YOLO元素检测，不做页面分类
        # 需要使用通用分类器来检测页面状态
        from .model_manager import ModelManager
        model_manager = ModelManager.get_instance()
        
        # 如果没有提供检测器，从ModelManager获取共享的YOLO识别器
        if detector is None:
            self.detector = model_manager.get_page_detector_integrated()
            print(f"[Navigator] 从ModelManager获取共享的YOLO识别器")
        else:
            # 使用传入的检测器（应该是从ModelManager获取的共享实例）
            self.detector = detector
        
        # [2026-03-05] 获取通用分类器用于页面状态检测
        self.general_classifier = model_manager.get_general_classifier()
        
        # 详细日志已关闭，保持日志整洁
        # 如需调试YOLO检测问题，可以取消下面的注释
        # self.detector.set_verbose(True)
        
        # 初始化静默日志记录器
        self._silent_log = get_silent_logger()
        
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
            
            # [2026-03-05] 修复数组比较错误：检查 texts 是否为 None 或长度为 0
            if ocr_result.texts is None or len(ocr_result.texts) == 0:
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
        """导航到首页（简化版：只点击首页按钮，不做页面检测）
        
        [2026-03-03] 修复原因：删除所有页面检测逻辑，避免 YOLO 检测器误判导致点击错误位置
        Navigator 只负责导航操作（点击按钮），页面检测由调用方负责
        
        Args:
            device_id: 设备ID
            max_attempts: 最大尝试次数（默认3次）
            
        Returns:
            bool: 是否成功点击首页按钮
        """
        self._silent_log.info(f"[导航到首页] 点击首页按钮...")
        
        # 直接点击首页按钮（使用固定坐标）
        await self.adb.tap(device_id, 90, 920)
        await asyncio.sleep(1.0)
        
        self._silent_log.info(f"[导航到首页] ✓ 已点击首页按钮")
        return True
    
    async def navigate_to_profile(self, device_id: str, max_attempts: int = 3) -> bool:
        """导航到个人页并处理广告（统一方法）
        
        核心逻辑（与 _navigate_to_profile_with_ad_handling 一致）：
        1. 点击"我的"按钮（YOLO + OCR）
        2. 高频扫描页面状态（每0.05秒）
        3. 检测到广告 → 立即用返回键关闭 → 继续扫描
        4. 检测到正常个人页 → 返回成功
        5. 超时（5秒）→ 重试
        
        Args:
            device_id: 设备ID
            max_attempts: 最大尝试次数（默认3次）
            
        Returns:
            bool: 是否成功到达我的页面
        """
        self._silent_log.info(f"[导航到我的页面] 开始导航，最多尝试 {max_attempts} 次")
        
        for attempt in range(max_attempts):
            self._silent_log.info(f"[导航到我的页面] 尝试 {attempt + 1}/{max_attempts}")
            
            # [2026-03-05] 修复原因：使用通用分类器检测页面状态
            # page_detector_integrated 只做YOLO元素检测，返回UNKNOWN
            # 需要使用通用分类器来检测页面状态
            if self.general_classifier:
                # 使用通用分类器检测页面
                from .screen_capture import ScreenCapture
                screen_capture = ScreenCapture(self.adb)
                screenshot_np = await screen_capture.capture(device_id)
                
                if screenshot_np:
                    from PIL import Image
                    import cv2
                    screenshot = Image.fromarray(cv2.cvtColor(screenshot_np, cv2.COLOR_BGR2RGB))
                    
                    # 使用通用分类器预测
                    from torchvision import transforms
                    import torch
                    
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    
                    image_tensor = transform(screenshot).unsqueeze(0)
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    image_tensor = image_tensor.to(device)
                    
                    with torch.no_grad():
                        output = self.general_classifier._model(image_tensor)
                        probabilities = torch.nn.functional.softmax(output, dim=1)
                        confidence, predicted = probabilities.max(1)
                        
                        # 加载类别
                        import json
                        from pathlib import Path
                        classes_path = Path(__file__).parent.parent / "models" / "page_classes.json"
                        with open(classes_path, 'r', encoding='utf-8') as f:
                            classes = json.load(f)
                        
                        predicted_class = classes[predicted.item()]
                        
                        # 映射到PageState
                        from .page_detector import PageState
                        page_state_map = {
                            "首页": PageState.HOME,
                            "个人页已登陆": PageState.PROFILE_LOGGED,
                            "个人页未登陆": PageState.PROFILE,
                            "个人页广告": PageState.PROFILE_AD,
                            "登录页": PageState.LOGIN,
                        }
                        
                        current_state = page_state_map.get(predicted_class, PageState.UNKNOWN)
                        self._silent_log.info(f"[导航到我的页面] 当前页面: {current_state.value} (通用分类器: {predicted_class}, 置信度: {confidence.item():.2%})")
                else:
                    # 截图失败，使用YOLO检测器
                    page_result = await self.detector.detect_page(
                        device_id, use_cache=False, detect_elements=False
                    )
                    current_state = page_result.state if page_result else PageState.UNKNOWN
                    self._silent_log.info(f"[导航到我的页面] 当前页面: {current_state.value} (YOLO)")
            else:
                # 没有通用分类器，使用YOLO检测器
                page_result = await self.detector.detect_page(
                    device_id, use_cache=False, detect_elements=False
                )
                current_state = page_result.state if page_result else PageState.UNKNOWN
                self._silent_log.info(f"[导航到我的页面] 当前页面: {current_state.value} (YOLO)")
                
                # 已经在我的页面
                if current_state in [PageState.PROFILE, PageState.PROFILE_LOGGED]:
                    self._silent_log.info(f"[导航到我的页面] ✓ 已在我的页面")
                    return True
                
                # 第1类：需要返回首页的页面（说明不在个人页区域）
                pages_need_go_home = [
                    PageState.ARTICLE,   # 文章页
                    PageState.SEARCH,    # 搜索页
                    PageState.CATEGORY,  # 分类页
                ]
                
                # 第2类：需要返回个人页的页面（说明在个人页区域的子页面）
                pages_need_go_back = [
                    PageState.SETTINGS,           # 设置页
                    PageState.TRANSACTION_HISTORY,# 交易流水页
                    PageState.COUPON,             # 优惠券页
                    PageState.POINTS_PAGE,        # 积分页
                ]
                
                # 处理第1类：返回首页
                if current_state in pages_need_go_home:
                    self._silent_log.info(f"[导航到我的页面] 检测到{current_state.value}，需要先返回首页...")
                    
                    # 点击首页按钮
                    if current_state == PageState.CATEGORY:
                        # 分类页：使用YOLO检测首页按钮
                        home_button_pos = await self.detector.find_button_yolo(
                            device_id, 
                            '分类页',
                            '首页按钮',
                            conf_threshold=0.5
                        )
                        if home_button_pos:
                            await self.adb.tap(device_id, home_button_pos[0], home_button_pos[1])
                        else:
                            # 降级：使用默认坐标
                            await self.adb.tap(device_id, self.TAB_HOME[0], self.TAB_HOME[1])
                    else:
                        # 其他页面：优先点击返回按钮，失败则按返回键
                        back_button_pos = await self.detector.find_button_yolo(
                            device_id, 
                            current_state.value,
                            '返回按钮',
                            conf_threshold=0.5
                        )
                        if back_button_pos:
                            await self.adb.tap(device_id, back_button_pos[0], back_button_pos[1])
                        else:
                            # 降级：按返回键
                            await self.adb.press_back(device_id)
                    
                    await asyncio.sleep(1)
                    self.detector.clear_cache(device_id)
                    continue
                
                # 处理第2类：返回个人页
                if current_state in pages_need_go_back:
                    self._silent_log.info(f"[导航到我的页面] 检测到{current_state.value}，按返回键返回个人页...")
                    
                    # 优先点击返回按钮（YOLO），失败则按返回键
                    back_button_pos = await self.detector.find_button_yolo(
                        device_id, 
                        current_state.value if current_state != PageState.POINTS_PAGE else '积分页',
                        '返回按钮',
                        conf_threshold=0.5
                    )
                    if back_button_pos:
                        self._silent_log.info(f"  YOLO检测到返回按钮: {back_button_pos}")
                        await self.adb.tap(device_id, back_button_pos[0], back_button_pos[1])
                    else:
                        # 降级：按返回键
                        self._silent_log.info(f"  使用返回键")
                        await self.adb.press_back(device_id)
                    
                    await asyncio.sleep(1)
                    self.detector.clear_cache(device_id)
                    
                    # 检查是否已经回到个人页
                    page_result = await self.detector.detect_page(device_id, use_cache=False, detect_elements=False)
                    if page_result and page_result.state in [PageState.PROFILE, PageState.PROFILE_LOGGED]:
                        self._silent_log.info(f"[导航到我的页面] ✓ 已返回到个人页")
                        return True
                    else:
                        self._silent_log.info(f"[导航到我的页面] ⚠️ 返回后页面状态: {page_result.state.value if page_result else 'unknown'}，重试...")
                        continue
            
            # 确保在首页
            # [2026-03-05] 修复原因：page_result 可能未定义，需要重新检测
            page_result = await self.detector.detect_page(device_id, use_cache=False, detect_elements=False)
            if page_result and page_result.state not in [PageState.HOME]:
                self._silent_log.info(f"[导航到我的页面] 当前不在首页，先返回首页...")
                success = await self.navigate_to_home(device_id, max_attempts=2)
                if not success:
                    self._silent_log.info(f"[导航到我的页面] ⚠️ 无法返回首页，重试...")
                    continue
                await asyncio.sleep(0.5)
            
            # 定义辅助函数：YOLO + OCR检测按钮
            async def try_detect_and_tap(button_name: str, model_name: str = '首页') -> bool:
                # 1. 尝试YOLO（使用正确的类别名称）
                # 注意：首页模型的类别名称是 '我的按钮'、'每日签到按钮'（带"按钮"后缀）
                # avatar_homepage模型的类别名称是 '首页按钮'、'头像'（带"按钮"后缀）
                yolo_class_name = f"{button_name}按钮" if model_name == '首页' else button_name
                button_pos = await self.detector.find_button_yolo(
                    device_id, 
                    model_name,
                    yolo_class_name,  # 使用正确的类别名称
                    conf_threshold=0.5
                )
                
                if button_pos:
                    # 验证位置是否在底部导航栏区域（y > 850）
                    if button_pos[1] > 850:
                        self._silent_log.log(f"  YOLO检测到'{button_name}'按钮: {button_pos}")
                        await self.adb.tap(device_id, button_pos[0], button_pos[1])
                        return True
                    else:
                        self._silent_log.log(f"  YOLO检测到'{button_name}'按钮: {button_pos}，但不在底部导航栏区域，忽略")
                
                # 2. 降级到OCR
                self._silent_log.log(f"  YOLO未检测到，尝试OCR...")
                ocr_pos = await self.screen_capture.find_text_location(device_id, button_name)
                
                if ocr_pos:
                    # 验证位置是否在底部导航栏区域（y > 850）
                    if ocr_pos[1] > 850:
                        self._silent_log.log(f"  OCR检测到'{button_name}'按钮: {ocr_pos}")
                        await self.adb.tap(device_id, ocr_pos[0], ocr_pos[1])
                        return True
                    else:
                        self._silent_log.log(f"  OCR检测到'{button_name}'按钮: {ocr_pos}，但不在底部导航栏区域，忽略")
                
                # 3. 使用默认坐标作为最后的降级方案
                self._silent_log.log(f"  检测失败，使用默认坐标: {self.TAB_MY}")
                await self.adb.tap(device_id, self.TAB_MY[0], self.TAB_MY[1])
                return True
            
            # 尝试检测并点击"我的"按钮（使用首页模型）
            success = await try_detect_and_tap("我的", "首页")
            
            if not success:
                self._silent_log.info(f"  未检测到'我的'按钮，可能不在首页，尝试先导航到首页...")
                
                # 尝试点击首页按钮（使用分类页模型）
                # 注意：首页模型没有"首页按钮"类别，只有"我的按钮"和"每日签到按钮"
                # 所以这里使用分类页模型
                home_success = await try_detect_and_tap("首页", "分类页")
                
                if home_success:
                    self._silent_log.info(f"  已点击首页，等待页面加载...")
                    await asyncio.sleep(0.5)
                    
                    # 再次尝试点击"我的"按钮（此时在首页，使用首页模型）
                    success = await try_detect_and_tap("我的", "首页")
                
                # 如果所有检测方法都失败，已经使用了默认坐标
                if not success:
                    self._silent_log.info(f"  ❌ 所有检测方法都失败，但已使用默认坐标")
                    success = True  # 标记为成功，因为已经点击了默认坐标
            
            # 点击"我的"后，等1秒
            await asyncio.sleep(1.0)
            
            # 按返回键
            await self.adb.press_back(device_id)
            
            # 清除缓存
            self.detector.clear_cache(device_id)
            
            # 开始检测页面状态和高频扫描
            max_scan_time = 5.0
            scan_interval = 0.05  # 每50毫秒扫描一次
            start_time = asyncio.get_event_loop().time()
            ad_closed_count = 1  # 已经按了一次返回键
            
            while (asyncio.get_event_loop().time() - start_time) < max_scan_time:
                # 检测当前页面状态
                page_result = await self.detector.detect_page(
                    device_id, use_cache=False, detect_elements=False
                )
                
                if not page_result or not page_result.state:
                    await asyncio.sleep(scan_interval)
                    continue
                
                current_state = page_result.state
                
                # 检测到正常个人页 → 成功
                if current_state in [PageState.PROFILE, PageState.PROFILE_LOGGED]:
                    elapsed = asyncio.get_event_loop().time() - start_time
                    self._silent_log.info(f"  ✓ 到达个人页（耗时: {elapsed:.1f}秒，关闭广告: {ad_closed_count}次）")
                    return True
                
                # 检测到广告 → 立即用返回键关闭
                elif current_state == PageState.PROFILE_AD:
                    self._silent_log.info(f"  ⚠️ 检测到个人页广告，使用返回键关闭...")
                    await self.adb.press_back(device_id)
                    ad_closed_count += 1
                    
                    # 等待0.3秒让广告关闭动画完成
                    await asyncio.sleep(0.3)
                    
                    # 清除缓存
                    self.detector.clear_cache(device_id)
                    
                    # 继续扫描（可能还有广告，或者已经到达个人页）
                    continue
                
                # 其他状态 → 继续扫描
                else:
                    await asyncio.sleep(scan_interval)
            
            # 超时，记录日志并重试
            elapsed = asyncio.get_event_loop().time() - start_time
            self._silent_log.info(f"  ❌ 导航到个人页超时（耗时: {elapsed:.1f}秒，关闭广告: {ad_closed_count}次），重试...")
        
        # 所有尝试都失败
        self._silent_log.info(f"[导航到我的页面] ✗ 所有尝试都失败")
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
            
            # 处理首页公告弹窗（点击弹窗外上方空白区域关闭）
            if result.state == PageState.HOME_NOTICE:
                print(f"  → 检测到首页公告弹窗，点击弹窗外上方空白区域关闭...")
                await self.adb.tap(device_id, 270, 200)
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
        """导航到个人页面（简化版，不做页面检测）
        
        [2026-03-03] 修复原因：删除所有页面检测逻辑，避免YOLO检测器误判
        Navigator 只负责导航操作（点击按钮），页面检测由调用方负责
        
        Args:
            device_id: 设备ID
            cache: 检测缓存（已废弃，保留以兼容旧代码）
            max_attempts: 最大尝试次数（已废弃，保留以兼容旧代码）
            log_callback: 日志回调函数（可选）
            
        Returns:
            bool: 始终返回True（表示已执行点击操作）
        """
        # 定义日志函数
        def log(msg):
            if log_callback:
                log_callback(msg)
        
        # 直接点击"我的"按钮，不做任何页面检测
        # 页面检测由调用方负责
        await self.adb.tap(device_id, self.TAB_MY[0], self.TAB_MY[1])
        await asyncio.sleep(1.0)  # 等待页面切换
        
        return True  # 始终返回True，表示已执行点击操作

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
            
            # 1. 检测当前页面
            result = await self.detector.detect_page(device_id, use_ocr=True)
            if not result or not result.state:
                print(f"[导航] 无法检测页面状态")
                await asyncio.sleep(2)
                continue
            
            current_state = result.state
            print(f"[导航] 当前页面: {current_state.value}")
            
            # 2. 如果已在抽奖页面,返回成功
            if "抽奖页面" in result.details:
                print(f"[导航] ✓ 已在抽奖页面")
                return True
            
            # 3. 处理弹窗
            if current_state == PageState.POPUP:
                print(f"[导航] 检测到弹窗,尝试关闭...")
                await self.detector.close_popup(device_id)
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
                try:
                    result = await ui_automation.click_by_text(device_id, keyword, timeout=5)
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
