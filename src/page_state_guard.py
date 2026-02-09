"""
页面状态守卫模块 - 确保每个操作在正确的页面执行
Page State Guard Module - Ensure every operation executes on the correct page
"""

import asyncio
from typing import Optional, Callable, Any, List, Union
from functools import wraps

from .page_detector import PageState
from .adb_bridge import ADBBridge


class PageStateGuard:
    """页面状态守卫 - 在操作前后验证页面状态"""
    
    def __init__(self, adb: ADBBridge, detector: 'PageDetectorIntegrated'):
        """初始化页面状态守卫
        
        Args:
            adb: ADB桥接对象
            detector: 页面检测器（整合检测器或混合检测器）
        """
        self.adb = adb
        self.detector = detector
    
    async def verify_page_state(self, device_id: str, expected_state: PageState, 
                               operation_name: str, max_retries: int = 2) -> bool:
        """验证页面状态是否符合预期
        
        Args:
            device_id: 设备ID
            expected_state: 期望的页面状态
            operation_name: 操作名称（用于日志）
            max_retries: 最大重试次数
            
        Returns:
            bool: 是否符合预期
        """
        # 首先检查应用是否在前台
        is_foreground = await self.adb.is_app_in_foreground(device_id, "com.xmwl.shop")
        if not is_foreground:
            print(f"  [{operation_name}] ❌ 应用不在前台运行（可能在其他应用或设置中）")
            # 强制停止应用，然后重新启动
            print(f"  [{operation_name}] → 强制停止应用...")
            await self.adb.stop_app(device_id, "com.xmwl.shop")
            await asyncio.sleep(1)
            print(f"  [{operation_name}] → 重新启动应用...")
            await self.adb.start_app(device_id, "com.xmwl.shop")
            await asyncio.sleep(5)
            print(f"  [{operation_name}] ✓ 应用已重新启动")
        
        for attempt in range(max_retries):
            # 使用整合检测器检测页面
            page_result = await self.detector.detect_page(device_id, use_ocr=True)
            
            # 检查返回值是否有效
            if not page_result or not page_result.state:
                print(f"  [{operation_name}] ⚠️ 无法检测页面状态")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                    continue
                print(f"  [{operation_name}] ❌ 页面状态验证失败: 无法检测页面")
                return False
            
            if page_result.state == expected_state:
                if attempt > 0:
                    print(f"  [{operation_name}] ✓ 页面状态验证成功（第{attempt+1}次尝试）")
                return True
            
            # 处理异常页面（传递重试次数）
            if await self._handle_unexpected_page(device_id, page_result.state, expected_state, operation_name, retry_count=attempt):
                # 异常处理成功，继续验证
                await asyncio.sleep(1)
                continue
            
            if attempt < max_retries - 1:
                print(f"  [{operation_name}] ⚠️ 页面状态不符: {page_result.state.value}, "
                      f"期望: {expected_state.value}，等待后重试...")
                await asyncio.sleep(2)
        
        # 最后一次检查
        page_result = await self.detector.detect_page(device_id, use_ocr=True)
        if not page_result or not page_result.state:
            print(f"  [{operation_name}] ❌ 页面状态验证失败: 无法检测页面")
        else:
            print(f"  [{operation_name}] ❌ 页面状态验证失败: {page_result.state.value}, "
                  f"期望: {expected_state.value}")
        return False
    
    async def get_current_page_state(self, device_id: str, operation_name: str, 
                                    scenario: str = 'guard_general') -> PageState:
        """获取当前页面状态（带重试和错误处理）
        
        Args:
            device_id: 设备ID
            operation_name: 操作名称（用于日志）
            scenario: 场景名称（用于选择优先级模板，默认 'guard_general'）
            
        Returns:
            PageState: 当前页面状态，失败返回 UNKNOWN
        """
        # 首先检查应用是否在前台
        is_foreground = await self.adb.is_app_in_foreground(device_id, "com.xmwl.shop")
        if not is_foreground:
            print(f"  [{operation_name}] ❌ 应用不在前台运行（可能已退出或在其他应用）")
            return PageState.LAUNCHER  # 返回桌面状态
        
        # 使用整合检测器检测页面（页面分类器 + YOLO）
        page_result = await self.detector.detect_page(device_id, use_ocr=True)
        
        if not page_result or not page_result.state:
            print(f"  [{operation_name}] ⚠️ 无法检测页面状态，返回 UNKNOWN")
            return PageState.UNKNOWN
        
        return page_result.state
    
    async def _handle_unexpected_page(self, device_id: str, current_state: PageState, 
                                      expected_state: PageState, operation_name: str,
                                      retry_count: int = 0) -> bool:
        """处理非预期的页面状态
        
        根据页面类型采用不同的返回策略：
        - 分类页：直接点击首页按钮
        - 积分页/交易流水/商品列表/设置/搜索/文章/优惠券：优先点击返回按钮（YOLO），失败则按返回键
        
        Args:
            device_id: 设备ID
            current_state: 当前页面状态
            expected_state: 期望的页面状态
            operation_name: 操作名称
            retry_count: 重试次数（用于判断是否应该重启应用）
            
        Returns:
            bool: 是否成功处理
        """
        # 检查重试次数，如果超过5次，重启应用
        if retry_count >= 5:
            print(f"  [{operation_name}] ❌ 页面处理失败（已重试{retry_count}次），重启应用...")
            # 重启应用
            await self.adb.stop_app(device_id, "com.xmwl.shop")
            await asyncio.sleep(1)
            await self.adb.start_app(device_id, "com.xmwl.shop")
            await asyncio.sleep(5)
            print(f"  [{operation_name}] ✓ 应用已重启")
            return True
        
        if retry_count > 0:
            print(f"  [{operation_name}] ⚠️ 第{retry_count + 1}次尝试处理页面...")
        
        # 辅助方法：优先点击返回按钮（YOLO），失败则按返回键
        async def try_click_back_button(page_name: str, model_name: str = None) -> bool:
            """尝试点击返回按钮，失败则按返回键
            
            Args:
                page_name: 页面名称（用于日志）
                model_name: YOLO模型名称（如果有专用模型）
            
            Returns:
                bool: 是否成功处理
            """
            print(f"  [{operation_name}] ⚠️ 检测到{page_name}，尝试点击返回按钮...")
            
            # 如果有专用模型，使用YOLO检测返回按钮
            if model_name:
                back_button_pos = await self.detector.find_button_yolo(
                    device_id, 
                    model_name,
                    '返回按钮',
                    conf_threshold=0.5
                )
                if back_button_pos:
                    print(f"  [{operation_name}] → YOLO检测到返回按钮: {back_button_pos}")
                    await self.adb.tap(device_id, back_button_pos[0], back_button_pos[1])
                    await asyncio.sleep(2)
                    return True
            
            # 降级：按返回键
            print(f"  [{operation_name}] → 使用返回键")
            await self.adb.press_back(device_id)
            await asyncio.sleep(2)
            return True
        
        # 1. 如果退出到桌面，强制停止后重新启动应用
        if current_state == PageState.LAUNCHER:
            print(f"  [{operation_name}] ❌ 检测到应用已退出到桌面！")
            print(f"  [{operation_name}] → 强制停止应用...")
            await self.adb.stop_app(device_id, "com.xmwl.shop")
            await asyncio.sleep(1)
            print(f"  [{operation_name}] → 重新启动应用...")
            await self.adb.start_app(device_id, "com.xmwl.shop")
            await asyncio.sleep(5)  # 等待应用启动
            print(f"  [{operation_name}] ✓ 应用已重新启动")
            return True
        
        # 2. 如果是弹窗，尝试关闭
        if current_state == PageState.POPUP:
            print(f"  [{operation_name}] ⚠️ 检测到弹窗，尝试关闭...")
            success = await self.detector.close_popup(device_id)
            if success:
                print(f"  [{operation_name}] ✓ 弹窗已关闭")
                return True
            else:
                print(f"  [{operation_name}] ❌ 弹窗关闭失败")
                # 注意：不要直接返回False，因为可能是误判
                # 继续尝试其他处理方式
                print(f"  [{operation_name}] → 尝试按返回键...")
                await self.adb.press_back(device_id)
                await asyncio.sleep(2)
                return True
        
        # 3. 如果是登录页，记录并返回失败（需要上层处理）
        if current_state == PageState.LOGIN:
            print(f"  [{operation_name}] ⚠️ 检测到登录页面，需要重新登录")
            return False
        
        # 4. 如果是加载页，等待
        if current_state == PageState.LOADING:
            print(f"  [{operation_name}] ⚠️ 页面加载中，等待...")
            await asyncio.sleep(3)
            return True
        
        # 5. 如果是广告页，尝试跳过
        if current_state == PageState.AD:
            print(f"  [{operation_name}] ⚠️ 检测到广告，尝试跳过...")
            # 尝试点击跳过按钮
            await self.adb.tap(device_id, 480, 50)  # 右上角跳过按钮
            await asyncio.sleep(1)
            return True
        
        # 6. 如果是首页公告弹窗，按返回键关闭
        if current_state == PageState.HOME_NOTICE:
            print(f"  [{operation_name}] ⚠️ 检测到首页公告弹窗，按返回键关闭...")
            await self.adb.press_back(device_id)
            await asyncio.sleep(1)
            return True
        
        # 6.5. 如果是温馨提示弹窗，按返回键关闭
        if current_state == PageState.WARMTIP:
            print(f"  [{operation_name}] ⚠️ 检测到温馨提示弹窗，按返回键关闭...")
            await self.adb.press_back(device_id)
            await asyncio.sleep(1)
            return True
        
        # 6.6. 如果是首页异常代码弹窗，点击确认按钮
        if current_state == PageState.HOME_ERROR_POPUP:
            print(f"  [{operation_name}] ⚠️ 检测到首页异常代码弹窗，点击确认按钮...")
            success = await self.detector.click_element(device_id, "确认按钮")
            if not success:
                # 使用固定坐标 (根据标注数据，确认按钮中心约在 265, 637)
                await self.adb.tap(device_id, 265, 637)
                print(f"  [{operation_name}] → 使用固定坐标点击确认按钮")
            await asyncio.sleep(1)
            return True
        
        # 7. 如果是文章页，可能有多级，需要多次返回
        if current_state == PageState.ARTICLE:
            return await try_click_back_button("文章页")
        
        # 8. 如果是搜索页，优先点击返回按钮
        if current_state == PageState.SEARCH:
            return await try_click_back_button("搜索页")
        
        # 9. 如果是积分页，优先点击返回按钮
        if current_state == PageState.POINTS_PAGE:
            return await try_click_back_button("积分页", "积分页")
        
        # 10. 如果是设置页，优先点击返回按钮
        if current_state == PageState.SETTINGS:
            return await try_click_back_button("设置页")
        
        # 11. 如果是交易流水页，优先点击返回按钮
        if current_state == PageState.TRANSACTION_HISTORY:
            return await try_click_back_button("交易流水页")
        
        # 12. 如果是优惠券页，优先点击返回按钮
        if current_state == PageState.COUPON:
            return await try_click_back_button("优惠券页")
        
        # 13. 如果是分类页，点击首页按钮
        if current_state == PageState.CATEGORY:
            print(f"  [{operation_name}] ⚠️ 检测到分类页，点击首页按钮...")
            # 使用YOLO检测首页按钮
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
                await self.adb.tap(device_id, 90, 920)  # 首页按钮默认坐标
            await asyncio.sleep(2)
            return True
        
        # 14. 如果是未知页面，获取详细信息判断如何处理
        if current_state == PageState.UNKNOWN:
            # 获取页面详细信息
            page_result = await self.detector.detect_page(device_id, use_ocr=True)
            if page_result and page_result.details:
                details = page_result.details
                print(f"  [{operation_name}] ⚠️ 检测到未知页面: {details}")
                
                # 检查是否是需要返回的异常页面
                exception_keywords = [
                    "异常页面", "商品列表", "商品详情", "活动页面", "文章列表",
                    "产品套餐", "抽奖页面", "分类页", "其他页面"
                ]
                
                is_exception = any(keyword in details for keyword in exception_keywords)
                
                if is_exception:
                    print(f"  [{operation_name}] → 确认为异常页面，按返回键...")
                    await self.adb.press_back(device_id)
                    await asyncio.sleep(2)
                    return True
                else:
                    # 不确定是否是异常页面，尝试按返回键
                    print(f"  [{operation_name}] → 未知页面类型，尝试按返回键...")
                    await self.adb.press_back(device_id)
                    await asyncio.sleep(2)
                    return True
            else:
                # 无法获取详细信息，尝试按返回键
                print(f"  [{operation_name}] ⚠️ 检测到未知页面（无详细信息），尝试返回...")
                await self.adb.press_back(device_id)
                await asyncio.sleep(2)
                return True
        
        # 15. 其他未知状态
        print(f"  [{operation_name}] ⚠️ 未处理的页面状态: {current_state.value}")
        return False
    
    async def execute_with_guard(self, device_id: str, operation_name: str,
                                 pre_state: PageState, post_state: Optional[PageState],
                                 operation: Callable, *args, **kwargs) -> Any:
        """在页面状态守卫下执行操作
        
        Args:
            device_id: 设备ID
            operation_name: 操作名称
            pre_state: 操作前期望的页面状态
            post_state: 操作后期望的页面状态（None表示不验证）
            operation: 要执行的操作（async函数）
            *args, **kwargs: 传递给操作的参数
            
        Returns:
            操作的返回值，失败返回None
        """
        # 1. 操作前验证
        if not await self.verify_page_state(device_id, pre_state, f"{operation_name}-前置检查"):
            print(f"  [{operation_name}] ❌ 前置页面状态验证失败，操作取消")
            return None
        
        # 2. 执行操作
        try:
            print(f"  [{operation_name}] 执行操作...")
            result = await operation(*args, **kwargs)
        except Exception as e:
            print(f"  [{operation_name}] ❌ 操作执行异常: {e}")
            return None
        
        # 3. 操作后验证（如果指定了post_state）
        if post_state is not None:
            await asyncio.sleep(1)  # 等待页面稳定
            if not await self.verify_page_state(device_id, post_state, f"{operation_name}-后置检查"):
                print(f"  [{operation_name}] ⚠️ 后置页面状态验证失败")
                # 不返回None，因为操作可能已经成功
        
        return result
    
    async def safe_navigate(self, device_id: str, target_state: PageState,
                           navigate_func: Callable, operation_name: str,
                           max_attempts: int = 3) -> bool:
        """安全导航到目标页面（带重试和验证）
        
        Args:
            device_id: 设备ID
            target_state: 目标页面状态
            navigate_func: 导航函数
            operation_name: 操作名称
            max_attempts: 最大尝试次数
            
        Returns:
            bool: 是否成功导航
        """
        for attempt in range(max_attempts):
            # 1. 执行导航
            try:
                success = await navigate_func(device_id)
                if not success:
                    print(f"  [{operation_name}] ⚠️ 导航函数返回失败（第{attempt+1}次）")
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(2)
                        continue
                    return False
            except Exception as e:
                print(f"  [{operation_name}] ❌ 导航异常: {e}")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(2)
                    continue
                return False
            
            # 2. 验证是否到达目标页面
            await asyncio.sleep(1)
            page_result = await self.detector.detect_page(device_id, use_ocr=True)
            
            # 检查返回值是否有效
            if not page_result or not page_result.state:
                print(f"  [{operation_name}] ⚠️ 无法检测页面状态")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(2)
                    continue
                return False
            
            if page_result.state == target_state:
                print(f"  [{operation_name}] ✓ 成功导航到目标页面")
                return True
            
            # 3. 未到达，重试
            if attempt < max_attempts - 1:
                print(f"  [{operation_name}] ⚠️ 未到达目标页面（当前: {page_result.state.value}），重试...")
                await asyncio.sleep(2)
        
        print(f"  [{operation_name}] ❌ 导航失败，已尝试{max_attempts}次")
        return False
    
    async def ensure_page_state(self, device_id: str, target_state: PageState,
                               navigate_func: Callable, operation_name: str) -> bool:
        """确保当前在目标页面（如果不在则导航）
        
        Args:
            device_id: 设备ID
            target_state: 目标页面状态
            navigate_func: 导航函数
            operation_name: 操作名称
            
        Returns:
            bool: 是否成功
        """
        # 1. 检查当前页面
        page_result = await self.detector.detect_page(device_id, use_ocr=True)
        
        # 检查返回值是否有效
        if page_result is None or page_result.state is None:
            print(f"  [{operation_name}] ❌ 无法检测当前页面状态")
            return False
        
        # 2. 如果已经在目标页面，直接返回
        if page_result.state == target_state:
            print(f"  [{operation_name}] ✓ 已在目标页面")
            return True
        
        # 3. 处理异常页面
        if page_result.state in [PageState.POPUP, PageState.AD]:
            await self._handle_unexpected_page(device_id, page_result.state, target_state, operation_name, retry_count=0)
            await asyncio.sleep(1)
            # 再次检查
            page_result = await self.detector.detect_page(device_id, use_ocr=True)
            if page_result and page_result.state == target_state:
                return True
        
        # 4. 需要导航
        if page_result and page_result.state:
            print(f"  [{operation_name}] 当前页面: {page_result.state.value}, 需要导航到: {target_state.value}")
        else:
            print(f"  [{operation_name}] 无法检测当前页面，尝试导航到: {target_state.value}")
        return await self.safe_navigate(device_id, target_state, navigate_func, operation_name)


def with_page_guard(pre_state: PageState, post_state: Optional[PageState] = None):
    """装饰器：为方法添加页面状态守卫
    
    Args:
        pre_state: 操作前期望的页面状态
        post_state: 操作后期望的页面状态（None表示不验证）
    
    Usage:
        @with_page_guard(PageState.HOME, PageState.PROFILE)
        async def navigate_to_profile(self, device_id: str):
            # 操作代码
            pass
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(self, device_id: str, *args, **kwargs):
            # 假设self有guard属性
            if not hasattr(self, 'guard'):
                # 如果没有guard，直接执行
                return await func(self, device_id, *args, **kwargs)
            
            operation_name = func.__name__
            return await self.guard.execute_with_guard(
                device_id, operation_name, pre_state, post_state,
                func, self, device_id, *args, **kwargs
            )
        return wrapper
    return decorator
