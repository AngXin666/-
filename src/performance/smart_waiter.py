"""
智能等待器
Smart Waiter - 页面变化检测
"""

import asyncio
from typing import List, Optional
from .page_change_detector import PageChangeDetector


class SmartWaiter:
    """智能等待器
    
    核心理念：
    1. 定期用深度学习检测页面类型
    2. 检测到期望页面立即返回
    3. 超时只是防止卡死的保护机制
    """
    
    def __init__(self):
        """初始化智能等待器"""
        # 获取静默日志记录器（用于详细调试信息）
        from ..logger import get_silent_logger
        self._debug_logger = get_silent_logger()
        
        # 初始化页面变化检测器（感知哈希）
        self._change_detector = PageChangeDetector(hash_size=8)
    
    
    async def wait_for_page_change(
        self,
        device_id: str,
        detector,  # PageDetectorIntegrated
        expected_states: List,  # List[PageState]
        max_wait: float = 15.0,  # 超时保护（防止卡死）
        poll_interval: float = 0.5,  # 轮询间隔
        log_callback=None,
        ignore_loading: bool = True,
        stability_check: bool = False,
        stability_count: int = 1,
        adb_bridge=None,  # 保留参数兼容性，但不使用
        detection_mode: str = "deep_learning"  # 检测模式：deep_learning/hybrid/fast
    ) -> Optional[any]:
        """等待页面变化到期望状态
        
        检测模式：
        - deep_learning: 纯深度学习检测（精准，适合签到/转账）
        - hybrid: 混合检测（感知哈希+深度学习，适合启动流程）
        - fast: 快速检测（仅感知哈希，适合快速变化场景）
        
        工作流程：
        1. 根据detection_mode选择检测策略
        2. 检测到期望状态立即返回
        3. 超时保护：max_wait 秒后强制返回
        
        Args:
            device_id: 设备ID
            detector: 页面检测器
            expected_states: 期望的页面状态列表
            max_wait: 超时保护时间（秒，默认15秒）
            poll_interval: 轮询间隔（秒，默认0.5秒）
            log_callback: 日志回调函数
            ignore_loading: 是否忽略loading状态
            stability_check: 是否进行稳定性检测
            stability_count: 稳定性检测次数
            adb_bridge: 保留参数兼容性（不使用）
            detection_mode: 检测模式
            
        Returns:
            检测到的页面结果，超时返回None
        """
        if detection_mode == "hybrid":
            return await self._wait_hybrid_detection(
                device_id, detector, expected_states, max_wait, poll_interval,
                log_callback, ignore_loading, stability_check, stability_count, adb_bridge
            )
        elif detection_mode == "fast":
            return await self._wait_fast_detection(
                device_id, detector, expected_states, max_wait, poll_interval,
                log_callback, adb_bridge
            )
        else:  # deep_learning
            return await self._wait_deep_learning_detection(
                device_id, detector, expected_states, max_wait, poll_interval,
                log_callback, ignore_loading, stability_check, stability_count
            )
    
    async def _wait_deep_learning_detection(
        self,
        device_id: str,
        detector,
        expected_states: List,
        max_wait: float,
        poll_interval: float,
        log_callback,
        ignore_loading: bool,
        stability_check: bool,
        stability_count: int
    ) -> Optional[any]:
        """纯深度学习检测模式（原有逻辑）"""
        from ..page_detector import PageState
        
        start_time = asyncio.get_event_loop().time()
        
        # 页面状态跟踪
        last_state = None
        last_confidence = 0.0
        same_state_count = 0
        
        # 统计信息
        total_checks = 0
        state_changes = 0
        
        # 检测器类型
        detector_type = type(detector).__name__
        self._debug_logger.debug(f"[SmartWaiter] 检测器类型: {detector_type}")
        
        # 清除缓存，确保第一次检测就是最新状态
        if detector_type == 'PageDetectorIntegrated' and hasattr(detector, '_detection_cache'):
            detector._detection_cache.clear(device_id)
            self._debug_logger.debug(f"[SmartWaiter] 已清除检测缓存")
        
        while asyncio.get_event_loop().time() - start_time < max_wait:
            total_checks += 1
            current_time = asyncio.get_event_loop().time()
            
            # 深度学习检测页面类型
            if detector_type == 'PageDetectorIntegrated':
                result = await detector.detect_page(device_id, use_cache=False, detect_elements=False)
            else:
                result = await detector.detect_page(device_id, use_ocr=False, use_dl=True)
            
            if not result or not result.state:
                await asyncio.sleep(poll_interval)
                continue
            
            current_state = result.state
            current_confidence = result.confidence
            
            # 调试日志
            self._debug_logger.debug(
                f"[SmartWaiter] 检测 #{total_checks}: {current_state.value} (置信度{current_confidence:.2%})"
            )
            
            # 检测到状态变化
            if current_state != last_state:
                state_changes += 1
                if log_callback and last_state is not None:
                    log_callback(f"页面变化: {last_state.value} → {current_state.value}")
                    
                    elapsed = asyncio.get_event_loop().time() - start_time
                    self._debug_logger.debug(
                        f"[SmartWaiter] 页面变化: {last_state.value} → {current_state.value} "
                        f"(耗时{elapsed:.2f}秒, 置信度{current_confidence:.2%})"
                    )
                
                # 重置稳定性计数
                same_state_count = 1
                last_state = current_state
                last_confidence = current_confidence
                
                # 清除缓存
                if detector_type == 'PageDetectorIntegrated' and hasattr(detector, '_detection_cache'):
                    detector._detection_cache.clear(device_id)
            else:
                # 状态相同，增加计数
                same_state_count += 1
            
            # 检查是否是期望状态
            if current_state in expected_states:
                if stability_check:
                    if same_state_count >= stability_count:
                        elapsed = asyncio.get_event_loop().time() - start_time
                        if log_callback:
                            log_callback(f"✓ 页面稳定在 {current_state.value}，耗时{elapsed:.2f}秒")
                        
                        self._debug_logger.debug(
                            f"[SmartWaiter] ✓ 页面稳定在 {current_state.value} "
                            f"(连续{same_state_count}次, 置信度{current_confidence:.2%}, 耗时{elapsed:.2f}秒)"
                        )
                        self._debug_logger.debug(
                            f"[SmartWaiter] 统计: 总检测{total_checks}次, 状态变化{state_changes}次"
                        )
                        return result
                    else:
                        if log_callback:
                            log_callback(f"确认中: {current_state.value}（{same_state_count}/{stability_count}次）")
                        
                        self._debug_logger.debug(
                            f"[SmartWaiter] 确认中: {current_state.value} "
                            f"({same_state_count}/{stability_count}次, 置信度{current_confidence:.2%})"
                        )
                else:
                    # 不需要稳定性检测，直接返回
                    elapsed = asyncio.get_event_loop().time() - start_time
                    if log_callback:
                        log_callback(f"✓ 检测到 {current_state.value}，耗时{elapsed:.2f}秒")
                    
                    self._debug_logger.debug(
                        f"[SmartWaiter] 统计: 总检测{total_checks}次, 状态变化{state_changes}次"
                    )
                    return result
            
            # 如果是loading状态且ignore_loading=True
            elif ignore_loading and current_state == PageState.LOADING:
                same_state_count = 0
                if log_callback and total_checks % 5 == 1:
                    elapsed = asyncio.get_event_loop().time() - start_time
                    log_callback(f"页面加载中...（{elapsed:.0f}秒）")
                
                if total_checks % 3 == 1:
                    elapsed = asyncio.get_event_loop().time() - start_time
                    self._debug_logger.debug(
                        f"[SmartWaiter] 页面加载中 (已等待{elapsed:.1f}秒, 检测{total_checks}次)"
                    )
            
            # 继续轮询
            await asyncio.sleep(poll_interval)
        
        # 超时保护触发
        elapsed = asyncio.get_event_loop().time() - start_time
        if log_callback:
            log_callback(f"⚠️ 等待超时（{elapsed:.1f}秒）")
        
        self._debug_logger.warning(f"[SmartWaiter] ⚠️ 超时保护触发 (耗时{elapsed:.1f}秒)")
        self._debug_logger.warning(
            f"[SmartWaiter] 统计: 总检测{total_checks}次, 状态变化{state_changes}次"
        )
        if last_state:
            self._debug_logger.warning(
                f"[SmartWaiter] 最后状态: {last_state.value} (置信度{last_confidence:.2%})"
            )
        
        return None
    
    async def _wait_hybrid_detection(
        self,
        device_id: str,
        detector,
        expected_states: List,
        max_wait: float,
        poll_interval: float,
        log_callback,
        ignore_loading: bool,
        stability_check: bool,
        stability_count: int,
        adb_bridge
    ) -> Optional[any]:
        """混合检测模式：感知哈希检测变化 + 深度学习识别页面
        
        适用场景：启动流程（页面快速切换）
        """
        from ..page_detector import PageState
        
        start_time = asyncio.get_event_loop().time()
        
        # 重置变化检测器
        self._change_detector.reset()
        
        # 统计信息
        total_checks = 0
        hash_changes = 0
        dl_checks = 0
        
        self._debug_logger.debug(f"[SmartWaiter] 混合检测模式启动")
        
        while asyncio.get_event_loop().time() - start_time < max_wait:
            total_checks += 1
            
            # 获取截图
            if adb_bridge is None:
                adb_bridge = detector.adb if hasattr(detector, 'adb') else None
            
            if adb_bridge:
                screenshot_data = await adb_bridge.screencap(device_id)
                if not screenshot_data:
                    await asyncio.sleep(poll_interval)
                    continue
                
                # 使用感知哈希检测变化
                changed, distance = await self._change_detector.detect_change(screenshot_data, threshold=5)
                
                if changed:
                    hash_changes += 1
                    if log_callback:
                        log_callback(f"检测到页面变化（距离:{distance}）")
                    
                    self._debug_logger.debug(
                        f"[SmartWaiter] 感知哈希检测到变化 #{hash_changes} (距离:{distance})"
                    )
                    
                    # 变化时用深度学习识别页面类型
                    dl_checks += 1
                    detector_type = type(detector).__name__
                    
                    if detector_type == 'PageDetectorIntegrated':
                        result = await detector.detect_page(device_id, use_cache=False, detect_elements=False)
                    else:
                        result = await detector.detect_page(device_id, use_ocr=False, use_dl=True)
                    
                    if result and result.state:
                        current_state = result.state
                        
                        self._debug_logger.debug(
                            f"[SmartWaiter] 深度学习识别: {current_state.value} (置信度{result.confidence:.2%})"
                        )
                        
                        # 检查是否是期望状态
                        if current_state in expected_states:
                            elapsed = asyncio.get_event_loop().time() - start_time
                            if log_callback:
                                log_callback(f"✓ 检测到 {current_state.value}，耗时{elapsed:.2f}秒")
                            
                            self._debug_logger.debug(
                                f"[SmartWaiter] 统计: 总检测{total_checks}次, 哈希变化{hash_changes}次, 深度学习{dl_checks}次"
                            )
                            return result
            
            # 继续轮询
            await asyncio.sleep(poll_interval)
        
        # 超时
        elapsed = asyncio.get_event_loop().time() - start_time
        if log_callback:
            log_callback(f"⚠️ 等待超时（{elapsed:.1f}秒）")
        
        self._debug_logger.warning(f"[SmartWaiter] ⚠️ 混合检测超时 (耗时{elapsed:.1f}秒)")
        self._debug_logger.warning(
            f"[SmartWaiter] 统计: 总检测{total_checks}次, 哈希变化{hash_changes}次, 深度学习{dl_checks}次"
        )
        
        return None
    
    async def _wait_fast_detection(
        self,
        device_id: str,
        detector,
        expected_states: List,
        max_wait: float,
        poll_interval: float,
        log_callback,
        adb_bridge
    ) -> Optional[any]:
        """快速检测模式：仅使用感知哈希检测变化
        
        适用场景：只需要知道页面是否变化，不需要识别具体页面类型
        """
        start_time = asyncio.get_event_loop().time()
        
        # 重置变化检测器
        self._change_detector.reset()
        
        total_checks = 0
        changes = 0
        
        self._debug_logger.debug(f"[SmartWaiter] 快速检测模式启动")
        
        while asyncio.get_event_loop().time() - start_time < max_wait:
            total_checks += 1
            
            # 获取截图
            if adb_bridge is None:
                adb_bridge = detector.adb if hasattr(detector, 'adb') else None
            
            if adb_bridge:
                screenshot_data = await adb_bridge.screencap(device_id)
                if not screenshot_data:
                    await asyncio.sleep(poll_interval)
                    continue
                
                # 使用感知哈希检测变化
                changed, distance = await self._change_detector.detect_change(screenshot_data, threshold=5)
                
                if changed:
                    changes += 1
                    if log_callback:
                        log_callback(f"检测到变化 #{changes}（距离:{distance}）")
                    
                    self._debug_logger.debug(
                        f"[SmartWaiter] 快速检测到变化 #{changes} (距离:{distance})"
                    )
            
            await asyncio.sleep(poll_interval)
        
        # 超时
        elapsed = asyncio.get_event_loop().time() - start_time
        if log_callback:
            log_callback(f"快速检测完成，共检测到{changes}次变化")
        
        self._debug_logger.debug(
            f"[SmartWaiter] 快速检测完成: 总检测{total_checks}次, 变化{changes}次"
        )
        
        return None
    
    async def wait_for_condition(
        self,
        condition_func,
        max_wait: float = 5.0,
        poll_interval: float = 0.5,
        log_callback=None
    ) -> bool:
        """等待条件满足
        
        Args:
            condition_func: 条件函数，返回True表示条件满足
            max_wait: 最大等待时间（秒）
            poll_interval: 轮询间隔（秒）
            log_callback: 可选的日志回调函数
            
        Returns:
            条件是否在超时前满足
        """
        start_time = asyncio.get_event_loop().time()
        attempt = 0
        
        while asyncio.get_event_loop().time() - start_time < max_wait:
            attempt += 1
            
            # 检查条件
            if await condition_func() if asyncio.iscoroutinefunction(condition_func) else condition_func():
                # 条件满足，立即返回
                if log_callback:
                    elapsed = asyncio.get_event_loop().time() - start_time
                    log_callback(f"✓ 条件满足，耗时 {elapsed:.2f}秒（尝试{attempt}次）")
                return True
            
            # 条件未满足，等待后继续
            await asyncio.sleep(poll_interval)
        
        # 超时
        if log_callback:
            log_callback(f"⚠️ 等待超时（{max_wait}秒），条件未满足")
        return False


# ============================================================================
# 全局便捷函数
# ============================================================================

# 全局单例
_global_waiter = SmartWaiter()


async def wait_for_page(
    device_id: str,
    detector,
    expected_states: List,
    log_callback=None
) -> Optional[any]:
    """全局便捷函数：等待页面变化
    
    可以在项目任何地方直接调用，无需创建实例。
    内置最佳实践参数：
    - max_wait=15.0 (超时保护)
    - poll_interval=0.5 (每0.5秒检测一次)
    - stability_check=False (检测到即返回)
    - ignore_loading=True (忽略loading状态)
    
    **超时处理**：
    - 如果15秒超时未检测到目标页面，返回 None
    - 调用方应该标记失败，失败的账号会在所有账户完成后统一重试
    
    Args:
        device_id: 设备ID
        detector: 页面检测器
        expected_states: 期望的页面状态列表
        log_callback: 日志回调函数
        
    Returns:
        检测到的页面结果，超时返回None
        
    示例：
        from src.performance.smart_waiter import wait_for_page
        
        # 简单调用
        result = await wait_for_page(device_id, detector, [PageState.HOME])
        
        # 带日志
        result = await wait_for_page(
            device_id, detector, [PageState.PROFILE_LOGGED],
            log_callback=lambda msg: print(f"[等待] {msg}")
        )
    """
    # 第一次尝试：正常等待15秒
    result = await _global_waiter.wait_for_page_change(
        device_id=device_id,
        detector=detector,
        expected_states=expected_states,
        max_wait=15.0,
        poll_interval=0.5,  # 每0.5秒检测一次
        log_callback=log_callback,
        ignore_loading=True,
        stability_check=False,  # 检测到即返回
        stability_count=1
    )
    
    # 如果检测成功，直接返回
    if result:
        return result
    
    # 如果15秒超时，直接返回 None，标记失败
    # 失败的账号会在所有账户完成后统一重试
    if log_callback:
        log_callback("⚠️ 15秒超时，标记失败")
    
    return None
