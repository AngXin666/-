"""
智能等待器
Smart Waiter - 页面变化检测
"""

import asyncio
from typing import List, Optional


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
        adb_bridge=None  # 保留参数兼容性，但不使用
    ) -> Optional[any]:
        """等待页面变化到期望状态（定期深度学习检测）
        
        工作流程：
        1. 定期用深度学习检测页面类型
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
            
        Returns:
            检测到的页面结果，超时返回None
        """
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
    return await _global_waiter.wait_for_page_change(
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
