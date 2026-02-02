"""
等待辅助模块 - 提供动态等待功能
Wait Helper Module - Provide dynamic wait functionality
"""

import asyncio
from typing import Callable, Optional, Any
from enum import Enum


class WaitResult(Enum):
    """等待结果"""
    SUCCESS = "success"
    TIMEOUT = "timeout"
    ERROR = "error"


async def wait_for_condition(
    condition_func: Callable[[], Any],
    timeout: float = 10.0,
    check_interval: float = 0.2,
    success_condition: Optional[Callable[[Any], bool]] = None
) -> tuple[WaitResult, Any]:
    """等待条件满足（动态检测）
    
    Args:
        condition_func: 条件检测函数（async或sync）
        timeout: 超时时间（秒）
        check_interval: 检测间隔（秒）
        success_condition: 成功条件判断函数，如果为None则检查返回值是否为True
        
    Returns:
        (结果状态, 最后一次检测的返回值)
        
    Example:
        # 等待页面加载完成
        result, page_state = await wait_for_condition(
            lambda: detector.detect_page(device_id),
            timeout=5.0,
            success_condition=lambda state: state == PageState.HOME
        )
    """
    start_time = asyncio.get_event_loop().time()
    last_result = None
    
    while True:
        try:
            # 执行条件检测
            if asyncio.iscoroutinefunction(condition_func):
                result = await condition_func()
            else:
                result = condition_func()
            
            last_result = result
            
            # 判断是否成功
            if success_condition:
                if success_condition(result):
                    return (WaitResult.SUCCESS, result)
            else:
                if result:
                    return (WaitResult.SUCCESS, result)
            
            # 检查超时
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= timeout:
                return (WaitResult.TIMEOUT, last_result)
            
            # 等待下一次检测
            await asyncio.sleep(check_interval)
            
        except Exception as e:
            return (WaitResult.ERROR, e)


async def wait_for_page_state(
    detector,
    device_id: str,
    expected_state,
    timeout: float = 5.0,
    check_interval: float = 0.3
) -> tuple[WaitResult, Any]:
    """等待页面状态（专用于页面检测）
    
    Args:
        detector: 页面检测器
        device_id: 设备ID
        expected_state: 期望的页面状态
        timeout: 超时时间（秒）
        check_interval: 检测间隔（秒）
        
    Returns:
        (结果状态, 页面检测结果)
    """
    async def check_page():
        result = await detector.detect_page(device_id, use_ocr=True)
        return result
    
    def is_expected_state(result):
        return result.state == expected_state
    
    return await wait_for_condition(
        check_page,
        timeout=timeout,
        check_interval=check_interval,
        success_condition=is_expected_state
    )


async def wait_after_action(
    min_wait: float = 0.3,
    max_wait: float = 2.0,
    check_func: Optional[Callable[[], Any]] = None,
    check_interval: float = 0.2
) -> float:
    """操作后动态等待
    
    在执行操作（如点击、输入）后，动态等待页面响应：
    - 至少等待 min_wait 秒（给系统响应时间）
    - 如果提供了 check_func，则快速检测直到条件满足或超时
    - 最多等待 max_wait 秒
    
    Args:
        min_wait: 最小等待时间（秒）
        max_wait: 最大等待时间（秒）
        check_func: 可选的检测函数，返回True表示可以继续
        check_interval: 检测间隔（秒）
        
    Returns:
        实际等待的时间（秒）
        
    Example:
        # 点击后等待页面稳定
        await adb.tap(device_id, x, y)
        await wait_after_action(
            min_wait=0.3,
            max_wait=2.0,
            check_func=lambda: page_is_stable()
        )
    """
    start_time = asyncio.get_event_loop().time()
    
    # 先等待最小时间
    await asyncio.sleep(min_wait)
    
    # 如果没有检测函数，直接返回
    if not check_func:
        return min_wait
    
    # 动态检测
    while True:
        elapsed = asyncio.get_event_loop().time() - start_time
        
        # 超过最大等待时间
        if elapsed >= max_wait:
            return elapsed
        
        # 检测条件
        try:
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()
            
            if result:
                return elapsed
        except:
            pass
        
        # 等待下一次检测
        await asyncio.sleep(check_interval)


async def smart_sleep(
    base_time: float,
    condition_func: Optional[Callable[[], Any]] = None,
    max_time: Optional[float] = None
) -> float:
    """智能延时
    
    根据条件动态调整延时：
    - 如果没有条件函数，等待 base_time
    - 如果有条件函数，快速检测直到条件满足或达到 max_time
    
    Args:
        base_time: 基础等待时间（秒）
        condition_func: 可选的条件函数
        max_time: 最大等待时间（秒），默认为 base_time * 2
        
    Returns:
        实际等待的时间（秒）
    """
    if not condition_func:
        await asyncio.sleep(base_time)
        return base_time
    
    if max_time is None:
        max_time = base_time * 2
    
    return await wait_after_action(
        min_wait=base_time * 0.3,  # 最小等待基础时间的30%
        max_wait=max_time,
        check_func=condition_func,
        check_interval=0.2
    )


async def wait_for_page(
    device_id: str,
    detector,
    expected_states: list,
    timeout: float = 5.0,
    log_callback: Optional[Callable[[str], None]] = None
) -> bool:
    """等待页面到达指定状态（兼容旧代码）
    
    Args:
        device_id: 设备ID
        detector: 页面检测器
        expected_states: 期望的页面状态列表
        timeout: 超时时间（秒）
        log_callback: 日志回调函数
        
    Returns:
        bool: 是否成功到达期望状态
    """
    if log_callback:
        log_callback(f"等待页面状态: {[s.value if hasattr(s, 'value') else str(s) for s in expected_states]}")
    
    start_time = asyncio.get_event_loop().time()
    
    while True:
        try:
            # 检测当前页面状态
            result = await detector.detect_page(device_id, use_cache=False)
            
            if result and result.state in expected_states:
                if log_callback:
                    log_callback(f"✓ 已到达期望页面: {result.state.value}")
                return True
            
            # 检查超时
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= timeout:
                if log_callback:
                    current_state = result.state.value if result else "未知"
                    log_callback(f"⚠️ 等待超时，当前页面: {current_state}")
                return False
            
            # 等待下一次检测
            await asyncio.sleep(0.1)  # 优化：减少到100ms，提高响应速度
            
        except Exception as e:
            if log_callback:
                log_callback(f"⚠️ 页面检测异常: {e}")
            return False
