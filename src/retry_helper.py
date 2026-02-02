"""
重试辅助模块 - 简单实用的重试机制
Retry Helper Module
"""

import asyncio
import functools
from typing import Callable, Optional, Tuple, Type


def async_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable] = None
):
    """异步重试装饰器
    
    Args:
        max_attempts: 最大尝试次数
        delay: 初始延迟时间（秒）
        backoff: 退避因子（每次重试延迟时间乘以此因子）
        exceptions: 需要重试的异常类型
        on_retry: 重试时的回调函数 (attempt, exception)
    
    Example:
        @async_retry(max_attempts=3, delay=1.0, backoff=2.0)
        async def my_function():
            # 可能失败的操作
            pass
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts:
                        # 最后一次尝试失败，抛出异常
                        raise
                    
                    # 调用重试回调
                    if on_retry:
                        try:
                            on_retry(attempt, e)
                        except:
                            pass
                    
                    # 等待后重试
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            
            # 理论上不会到这里，但为了安全
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator


async def retry_on_false(
    func: Callable,
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    on_retry: Optional[Callable] = None,
    *args,
    **kwargs
) -> bool:
    """重试返回 False 的函数（用于返回布尔值的函数）
    
    Args:
        func: 要重试的异步函数
        max_attempts: 最大尝试次数
        delay: 初始延迟时间（秒）
        backoff: 退避因子
        on_retry: 重试时的回调函数 (attempt)
        *args, **kwargs: 传递给函数的参数
    
    Returns:
        函数执行结果（True/False）
    
    Example:
        result = await retry_on_false(
            adb.tap, 
            max_attempts=3,
            device_id="127.0.0.1:62001",
            x=100, y=200
        )
    """
    current_delay = delay
    
    for attempt in range(1, max_attempts + 1):
        try:
            result = await func(*args, **kwargs)
            if result:
                return True
            
            # 结果为 False，需要重试
            if attempt < max_attempts:
                if on_retry:
                    try:
                        on_retry(attempt)
                    except:
                        pass
                
                await asyncio.sleep(current_delay)
                current_delay *= backoff
        except Exception:
            # 发生异常，也需要重试
            if attempt == max_attempts:
                return False
            
            if on_retry:
                try:
                    on_retry(attempt)
                except:
                    pass
            
            await asyncio.sleep(current_delay)
            current_delay *= backoff
    
    return False


async def retry_until_success(
    func: Callable,
    check_func: Callable,
    max_attempts: int = 3,
    delay: float = 1.0,
    on_retry: Optional[Callable] = None,
    *args,
    **kwargs
) -> bool:
    """重试直到检查函数返回 True
    
    Args:
        func: 要执行的异步函数
        check_func: 检查函数（异步），返回 True 表示成功
        max_attempts: 最大尝试次数
        delay: 每次重试的延迟时间
        on_retry: 重试时的回调函数 (attempt)
        *args, **kwargs: 传递给 func 的参数
    
    Returns:
        是否成功
    
    Example:
        # 点击按钮后检查是否跳转到目标页面
        result = await retry_until_success(
            adb.tap,
            lambda: check_page_state(device_id, PageState.HOME),
            max_attempts=3,
            device_id=device_id,
            x=100, y=200
        )
    """
    for attempt in range(1, max_attempts + 1):
        try:
            # 执行操作
            await func(*args, **kwargs)
            
            # 等待一下让操作生效
            await asyncio.sleep(0.5)
            
            # 检查是否成功
            if await check_func():
                return True
            
            # 未成功，需要重试
            if attempt < max_attempts:
                if on_retry:
                    try:
                        on_retry(attempt)
                    except:
                        pass
                
                await asyncio.sleep(delay)
        except Exception:
            if attempt == max_attempts:
                return False
            
            if on_retry:
                try:
                    on_retry(attempt)
                except:
                    pass
            
            await asyncio.sleep(delay)
    
    return False
