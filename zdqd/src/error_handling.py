"""统一错误处理模块

提供装饰器和工具函数用于统一的错误处理、重试机制和降级处理。
"""

import time
import functools
import logging
from typing import Callable, Any, Optional, Type, Tuple, Union
from enum import Enum
from logging_config import setup_logger

# 使用统一的日志配置
logger = setup_logger(__name__)


class ErrorSeverity(Enum):
    """错误严重程度"""
    LOW = "low"           # 低：可以忽略或使用默认值
    MEDIUM = "medium"     # 中：需要记录但不影响主流程
    HIGH = "high"         # 高：影响功能但可以降级
    CRITICAL = "critical" # 严重：必须处理，可能需要中断


def handle_errors(
    logger: logging.Logger,
    error_message: str = "操作失败",
    reraise: bool = False,
    default_return: Any = None,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    log_traceback: bool = True
) -> Callable:
    """统一的错误处理装饰器
    
    Args:
        logger: 日志记录器
        error_message: 错误消息前缀
        reraise: 是否重新抛出异常
        default_return: 发生错误时的默认返回值
        severity: 错误严重程度
        log_traceback: 是否记录完整堆栈跟踪
        
    Returns:
        装饰器函数
        
    Example:
        @handle_errors(logger, "登录失败", reraise=False, default_return=False)
        def login(username, password):
            # 登录逻辑
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 构建错误日志消息
                full_message = f"{error_message}: {type(e).__name__}: {str(e)}"
                
                # 根据严重程度选择日志级别
                if severity == ErrorSeverity.CRITICAL:
                    if log_traceback:
                        logger.critical(full_message, exc_info=True)
                    else:
                        logger.critical(full_message)
                elif severity == ErrorSeverity.HIGH:
                    if log_traceback:
                        logger.error(full_message, exc_info=True)
                    else:
                        logger.error(full_message)
                elif severity == ErrorSeverity.MEDIUM:
                    if log_traceback:
                        logger.warning(full_message, exc_info=True)
                    else:
                        logger.warning(full_message)
                else:  # LOW
                    logger.info(full_message)
                
                # 决定是否重新抛出异常
                if reraise:
                    raise
                
                return default_return
        return wrapper
    return decorator


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    logger: Optional[logging.Logger] = None
) -> Callable:
    """重试装饰器
    
    Args:
        max_attempts: 最大尝试次数
        delay: 初始延迟时间（秒）
        backoff: 延迟时间的倍增因子
        exceptions: 需要重试的异常类型元组
        logger: 日志记录器（可选）
        
    Returns:
        装饰器函数
        
    Example:
        @retry(max_attempts=3, delay=1.0, backoff=2.0)
        def unstable_operation():
            # 可能失败的操作
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts:
                        if logger:
                            logger.warning(
                                f"{func.__name__} 失败 (尝试 {attempt}/{max_attempts}): {e}, "
                                f"将在 {current_delay:.1f} 秒后重试"
                            )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        if logger:
                            logger.error(
                                f"{func.__name__} 在 {max_attempts} 次尝试后仍然失败: {e}"
                            )
            
            # 所有尝试都失败，抛出最后一个异常
            raise last_exception
        return wrapper
    return decorator


def with_fallback(
    fallback_func: Callable,
    logger: Optional[logging.Logger] = None,
    log_message: str = "主操作失败，使用降级方案"
) -> Callable:
    """降级处理装饰器
    
    当主函数失败时，自动调用降级函数。
    
    Args:
        fallback_func: 降级函数
        logger: 日志记录器（可选）
        log_message: 降级时的日志消息
        
    Returns:
        装饰器函数
        
    Example:
        def fallback_get_data():
            return {"default": "data"}
            
        @with_fallback(fallback_get_data, logger)
        def get_data_from_api():
            # 从API获取数据
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if logger:
                    logger.warning(f"{log_message}: {type(e).__name__}: {e}")
                
                # 调用降级函数
                try:
                    return fallback_func(*args, **kwargs)
                except Exception as fallback_error:
                    if logger:
                        logger.error(f"降级方案也失败了: {fallback_error}")
                    raise
        return wrapper
    return decorator


def safe_execute(
    func: Callable,
    *args,
    logger: Optional[logging.Logger] = None,
    error_message: str = "操作失败",
    default_return: Any = None,
    **kwargs
) -> Any:
    """安全执行函数，捕获所有异常
    
    Args:
        func: 要执行的函数
        *args: 函数的位置参数
        logger: 日志记录器（可选）
        error_message: 错误消息
        default_return: 发生错误时的默认返回值
        **kwargs: 函数的关键字参数
        
    Returns:
        函数的返回值，或发生错误时的默认返回值
        
    Example:
        result = safe_execute(risky_function, arg1, arg2, logger=logger, default_return=None)
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if logger:
            logger.error(f"{error_message}: {type(e).__name__}: {e}", exc_info=True)
        return default_return


class ErrorContext:
    """错误上下文管理器
    
    用于在代码块中统一处理错误。
    
    Example:
        with ErrorContext(logger, "数据处理失败"):
            # 处理数据的代码
            process_data()
    """
    
    def __init__(
        self,
        logger: logging.Logger,
        error_message: str = "操作失败",
        reraise: bool = True,
        severity: ErrorSeverity = ErrorSeverity.HIGH
    ):
        """初始化错误上下文
        
        Args:
            logger: 日志记录器
            error_message: 错误消息
            reraise: 是否重新抛出异常
            severity: 错误严重程度
        """
        self.logger = logger
        self.error_message = error_message
        self.reraise = reraise
        self.severity = severity
        self.exception: Optional[Exception] = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.exception = exc_val
            
            # 构建错误日志消息
            full_message = f"{self.error_message}: {exc_type.__name__}: {str(exc_val)}"
            
            # 根据严重程度选择日志级别
            if self.severity == ErrorSeverity.CRITICAL:
                self.logger.critical(full_message, exc_info=True)
            elif self.severity == ErrorSeverity.HIGH:
                self.logger.error(full_message, exc_info=True)
            elif self.severity == ErrorSeverity.MEDIUM:
                self.logger.warning(full_message, exc_info=True)
            else:  # LOW
                self.logger.info(full_message)
            
            # 决定是否抑制异常
            return not self.reraise
        
        return False
