"""统一日志配置模块

提供统一的日志格式和配置，确保所有模块的日志输出保持一致。

使用方法:
    from logging_config import setup_logger
    
    logger = setup_logger(__name__)
    logger.info("这是一条信息日志")
    logger.error("这是一条错误日志")
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import threading


# 全局日志配置
_LOG_DIR = Path("logs")
_LOG_LEVEL = logging.INFO
_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
_CONFIGURED_LOGGERS: Dict[str, logging.Logger] = {}
_CONFIG_LOCK = threading.Lock()


def set_log_directory(log_dir: str) -> None:
    """设置日志目录
    
    Args:
        log_dir: 日志目录路径
    """
    global _LOG_DIR
    _LOG_DIR = Path(log_dir)
    _LOG_DIR.mkdir(parents=True, exist_ok=True)


def set_log_level(level: str) -> None:
    """设置全局日志级别
    
    Args:
        level: 日志级别 ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
    """
    global _LOG_LEVEL
    _LOG_LEVEL = getattr(logging, level.upper(), logging.INFO)
    
    # 更新所有已配置的logger
    with _CONFIG_LOCK:
        for logger in _CONFIGURED_LOGGERS.values():
            logger.setLevel(_LOG_LEVEL)


def setup_logger(
    name: str,
    level: Optional[str] = None,
    log_to_file: bool = True,
    log_to_console: bool = True,
    file_prefix: str = "app"
) -> logging.Logger:
    """设置并返回一个配置好的logger
    
    Args:
        name: logger名称，通常使用 __name__
        level: 日志级别，None表示使用全局级别
        log_to_file: 是否输出到文件
        log_to_console: 是否输出到控制台
        file_prefix: 日志文件名前缀
        
    Returns:
        配置好的logger实例
        
    Example:
        logger = setup_logger(__name__)
        logger.info("应用启动")
    """
    with _CONFIG_LOCK:
        # 如果logger已经配置过，直接返回
        if name in _CONFIGURED_LOGGERS:
            return _CONFIGURED_LOGGERS[name]
        
        # 创建logger
        logger = logging.getLogger(name)
        
        # 设置日志级别
        if level:
            logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        else:
            logger.setLevel(_LOG_LEVEL)
        
        # 清除已有的处理器（避免重复）
        logger.handlers.clear()
        
        # 创建格式化器
        formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
        
        # 添加控制台处理器
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logger.level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # 添加文件处理器
        if log_to_file:
            # 确保日志目录存在
            _LOG_DIR.mkdir(parents=True, exist_ok=True)
            
            # 创建日志文件路径（按日期）
            log_filename = f"{file_prefix}_{datetime.now().strftime('%Y%m%d')}.log"
            log_path = _LOG_DIR / log_filename
            
            file_handler = logging.FileHandler(log_path, encoding='utf-8')
            file_handler.setLevel(logger.level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # 防止日志传播到父logger（避免重复输出）
        logger.propagate = False
        
        # 缓存logger
        _CONFIGURED_LOGGERS[name] = logger
        
        return logger


def setup_module_logger(
    module_name: str,
    level: Optional[str] = None
) -> logging.Logger:
    """为模块设置logger（简化版本）
    
    Args:
        module_name: 模块名称，通常使用 __name__
        level: 日志级别（可选）
        
    Returns:
        配置好的logger实例
        
    Example:
        # 在模块顶部
        logger = setup_module_logger(__name__)
    """
    return setup_logger(module_name, level=level)


def get_logger(name: str) -> logging.Logger:
    """获取已配置的logger
    
    如果logger不存在，会自动创建并配置。
    
    Args:
        name: logger名称
        
    Returns:
        logger实例
    """
    with _CONFIG_LOCK:
        if name in _CONFIGURED_LOGGERS:
            return _CONFIGURED_LOGGERS[name]
    
    # 如果不存在，创建新的
    return setup_logger(name)


def configure_third_party_loggers(level: str = 'WARNING') -> None:
    """配置第三方库的日志级别
    
    降低第三方库的日志输出，避免干扰应用日志。
    
    Args:
        level: 第三方库的日志级别
    """
    third_party_loggers = [
        'rapidocr',
        'RapidOCR',
        'ppocr',
        'onnxruntime',
        'PIL',
        'matplotlib',
        'urllib3',
        'requests'
    ]
    
    log_level = getattr(logging, level.upper(), logging.WARNING)
    
    for logger_name in third_party_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)


def reset_logging_config() -> None:
    """重置日志配置
    
    清除所有已配置的logger，用于测试或重新配置。
    """
    with _CONFIG_LOCK:
        for logger in _CONFIGURED_LOGGERS.values():
            logger.handlers.clear()
        _CONFIGURED_LOGGERS.clear()


class LogContext:
    """日志上下文管理器
    
    在特定代码块中临时修改日志级别。
    
    Example:
        logger = setup_logger(__name__)
        
        with LogContext(logger, 'DEBUG'):
            logger.debug("这条调试信息会被输出")
        
        logger.debug("这条调试信息不会被输出（如果原级别是INFO）")
    """
    
    def __init__(self, logger: logging.Logger, level: str):
        """初始化日志上下文
        
        Args:
            logger: logger实例
            level: 临时日志级别
        """
        self.logger = logger
        self.new_level = getattr(logging, level.upper(), logging.INFO)
        self.old_level = logger.level
    
    def __enter__(self):
        """进入上下文，设置新的日志级别"""
        self.logger.setLevel(self.new_level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文，恢复原日志级别"""
        self.logger.setLevel(self.old_level)
        return False


def log_function_call(logger: logging.Logger):
    """函数调用日志装饰器
    
    自动记录函数的调用和返回。
    
    Args:
        logger: logger实例
        
    Example:
        logger = setup_logger(__name__)
        
        @log_function_call(logger)
        def my_function(x, y):
            return x + y
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f"调用函数: {func.__name__}, 参数: args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"函数返回: {func.__name__}, 结果: {result}")
                return result
            except Exception as e:
                logger.error(f"函数异常: {func.__name__}, 错误: {e}", exc_info=True)
                raise
        return wrapper
    return decorator


# 初始化配置
def init_logging(
    log_dir: str = "logs",
    level: str = "INFO",
    configure_third_party: bool = True
) -> None:
    """初始化日志系统
    
    在应用启动时调用，配置全局日志设置。
    
    Args:
        log_dir: 日志目录
        level: 全局日志级别
        configure_third_party: 是否配置第三方库日志
        
    Example:
        # 在应用入口
        from logging_config import init_logging
        
        init_logging(log_dir="logs", level="INFO")
    """
    set_log_directory(log_dir)
    set_log_level(level)
    
    if configure_third_party:
        configure_third_party_loggers()


# 便捷函数：快速获取logger
def quick_logger(name: str = "app") -> logging.Logger:
    """快速获取一个配置好的logger
    
    Args:
        name: logger名称
        
    Returns:
        logger实例
        
    Example:
        from logging_config import quick_logger
        
        logger = quick_logger()
        logger.info("快速日志")
    """
    return setup_logger(name)


if __name__ == '__main__':
    # 测试代码
    init_logging(log_dir="logs", level="DEBUG")
    
    # 创建测试logger
    logger = setup_logger(__name__)
    
    # 测试不同级别的日志
    logger.debug("这是调试信息")
    logger.info("这是普通信息")
    logger.warning("这是警告信息")
    logger.error("这是错误信息")
    logger.critical("这是严重错误信息")
    
    # 测试日志上下文
    logger.setLevel(logging.INFO)
    logger.debug("这条不会显示（INFO级别）")
    
    with LogContext(logger, 'DEBUG'):
        logger.debug("这条会显示（临时DEBUG级别）")
    
    logger.debug("这条又不会显示了（恢复INFO级别）")
    
    print("\n日志测试完成！请查看 logs/ 目录下的日志文件。")
