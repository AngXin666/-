"""
日志记录模块
Logger Module
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class SilentLogger:
    """静默日志记录器 - 只写入文件，不显示在控制台"""
    
    def __init__(self, log_dir: str = "./logs"):
        """初始化静默日志记录器
        
        Args:
            log_dir: 日志目录
        """
        self.log_dir = log_dir
        
        # 创建日志目录
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        # 创建日志文件路径
        log_filename = f"debug_{datetime.now().strftime('%Y%m%d')}.log"
        self.log_path = Path(log_dir) / log_filename
    
    def log(self, message: str, level: str = "INFO") -> None:
        """记录日志到文件（不显示在控制台）
        
        Args:
            message: 日志消息
            level: 日志级别
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_line = f"{timestamp} | {level:8s} | {message}\n"
        
        try:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(log_line)
        except Exception:
            pass  # 静默失败，不影响主程序
    
    def info(self, message: str) -> None:
        """记录INFO级别日志"""
        self.log(message, "INFO")
    
    def debug(self, message: str) -> None:
        """记录DEBUG级别日志"""
        self.log(message, "DEBUG")
    
    def warning(self, message: str) -> None:
        """记录WARNING级别日志"""
        self.log(message, "WARNING")
    
    def error(self, message: str) -> None:
        """记录ERROR级别日志"""
        self.log(message, "ERROR")


# 全局静默日志实例
_silent_logger: Optional[SilentLogger] = None


def get_silent_logger(log_dir: str = "./logs") -> SilentLogger:
    """获取全局静默日志实例
    
    Args:
        log_dir: 日志目录
        
    Returns:
        静默日志记录器实例
    """
    global _silent_logger
    if _silent_logger is None:
        _silent_logger = SilentLogger(log_dir)
    return _silent_logger


class Logger:
    """日志记录器 - 支持控制台和文件双输出"""
    
    def __init__(self, name: str = "NoxAutomation", 
                 log_dir: str = "./logs",
                 log_level: str = "INFO"):
        """初始化日志记录器
        
        Args:
            name: 日志记录器名称
            log_dir: 日志目录
            log_level: 日志级别
        """
        self.name = name
        self.log_dir = log_dir
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        
        # 创建日志目录
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        # 创建日志记录器
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_level)
        
        # 清除已有的处理器
        self.logger.handlers.clear()
        
        # 设置日志格式
        self.formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(instance_id)-10s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 添加控制台处理器
        self._add_console_handler()
        
        # 添加文件处理器
        self._add_file_handler()
    
    def _add_console_handler(self) -> None:
        """添加控制台处理器"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(self.formatter)
        self.logger.addHandler(console_handler)
    
    def _add_file_handler(self) -> None:
        """添加文件处理器"""
        log_filename = f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
        log_path = Path(self.log_dir) / log_filename
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(self.formatter)
        self.logger.addHandler(file_handler)
    
    def _log(self, level: int, message: str, instance_id: str = "SYSTEM") -> None:
        """记录日志
        
        Args:
            level: 日志级别
            message: 日志消息
            instance_id: 实例 ID
        """
        extra = {'instance_id': instance_id}
        self.logger.log(level, message, extra=extra)
    
    def debug(self, message: str, instance_id: str = "SYSTEM") -> None:
        """记录调试日志"""
        self._log(logging.DEBUG, message, instance_id)
    
    def info(self, message: str, instance_id: str = "SYSTEM") -> None:
        """记录信息日志"""
        self._log(logging.INFO, message, instance_id)
    
    def warning(self, message: str, instance_id: str = "SYSTEM") -> None:
        """记录警告日志"""
        self._log(logging.WARNING, message, instance_id)
    
    def error(self, message: str, instance_id: str = "SYSTEM") -> None:
        """记录错误日志"""
        self._log(logging.ERROR, message, instance_id)
    
    def critical(self, message: str, instance_id: str = "SYSTEM") -> None:
        """记录严重错误日志"""
        self._log(logging.CRITICAL, message, instance_id)

    def log_operation(self, operation: str, result: str, 
                      instance_id: str = "SYSTEM", 
                      details: Optional[str] = None) -> None:
        """记录操作日志
        
        Args:
            operation: 操作名称
            result: 操作结果 (SUCCESS/FAILED)
            instance_id: 实例 ID
            details: 详细信息
        """
        message = f"[{operation}] {result}"
        if details:
            message += f" - {details}"
        
        if result == "SUCCESS":
            self.info(message, instance_id)
        else:
            self.error(message, instance_id)
    
    def log_account_start(self, phone: str, instance_id: str) -> None:
        """记录账号处理开始"""
        self.info(f"开始处理账号: {phone}", instance_id)
    
    def log_account_complete(self, phone: str, success: bool, 
                             instance_id: str, details: Optional[str] = None) -> None:
        """记录账号处理完成"""
        status = "成功" if success else "失败"
        message = f"账号处理完成: {phone} - {status}"
        if details:
            message += f" ({details})"
        
        if success:
            self.info(message, instance_id)
        else:
            self.error(message, instance_id)
    
    def log_emulator_start(self, index: int, success: bool) -> None:
        """记录模拟器启动"""
        instance_id = f"Nox_{index}"
        if success:
            self.info("模拟器启动成功", instance_id)
        else:
            self.error("模拟器启动失败", instance_id)
    
    def log_emulator_stop(self, index: int, success: bool) -> None:
        """记录模拟器停止"""
        instance_id = f"Nox_{index}"
        if success:
            self.info("模拟器已停止", instance_id)
        else:
            self.error("模拟器停止失败", instance_id)
    
    def log_login(self, phone: str, success: bool, 
                  instance_id: str, error: Optional[str] = None) -> None:
        """记录登录操作"""
        if success:
            self.info(f"登录成功: {phone}", instance_id)
        else:
            self.error(f"登录失败: {phone} - {error}", instance_id)
    
    def log_sign_in(self, phone: str, success: bool, already_signed: bool,
                    instance_id: str) -> None:
        """记录签到操作"""
        if already_signed:
            self.info(f"今日已签到: {phone}", instance_id)
        elif success:
            self.info(f"签到成功: {phone}", instance_id)
        else:
            self.error(f"签到失败: {phone}", instance_id)
    
    def log_draw(self, phone: str, draw_count: int, total_amount: float,
                 instance_id: str) -> None:
        """记录抽奖操作"""
        self.info(f"抽奖完成: {phone} - 次数:{draw_count}, 金额:{total_amount}", instance_id)
    
    def log_balance(self, phone: str, balance: Optional[float], 
                    balance_type: str, instance_id: str) -> None:
        """记录余额查询"""
        if balance is not None:
            self.info(f"{balance_type}余额: {phone} - {balance}", instance_id)
        else:
            self.warning(f"余额获取失败: {phone}", instance_id)


# 全局日志实例
_logger: Optional[Logger] = None


def get_logger(name: str = "NoxAutomation", 
               log_dir: str = "./logs",
               log_level: str = "INFO") -> Logger:
    """获取全局日志实例
    
    Args:
        name: 日志记录器名称
        log_dir: 日志目录
        log_level: 日志级别
        
    Returns:
        日志记录器实例
    """
    global _logger
    if _logger is None:
        _logger = Logger(name, log_dir, log_level)
    return _logger
