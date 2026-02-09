"""
账号日志管理器 - 为每个账号创建独立的日志文件
Account Logger Manager - Create separate log files for each account
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import threading


class AccountLogger:
    """账号日志管理器 - 为每个账号创建独立的日志文件"""
    
    # 全局日志记录器缓存
    _loggers: Dict[str, logging.Logger] = {}
    _lock = threading.Lock()
    
    def __init__(self, log_dir: str = "./logs/accounts"):
        """初始化账号日志管理器
        
        Args:
            log_dir: 账号日志目录
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 按日期创建子目录
        self.date_dir = self.log_dir / datetime.now().strftime('%Y%m%d')
        self.date_dir.mkdir(parents=True, exist_ok=True)
    
    def get_logger(self, phone: str, user_id: Optional[str] = None) -> logging.Logger:
        """获取或创建账号专属的日志记录器
        
        Args:
            phone: 手机号
            user_id: 用户ID（可选）
            
        Returns:
            logging.Logger: 账号专属的日志记录器
        """
        # 生成日志记录器名称
        if user_id:
            logger_name = f"account_{phone}_{user_id}"
            log_filename = f"{phone}_{user_id}.log"
        else:
            logger_name = f"account_{phone}"
            log_filename = f"{phone}.log"
        
        # 检查缓存
        with self._lock:
            if logger_name in self._loggers:
                return self._loggers[logger_name]
            
            # 创建新的日志记录器
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.INFO)
            logger.handlers.clear()
            logger.propagate = False
            
            # 创建日志文件路径
            log_path = self.date_dir / log_filename
            
            # 创建文件处理器
            file_handler = logging.FileHandler(log_path, encoding='utf-8', mode='a')
            file_handler.setLevel(logging.INFO)
            
            # 设置日志格式
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            
            # 缓存日志记录器
            self._loggers[logger_name] = logger
            
            return logger
    
    def log_account_start(self, phone: str, user_id: Optional[str] = None):
        """记录账号处理开始
        
        Args:
            phone: 手机号
            user_id: 用户ID（可选）
        """
        logger = self.get_logger(phone, user_id)
        logger.info("=" * 60)
        logger.info(f"开始处理账号: {phone}")
        if user_id:
            logger.info(f"用户ID: {user_id}")
        logger.info("=" * 60)
    
    def log_account_end(self, phone: str, success: bool, message: str = "", 
                       user_id: Optional[str] = None):
        """记录账号处理结束
        
        Args:
            phone: 手机号
            success: 是否成功
            message: 结果消息
            user_id: 用户ID（可选）
        """
        logger = self.get_logger(phone, user_id)
        logger.info("=" * 60)
        status = "成功" if success else "失败"
        logger.info(f"账号处理完成: {status}")
        if message:
            logger.info(f"结果: {message}")
        logger.info("=" * 60)
        logger.info("")  # 空行分隔
    
    def log_step(self, phone: str, step_number: int, title: str, 
                user_id: Optional[str] = None):
        """记录步骤
        
        Args:
            phone: 手机号
            step_number: 步骤编号
            title: 步骤标题
            user_id: 用户ID（可选）
        """
        logger = self.get_logger(phone, user_id)
        logger.info(f"步骤{step_number}: {title}")
    
    def log_action(self, phone: str, description: str, 
                  user_id: Optional[str] = None):
        """记录操作
        
        Args:
            phone: 手机号
            description: 操作描述
            user_id: 用户ID（可选）
        """
        logger = self.get_logger(phone, user_id)
        logger.info(f"  → {description}")
    
    def log_success(self, phone: str, description: str, 
                   user_id: Optional[str] = None):
        """记录成功
        
        Args:
            phone: 手机号
            description: 成功描述
            user_id: 用户ID（可选）
        """
        logger = self.get_logger(phone, user_id)
        logger.info(f"  ✓ {description}")
    
    def log_error(self, phone: str, description: str, 
                 user_id: Optional[str] = None):
        """记录错误
        
        Args:
            phone: 手机号
            description: 错误描述
            user_id: 用户ID（可选）
        """
        logger = self.get_logger(phone, user_id)
        logger.error(f"  ✗ {description}")
    
    def log_warning(self, phone: str, description: str, 
                   user_id: Optional[str] = None):
        """记录警告
        
        Args:
            phone: 手机号
            description: 警告描述
            user_id: 用户ID（可选）
        """
        logger = self.get_logger(phone, user_id)
        logger.warning(f"  ⚠️ {description}")
    
    def log_info(self, phone: str, message: str, 
                user_id: Optional[str] = None):
        """记录信息
        
        Args:
            phone: 手机号
            message: 信息内容
            user_id: 用户ID（可选）
        """
        logger = self.get_logger(phone, user_id)
        logger.info(message)
    
    @classmethod
    def cleanup(cls):
        """清理所有日志记录器"""
        with cls._lock:
            for logger in cls._loggers.values():
                for handler in logger.handlers[:]:
                    handler.close()
                    logger.removeHandler(handler)
            cls._loggers.clear()


# 全局账号日志管理器实例
_account_logger: Optional[AccountLogger] = None
_account_logger_lock = threading.Lock()


def get_account_logger(log_dir: str = "./logs/accounts") -> AccountLogger:
    """获取全局账号日志管理器实例
    
    Args:
        log_dir: 账号日志目录
        
    Returns:
        AccountLogger: 账号日志管理器实例
    """
    global _account_logger
    if _account_logger is None:
        with _account_logger_lock:
            if _account_logger is None:
                _account_logger = AccountLogger(log_dir)
    return _account_logger
