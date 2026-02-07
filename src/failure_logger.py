#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
失败日志管理器
Failure Logger Manager

记录所有账号处理失败的详细信息到单独的日志文件
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional


class FailureLogger:
    """失败日志管理器 - 记录所有账号处理失败"""
    
    def __init__(self):
        """初始化失败日志管理器"""
        self._init_failure_logger()
    
    def _init_failure_logger(self):
        """初始化失败专用日志记录器"""
        # 创建logs目录
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建失败日志记录器
        self.failure_logger = logging.getLogger("account_failure")
        self.failure_logger.setLevel(logging.ERROR)
        self.failure_logger.propagate = False  # 不传播到父logger
        
        # 清除已有的处理器
        self.failure_logger.handlers.clear()
        
        # 失败日志文件
        failure_log_file = log_dir / f"failure_{datetime.now().strftime('%Y%m%d')}.log"
        failure_handler = logging.FileHandler(failure_log_file, encoding='utf-8')
        failure_handler.setLevel(logging.ERROR)
        failure_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        failure_handler.setFormatter(failure_formatter)
        self.failure_logger.addHandler(failure_handler)
    
    def log_failure(self, 
                   phone: str,
                   user_id: Optional[str] = None,
                   nickname: Optional[str] = None,
                   error_message: str = "",
                   error_type: str = "未知错误"):
        """记录账号处理失败
        
        Args:
            phone: 手机号
            user_id: 用户ID（可选）
            nickname: 昵称（可选）
            error_message: 错误详细信息
            error_type: 错误类型（登录失败、签到失败、转账失败等）
        """
        # 构建失败记录
        failure_record = f"账号: {phone}"
        
        if user_id:
            failure_record += f" | ID: {user_id}"
        
        if nickname:
            failure_record += f" | 昵称: {nickname}"
        
        failure_record += f" | 错误类型: {error_type}"
        
        if error_message:
            failure_record += f" | 详情: {error_message}"
        
        # 记录到失败日志
        self.failure_logger.error(failure_record)


# 全局实例
_failure_logger = None


def get_failure_logger() -> FailureLogger:
    """获取失败日志管理器单例"""
    global _failure_logger
    if _failure_logger is None:
        _failure_logger = FailureLogger()
    return _failure_logger
