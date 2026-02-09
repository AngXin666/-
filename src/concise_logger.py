"""
简洁日志输出模块

提供用户友好的简洁日志接口，同时保留详细的技术日志到文件。
支持为每个账号创建独立的日志文件。
支持日志级别过滤，只记录关键信息。
"""

from typing import Optional, Dict, Any
import logging


class LogFormatter:
    """日志格式化器，提供统一的日志格式"""
    
    @staticmethod
    def format_step(step_number: int, title: str) -> str:
        """
        格式化步骤日志
        
        Args:
            step_number: 步骤编号
            title: 步骤标题
        
        Returns:
            格式化后的步骤日志: "步骤X: 标题"
        """
        return f"步骤{step_number}: {title}"
    
    @staticmethod
    def format_action(description: str) -> str:
        """
        格式化操作日志
        
        Args:
            description: 操作描述
        
        Returns:
            格式化后的操作日志: "  → 操作描述"
        """
        return f"  → {description}"
    
    @staticmethod
    def format_success(description: str, data: Optional[Dict[str, Any]] = None) -> str:
        """
        格式化成功日志
        
        Args:
            description: 完成描述
            data: 关键数据（可选），如 {"余额": "100.00元", "积分": "500"}
        
        Returns:
            格式化后的成功日志: "  ✓ 完成描述 (数据)"
        """
        msg = f"  ✓ {description}"
        if data:
            data_str = ", ".join([f"{k}: {v}" for k, v in data.items()])
            msg += f" ({data_str})"
        return msg
    
    @staticmethod
    def format_error(description: str) -> str:
        """
        格式化错误日志
        
        Args:
            description: 错误描述
        
        Returns:
            格式化后的错误日志: "  ✗ 错误: 描述"
        """
        return f"  ✗ 错误: {description}"


class ConciseLogger:
    """
    简洁日志记录器
    
    提供用户友好的日志输出接口，同时支持详细的文件日志记录。
    支持为每个账号创建独立的日志文件。
    
    日志级别说明：
    - GUI日志：始终显示所有关键信息（步骤、操作、成功、错误）
    - 文件日志：只记录关键信息（步骤、成功、错误），不记录中间操作
    - 账号日志：记录所有信息（完整的处理流程）
    """
    
    def __init__(
        self,
        module_name: str,
        gui_logger: Optional[Any] = None,
        file_logger: Optional[logging.Logger] = None,
        account_logger: Optional[Any] = None,
        phone: Optional[str] = None,
        user_id: Optional[str] = None,
        verbose_file_log: bool = False
    ):
        """
        初始化简洁日志记录器
        
        Args:
            module_name: 模块名称，用于文件日志
            gui_logger: GUI日志记录器（可选）
            file_logger: 文件日志记录器（可选）
            account_logger: 账号日志管理器（可选）
            phone: 手机号（可选，用于账号独立日志）
            user_id: 用户ID（可选，用于账号独立日志）
            verbose_file_log: 是否在文件日志中记录详细信息（默认False，只记录关键信息）
        """
        self.module_name = module_name
        self.gui_logger = gui_logger
        self.file_logger = file_logger
        self.account_logger = account_logger
        self.phone = phone
        self.user_id = user_id
        self.verbose_file_log = verbose_file_log
    
    def step(self, step_number: int, title: str):
        """
        记录步骤开始
        
        Args:
            step_number: 步骤编号
            title: 步骤标题
        
        输出格式: 
        ============================
        步骤X: 标题
        """
        # GUI日志：添加分隔线 + 步骤标题
        if self.gui_logger:
            self.gui_logger.info("=" * 60)
            formatted_msg = LogFormatter.format_step(step_number, title)
            self.gui_logger.info(formatted_msg)
        
        # 文件日志：只记录步骤（关键信息）
        if self.file_logger:
            self.file_logger.info(f"[{self.module_name}] 步骤 {step_number}: {title}")
        
        # 账号独立日志：记录所有信息
        if self.account_logger and self.phone:
            self.account_logger.log_step(self.phone, step_number, title, self.user_id)
    
    def action(self, description: str):
        """
        记录操作
        
        Args:
            description: 操作描述
        
        输出格式: "  → 操作描述"
        """
        formatted_msg = LogFormatter.format_action(description)
        
        # GUI日志：简洁格式
        if self.gui_logger:
            self.gui_logger.info(formatted_msg)
        
        # 文件日志：只在详细模式下记录操作
        if self.file_logger and self.verbose_file_log:
            self.file_logger.info(f"[{self.module_name}] 操作: {description}")
        
        # 账号独立日志：记录所有信息
        if self.account_logger and self.phone:
            self.account_logger.log_action(self.phone, description, self.user_id)
    
    def success(self, description: str, data: Optional[Dict[str, Any]] = None):
        """
        记录成功完成
        
        Args:
            description: 完成描述
            data: 关键数据（可选），如 {"余额": "100.00元", "积分": "500"}
        
        输出格式: "  ✓ 完成描述 (数据)"
        """
        formatted_msg = LogFormatter.format_success(description, data)
        
        # GUI日志：简洁格式
        if self.gui_logger:
            self.gui_logger.info(formatted_msg)
        
        # 文件日志：记录成功信息（关键信息）
        if self.file_logger:
            if data:
                data_str = ", ".join([f"{k}={v}" for k, v in data.items()])
                self.file_logger.info(f"[{self.module_name}] 成功: {description} [{data_str}]")
            else:
                self.file_logger.info(f"[{self.module_name}] 成功: {description}")
        
        # 账号独立日志：记录所有信息
        if self.account_logger and self.phone:
            self.account_logger.log_success(self.phone, description, self.user_id)
    
    def error(self, description: str, exception: Optional[Exception] = None):
        """
        记录错误
        
        Args:
            description: 错误描述
            exception: 异常对象（可选）
        
        输出格式: "  ✗ 错误: 描述"
        文件日志: 包含完整堆栈信息
        """
        formatted_msg = LogFormatter.format_error(description)
        
        # GUI日志：简洁错误信息
        if self.gui_logger:
            self.gui_logger.error(formatted_msg)
        
        # 文件日志：记录错误信息（关键信息，始终记录）
        if self.file_logger:
            if exception:
                self.file_logger.error(
                    f"[{self.module_name}] 错误: {description} - {str(exception)}",
                    exc_info=True
                )
            else:
                self.file_logger.error(f"[{self.module_name}] 错误: {description}")
        
        # 账号独立日志：记录所有信息
        if self.account_logger and self.phone:
            self.account_logger.log_error(self.phone, description, self.user_id)
    
    def warning(self, description: str):
        """
        记录警告
        
        Args:
            description: 警告描述
        
        输出格式: "  ⚠️ 警告: 描述"
        """
        formatted_msg = f"  ⚠️ {description}"
        
        # GUI日志：显示警告
        if self.gui_logger:
            self.gui_logger.info(formatted_msg)
        
        # 文件日志：记录警告（关键信息）
        if self.file_logger:
            self.file_logger.warning(f"[{self.module_name}] 警告: {description}")
        
        # 账号独立日志：记录所有信息
        if self.account_logger and self.phone:
            self.account_logger.log_warning(self.phone, description, self.user_id)
    
    def debug(self, message: str):
        """
        记录调试信息（仅账号日志）
        
        Args:
            message: 调试信息
        """
        # 调试信息不记录到文件日志（避免过多日志）
        # 只记录到账号独立日志
        if self.account_logger and self.phone:
            self.account_logger.log_info(self.phone, f"[DEBUG] {message}", self.user_id)


