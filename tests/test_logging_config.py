"""测试统一日志配置模块

验证日志配置的功能和一致性。
"""

import sys
from pathlib import Path

# 添加 src 目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import logging
import tempfile
import shutil
from logging_config import (
    setup_logger,
    init_logging,
    set_log_level,
    get_logger,
    configure_third_party_loggers,
    reset_logging_config,
    LogContext,
    quick_logger
)


class TestLoggingConfig:
    """测试日志配置功能"""
    
    def setup_method(self):
        """每个测试前重置日志配置"""
        reset_logging_config()
        
        # 创建临时日志目录
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """每个测试后清理"""
        reset_logging_config()
        
        # 清理临时目录
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_setup_logger_basic(self):
        """测试基本的logger设置"""
        logger = setup_logger("test_module", log_to_file=False)
        
        assert logger is not None
        assert logger.name == "test_module"
        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0
    
    def test_setup_logger_with_level(self):
        """测试设置不同的日志级别"""
        logger = setup_logger("test_debug", level="DEBUG", log_to_file=False)
        
        assert logger.level == logging.DEBUG
    
    def test_setup_logger_singleton(self):
        """测试logger单例模式"""
        logger1 = setup_logger("test_singleton", log_to_file=False)
        logger2 = setup_logger("test_singleton", log_to_file=False)
        
        assert logger1 is logger2
    
    def test_setup_logger_with_file(self):
        """测试日志文件输出"""
        init_logging(log_dir=self.temp_dir, level="INFO")
        logger = setup_logger("test_file", log_to_file=True)
        
        logger.info("测试日志消息")
        
        # 检查日志文件是否创建
        log_files = list(Path(self.temp_dir).glob("*.log"))
        assert len(log_files) > 0
    
    def test_set_log_level(self):
        """测试全局日志级别设置"""
        logger = setup_logger("test_level", log_to_file=False)
        
        # 修改全局级别
        set_log_level("DEBUG")
        
        assert logger.level == logging.DEBUG
    
    def test_get_logger(self):
        """测试获取已配置的logger"""
        logger1 = setup_logger("test_get", log_to_file=False)
        logger2 = get_logger("test_get")
        
        assert logger1 is logger2
    
    def test_get_logger_auto_create(self):
        """测试自动创建logger"""
        logger = get_logger("test_auto_create")
        
        assert logger is not None
        assert logger.name == "test_auto_create"
    
    def test_configure_third_party_loggers(self):
        """测试第三方库日志配置"""
        configure_third_party_loggers(level="ERROR")
        
        # 检查第三方logger的级别
        rapidocr_logger = logging.getLogger("rapidocr")
        assert rapidocr_logger.level == logging.ERROR
    
    def test_log_context(self):
        """测试日志上下文管理器"""
        logger = setup_logger("test_context", level="INFO", log_to_file=False)
        
        assert logger.level == logging.INFO
        
        with LogContext(logger, "DEBUG"):
            assert logger.level == logging.DEBUG
        
        assert logger.level == logging.INFO
    
    def test_quick_logger(self):
        """测试快速获取logger"""
        logger = quick_logger("test_quick")
        
        assert logger is not None
        assert logger.name == "test_quick"
    
    def test_init_logging(self):
        """测试日志系统初始化"""
        init_logging(log_dir=self.temp_dir, level="DEBUG", configure_third_party=True)
        
        # 检查日志目录是否创建
        assert Path(self.temp_dir).exists()
        
        # 检查第三方logger是否配置
        rapidocr_logger = logging.getLogger("rapidocr")
        assert rapidocr_logger.level == logging.WARNING
    
    def test_logger_format_consistency(self):
        """测试日志格式一致性"""
        logger1 = setup_logger("module1", log_to_file=False)
        logger2 = setup_logger("module2", log_to_file=False)
        
        # 检查两个logger的格式化器是否一致
        handler1 = logger1.handlers[0]
        handler2 = logger2.handlers[0]
        
        assert handler1.formatter._fmt == handler2.formatter._fmt
    
    def test_logger_no_propagation(self):
        """测试日志不传播到父logger"""
        logger = setup_logger("test_propagate", log_to_file=False)
        
        assert logger.propagate is False
    
    def test_multiple_modules_logging(self):
        """测试多个模块使用统一日志配置"""
        # 模拟多个模块
        logger1 = setup_logger("module.submodule1", log_to_file=False)
        logger2 = setup_logger("module.submodule2", log_to_file=False)
        logger3 = setup_logger("another_module", log_to_file=False)
        
        # 所有logger应该有相同的格式
        assert len(logger1.handlers) > 0
        assert len(logger2.handlers) > 0
        assert len(logger3.handlers) > 0


class TestLoggingIntegration:
    """测试日志配置与其他模块的集成"""
    
    def setup_method(self):
        """每个测试前重置"""
        reset_logging_config()
    
    def teardown_method(self):
        """每个测试后清理"""
        reset_logging_config()
    
    def test_error_handling_module_logging(self):
        """测试error_handling模块使用统一日志"""
        from error_handling import logger as error_logger
        
        assert error_logger is not None
        assert error_logger.name == "error_handling"
    
    def test_resource_manager_module_logging(self):
        """测试resource_manager模块使用统一日志"""
        from resource_manager import logger as resource_logger
        
        assert resource_logger is not None
        assert resource_logger.name == "resource_manager"
    
    def test_selection_manager_module_logging(self):
        """测试selection_manager模块使用统一日志"""
        from src.selection_manager import logger as selection_logger
        
        assert selection_logger is not None
        assert selection_logger.name == "src.selection_manager"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
