"""
错误处理日志输出单元测试

测试各种错误类型是否输出正确的错误日志。
验证文件日志是否包含完整的堆栈信息。

**Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5**
"""

import logging
import pytest
from unittest.mock import Mock
from zdqd.src.concise_logger import ConciseLogger, LogFormatter


class TestErrorLogging:
    """错误日志输出测试类"""
    
    def test_timeout_error_message(self):
        """测试超时错误输出"操作超时" - Validates: Requirement 9.2"""
        gui_logger = Mock()
        file_logger = Mock(spec=logging.Logger)
        
        concise = ConciseLogger("test_module", gui_logger, file_logger)
        
        # 模拟超时错误
        try:
            raise TimeoutError("Operation timed out")
        except TimeoutError as e:
            concise.error("操作超时", e)
        
        # 验证GUI日志输出简洁信息
        gui_logger.error.assert_called_once()
        gui_call_arg = gui_logger.error.call_args[0][0]
        assert "操作超时" in gui_call_arg
        assert "✗" in gui_call_arg
        
        # 验证文件日志包含详细信息
        file_logger.error.assert_called_once()
        file_call_args = file_logger.error.call_args
        assert "操作超时" in file_call_args[0][0]
        assert file_call_args[1]['exc_info'] is True  # 验证包含堆栈信息
    
    def test_element_not_found_error_message(self):
        """测试元素未找到错误输出"未找到目标元素" - Validates: Requirement 9.3"""
        gui_logger = Mock()
        file_logger = Mock(spec=logging.Logger)
        
        concise = ConciseLogger("test_module", gui_logger, file_logger)
        
        # 模拟元素未找到错误
        class ElementNotFoundError(Exception):
            pass
        
        try:
            raise ElementNotFoundError("Element not found")
        except ElementNotFoundError as e:
            concise.error("未找到目标元素", e)
        
        # 验证GUI日志输出简洁信息
        gui_logger.error.assert_called_once()
        gui_call_arg = gui_logger.error.call_args[0][0]
        assert "未找到目标元素" in gui_call_arg
        assert "✗" in gui_call_arg
        
        # 验证文件日志包含详细信息
        file_logger.error.assert_called_once()
        file_call_args = file_logger.error.call_args
        assert "未找到目标元素" in file_call_args[0][0]
        assert file_call_args[1]['exc_info'] is True
    
    def test_network_error_message(self):
        """测试网络错误输出"网络连接失败" - Validates: Requirement 9.4"""
        gui_logger = Mock()
        file_logger = Mock(spec=logging.Logger)
        
        concise = ConciseLogger("test_module", gui_logger, file_logger)
        
        # 模拟网络错误
        class NetworkError(Exception):
            pass
        
        try:
            raise NetworkError("Network connection failed")
        except NetworkError as e:
            concise.error("网络连接失败", e)
        
        # 验证GUI日志输出简洁信息
        gui_logger.error.assert_called_once()
        gui_call_arg = gui_logger.error.call_args[0][0]
        assert "网络连接失败" in gui_call_arg
        assert "✗" in gui_call_arg
        
        # 验证文件日志包含详细信息
        file_logger.error.assert_called_once()
        file_call_args = file_logger.error.call_args
        assert "网络连接失败" in file_call_args[0][0]
        assert file_call_args[1]['exc_info'] is True
    
    def test_generic_exception_message(self):
        """测试通用异常输出"操作失败" - Validates: Requirement 9.1"""
        gui_logger = Mock()
        file_logger = Mock(spec=logging.Logger)
        
        concise = ConciseLogger("test_module", gui_logger, file_logger)
        
        # 模拟通用异常
        try:
            raise Exception("Something went wrong")
        except Exception as e:
            concise.error("操作失败", e)
        
        # 验证GUI日志输出简洁信息
        gui_logger.error.assert_called_once()
        gui_call_arg = gui_logger.error.call_args[0][0]
        assert "操作失败" in gui_call_arg
        assert "✗" in gui_call_arg
        
        # 验证文件日志包含详细信息
        file_logger.error.assert_called_once()
        file_call_args = file_logger.error.call_args
        assert "操作失败" in file_call_args[0][0]
        assert file_call_args[1]['exc_info'] is True
    
    def test_file_log_contains_stack_trace(self):
        """测试文件日志包含完整的错误堆栈信息 - Validates: Requirement 9.5"""
        gui_logger = Mock()
        file_logger = Mock(spec=logging.Logger)
        
        concise = ConciseLogger("test_module", gui_logger, file_logger)
        
        # 创建一个有堆栈的异常
        def inner_function():
            raise ValueError("Test error with stack trace")
        
        def outer_function():
            inner_function()
        
        try:
            outer_function()
        except ValueError as e:
            concise.error("测试错误", e)
        
        # 验证文件日志调用时传入了 exc_info=True
        file_logger.error.assert_called_once()
        file_call_args = file_logger.error.call_args
        
        # 验证包含异常信息
        assert "测试错误" in file_call_args[0][0]
        assert "Test error with stack trace" in file_call_args[0][0]
        
        # 验证 exc_info=True，这会让 logging 记录完整堆栈
        assert file_call_args[1]['exc_info'] is True


class TestErrorLogFormat:
    """错误日志格式测试类"""
    
    def test_error_log_format(self):
        """测试错误日志格式 - Validates: Requirement 9.1"""
        # 测试各种错误描述的格式
        error1 = LogFormatter.format_error("操作超时")
        error2 = LogFormatter.format_error("未找到目标元素")
        error3 = LogFormatter.format_error("网络连接失败")
        error4 = LogFormatter.format_error("操作失败")
        
        # 验证格式一致性
        assert error1 == "  ✗ 错误: 操作超时"
        assert error2 == "  ✗ 错误: 未找到目标元素"
        assert error3 == "  ✗ 错误: 网络连接失败"
        assert error4 == "  ✗ 错误: 操作失败"
        
        # 验证都以相同的前缀开始
        assert all(err.startswith("  ✗ 错误: ") for err in [error1, error2, error3, error4])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
