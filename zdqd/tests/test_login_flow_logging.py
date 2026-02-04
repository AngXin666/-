"""
登录流程日志输出单元测试

测试 auto_login 函数的简洁日志输出功能。
验证登录流程是否按照设计要求输出正确的日志序列和格式。

**Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5**
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch, call

from zdqd.src.auto_login import AutoLogin, LoginResult
from zdqd.src.page_detector_hybrid import PageState


class TestLoginFlowLogging:
    """登录流程日志输出测试类"""
    
    @pytest.fixture
    def mock_components(self):
        """创建模拟的组件"""
        mock_adb = AsyncMock()
        mock_adb.tap = AsyncMock()
        mock_adb.input_text = AsyncMock()
        mock_adb.key_event = AsyncMock()
        mock_adb.press_back = AsyncMock()
        
        mock_ui = Mock()
        mock_ui.adb_bridge = mock_adb
        
        mock_screen = Mock()
        
        return {
            'adb': mock_adb,
            'ui': mock_ui,
            'screen': mock_screen
        }
    
    @pytest.fixture
    def auto_login(self, mock_components):
        """创建 AutoLogin 实例"""
        with patch('zdqd.src.model_manager.ModelManager') as mock_model_manager:
            mock_instance = Mock()
            mock_model_manager.get_instance.return_value = mock_instance
            
            mock_detector = Mock()
            mock_instance.get_page_detector_integrated.return_value = mock_detector
            mock_instance.get_page_detector_hybrid.return_value = mock_detector
            
            auto_login = AutoLogin(
                ui_automation=mock_components['ui'],
                screen_capture=mock_components['screen'],
                adb_bridge=mock_components['adb'],
                enable_cache=False
            )
            
            # 设置检测器的模拟行为
            auto_login.detector = mock_detector
            auto_login.hybrid_detector = mock_detector
            
            return auto_login

    @pytest.mark.asyncio
    async def test_login_flow_logs_step_number(self, auto_login):
        """测试登录流程是否输出"步骤X: 登录账号" - Validates: Requirement 5.1"""
        log_callback = Mock()
        gui_logger = Mock()
        gui_logger.info = Mock()
        
        # 模拟页面检测结果
        mock_detect_result = Mock()
        mock_detect_result.state = PageState.LOGIN
        mock_detect_result.confidence = 0.95
        
        auto_login.detector.detect_page = AsyncMock(return_value=mock_detect_result)
        auto_login.page_detector.detect_page = AsyncMock(return_value=mock_detect_result)
        
        # 模拟登录成功后的页面状态
        async def mock_detect_page_sequence(*args, **kwargs):
            # 前几次返回登录页面，最后返回首页
            if not hasattr(mock_detect_page_sequence, 'call_count'):
                mock_detect_page_sequence.call_count = 0
            mock_detect_page_sequence.call_count += 1
            
            if mock_detect_page_sequence.call_count <= 3:
                result = Mock()
                result.state = PageState.LOGIN
                result.confidence = 0.95
                return result
            else:
                result = Mock()
                result.state = PageState.HOME
                result.confidence = 0.95
                return result
        
        auto_login.page_detector.detect_page = AsyncMock(side_effect=mock_detect_page_sequence)
        
        result = await auto_login.login(
            device_id="test_device",
            phone="13800138000",
            password="test123",
            log_callback=log_callback,
            use_cache=False,
            step_number=2,
            gui_logger=gui_logger
        )
        
        assert result.success is True
        assert gui_logger.info.called
        
        # 检查是否输出了步骤日志
        log_calls = [call[0][0] for call in gui_logger.info.call_args_list]
        assert any("步骤2: 登录账号" in log for log in log_calls), "应该输出'步骤2: 登录账号'"
    
    @pytest.mark.asyncio
    async def test_login_flow_logs_input_credentials(self, auto_login):
        """测试登录流程是否输出"→ 输入账号信息" - Validates: Requirement 5.3"""
        log_callback = Mock()
        gui_logger = Mock()
        gui_logger.info = Mock()
        
        # 模拟页面检测结果
        mock_detect_result = Mock()
        mock_detect_result.state = PageState.LOGIN
        mock_detect_result.confidence = 0.95
        
        auto_login.detector.detect_page = AsyncMock(return_value=mock_detect_result)
        
        # 模拟登录成功后的页面状态
        async def mock_detect_page_sequence(*args, **kwargs):
            if not hasattr(mock_detect_page_sequence, 'call_count'):
                mock_detect_page_sequence.call_count = 0
            mock_detect_page_sequence.call_count += 1
            
            if mock_detect_page_sequence.call_count <= 3:
                result = Mock()
                result.state = PageState.LOGIN
                result.confidence = 0.95
                return result
            else:
                result = Mock()
                result.state = PageState.HOME
                result.confidence = 0.95
                return result
        
        auto_login.page_detector.detect_page = AsyncMock(side_effect=mock_detect_page_sequence)
        
        result = await auto_login.login(
            device_id="test_device",
            phone="13800138000",
            password="test123",
            log_callback=log_callback,
            use_cache=False,
            step_number=2,
            gui_logger=gui_logger
        )
        
        assert result.success is True
        
        # 检查是否输出了输入账号信息的日志
        log_calls = [call[0][0] for call in gui_logger.info.call_args_list]
        assert any("输入账号信息" in log for log in log_calls), "应该输出'输入账号信息'"
    
    @pytest.mark.asyncio
    async def test_login_flow_logs_click_login(self, auto_login):
        """测试登录流程是否输出"→ 点击登录" - Validates: Requirement 5.4"""
        log_callback = Mock()
        gui_logger = Mock()
        gui_logger.info = Mock()
        
        # 模拟页面检测结果
        mock_detect_result = Mock()
        mock_detect_result.state = PageState.LOGIN
        mock_detect_result.confidence = 0.95
        
        auto_login.detector.detect_page = AsyncMock(return_value=mock_detect_result)
        
        # 模拟登录成功后的页面状态
        async def mock_detect_page_sequence(*args, **kwargs):
            if not hasattr(mock_detect_page_sequence, 'call_count'):
                mock_detect_page_sequence.call_count = 0
            mock_detect_page_sequence.call_count += 1
            
            if mock_detect_page_sequence.call_count <= 3:
                result = Mock()
                result.state = PageState.LOGIN
                result.confidence = 0.95
                return result
            else:
                result = Mock()
                result.state = PageState.HOME
                result.confidence = 0.95
                return result
        
        auto_login.page_detector.detect_page = AsyncMock(side_effect=mock_detect_page_sequence)
        
        result = await auto_login.login(
            device_id="test_device",
            phone="13800138000",
            password="test123",
            log_callback=log_callback,
            use_cache=False,
            step_number=2,
            gui_logger=gui_logger
        )
        
        assert result.success is True
        
        # 检查是否输出了点击登录的日志
        log_calls = [call[0][0] for call in gui_logger.info.call_args_list]
        assert any("点击登录" in log for log in log_calls), "应该输出'点击登录'"
    
    @pytest.mark.asyncio
    async def test_login_flow_logs_success(self, auto_login):
        """测试登录流程是否输出"✓ 登录成功" - Validates: Requirement 5.5"""
        log_callback = Mock()
        gui_logger = Mock()
        gui_logger.info = Mock()
        
        # 模拟页面检测结果
        mock_detect_result = Mock()
        mock_detect_result.state = PageState.LOGIN
        mock_detect_result.confidence = 0.95
        
        auto_login.detector.detect_page = AsyncMock(return_value=mock_detect_result)
        
        # 模拟登录成功后的页面状态
        async def mock_detect_page_sequence(*args, **kwargs):
            if not hasattr(mock_detect_page_sequence, 'call_count'):
                mock_detect_page_sequence.call_count = 0
            mock_detect_page_sequence.call_count += 1
            
            if mock_detect_page_sequence.call_count <= 3:
                result = Mock()
                result.state = PageState.LOGIN
                result.confidence = 0.95
                return result
            else:
                result = Mock()
                result.state = PageState.HOME
                result.confidence = 0.95
                return result
        
        auto_login.page_detector.detect_page = AsyncMock(side_effect=mock_detect_page_sequence)
        
        result = await auto_login.login(
            device_id="test_device",
            phone="13800138000",
            password="test123",
            log_callback=log_callback,
            use_cache=False,
            step_number=2,
            gui_logger=gui_logger
        )
        
        assert result.success is True
        
        # 检查是否输出了登录成功的日志
        log_calls = [call[0][0] for call in gui_logger.info.call_args_list]
        assert any("登录成功" in log for log in log_calls), "应该输出'登录成功'"
    
    @pytest.mark.asyncio
    async def test_login_flow_complete_sequence(self, auto_login):
        """测试登录流程是否按顺序输出完整的日志序列 - Validates: Requirements 5.1-5.5"""
        log_callback = Mock()
        gui_logger = Mock()
        gui_logger.info = Mock()
        
        # 模拟页面检测结果
        mock_detect_result = Mock()
        mock_detect_result.state = PageState.LOGIN
        mock_detect_result.confidence = 0.95
        
        auto_login.detector.detect_page = AsyncMock(return_value=mock_detect_result)
        
        # 模拟登录成功后的页面状态
        async def mock_detect_page_sequence(*args, **kwargs):
            if not hasattr(mock_detect_page_sequence, 'call_count'):
                mock_detect_page_sequence.call_count = 0
            mock_detect_page_sequence.call_count += 1
            
            if mock_detect_page_sequence.call_count <= 3:
                result = Mock()
                result.state = PageState.LOGIN
                result.confidence = 0.95
                return result
            else:
                result = Mock()
                result.state = PageState.HOME
                result.confidence = 0.95
                return result
        
        auto_login.page_detector.detect_page = AsyncMock(side_effect=mock_detect_page_sequence)
        
        result = await auto_login.login(
            device_id="test_device",
            phone="13800138000",
            password="test123",
            log_callback=log_callback,
            use_cache=False,
            step_number=2,
            gui_logger=gui_logger
        )
        
        assert result.success is True
        
        # 收集所有日志
        log_calls = [call[0][0] for call in gui_logger.info.call_args_list]
        log_text = "\n".join(log_calls)
        
        # 验证日志序列
        assert "步骤2: 登录账号" in log_text, "应该包含步骤日志"
        assert "输入账号信息" in log_text, "应该包含输入账号信息日志"
        assert "点击登录" in log_text, "应该包含点击登录日志"
        assert "登录成功" in log_text, "应该包含登录成功日志"
        
        # 验证日志顺序（步骤应该在操作之前）
        step_index = log_text.index("步骤2: 登录账号")
        input_index = log_text.index("输入账号信息")
        click_index = log_text.index("点击登录")
        success_index = log_text.index("登录成功")
        
        assert step_index < input_index, "步骤日志应该在输入账号信息之前"
        assert input_index < click_index, "输入账号信息应该在点击登录之前"
        assert click_index < success_index, "点击登录应该在登录成功之前"


class TestLoginFlowLogFormatting:
    """登录流程日志格式测试类"""
    
    def test_step_log_format(self):
        """测试步骤日志格式 - Validates: Requirement 5.1"""
        from zdqd.src.concise_logger import LogFormatter
        
        result = LogFormatter.format_step(2, "登录账号")
        assert result == "步骤2: 登录账号"
        
        result = LogFormatter.format_step(3, "登录账号")
        assert result == "步骤3: 登录账号"
    
    def test_action_log_format(self):
        """测试操作日志格式 - Validates: Requirements 5.2, 5.3, 5.4"""
        from zdqd.src.concise_logger import LogFormatter
        
        # 测试导航到登录页
        result = LogFormatter.format_action("导航到登录页")
        assert result == "  → 导航到登录页", f"Expected '  → 导航到登录页', got '{result}'"
        
        # 测试输入账号信息
        result = LogFormatter.format_action("输入账号信息")
        assert result == "  → 输入账号信息", f"Expected '  → 输入账号信息', got '{result}'"
        
        # 测试点击登录
        result = LogFormatter.format_action("点击登录")
        assert result == "  → 点击登录", f"Expected '  → 点击登录', got '{result}'"
    
    def test_success_log_format(self):
        """测试成功日志格式 - Validates: Requirement 5.5"""
        from zdqd.src.concise_logger import LogFormatter
        
        # 测试登录成功
        result = LogFormatter.format_success("登录成功")
        assert result == "  ✓ 登录成功", f"Expected '  ✓ 登录成功', got '{result}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
