"""
启动流程日志输出单元测试

测试 handle_startup_flow_integrated 函数的简洁日志输出功能。
验证启动流程是否按照设计要求输出正确的日志序列和格式。

**Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5**
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch, call

from zdqd.src.ximeng_automation import XimengAutomation
from zdqd.src.page_detector_hybrid import PageState


class TestStartupFlowLogging:
    """启动流程日志输出测试类"""
    
    @pytest.fixture
    def mock_components(self):
        """创建模拟的组件"""
        mock_adb = AsyncMock()
        mock_adb.start_app = AsyncMock(return_value=True)
        mock_adb.stop_app = AsyncMock(return_value=True)
        mock_adb.shell = AsyncMock(return_value="Success")
        mock_adb.tap = AsyncMock()
        mock_adb.press_back = AsyncMock()
        
        mock_ui = Mock()
        mock_ui.adb_bridge = mock_adb
        
        mock_screen = Mock()
        mock_login = Mock()
        
        return {
            'adb': mock_adb,
            'ui': mock_ui,
            'screen': mock_screen,
            'login': mock_login
        }
    
    @pytest.fixture
    def automation(self, mock_components):
        """创建 XimengAutomation 实例"""
        with patch('zdqd.src.model_manager.ModelManager') as mock_model_manager:
            mock_instance = Mock()
            mock_model_manager.get_instance.return_value = mock_instance
            
            mock_detector = Mock()
            mock_instance.get_page_detector_integrated.return_value = mock_detector
            mock_instance.get_page_detector_hybrid.return_value = mock_detector
            mock_instance.get_ocr_thread_pool.return_value = Mock()
            
            automation = XimengAutomation(
                ui_automation=mock_components['ui'],
                screen_capture=mock_components['screen'],
                auto_login=mock_components['login'],
                adb_bridge=mock_components['adb']
            )
            return automation

    @pytest.mark.asyncio
    async def test_startup_flow_logs_step_1(self, automation):
        """测试启动流程是否输出"步骤1: 启动应用" - Validates: Requirement 4.1"""
        log_callback = Mock()
        
        mock_detect_result = Mock()
        mock_detect_result.state = PageState.HOME
        mock_detect_result.confidence = 0.95
        
        automation.integrated_detector.detect_page = AsyncMock(return_value=mock_detect_result)
        
        result = await automation.handle_startup_flow_integrated(
            device_id="test_device",
            log_callback=log_callback
        )
        
        assert result is True
        assert log_callback.called
        
        log_calls = [call[0][0] for call in log_callback.call_args_list]
        assert any("步骤1: 启动应用" in log for log in log_calls), "应该输出'步骤1: 启动应用'"
    
    @pytest.mark.asyncio
    async def test_startup_flow_logs_reach_home(self, automation):
        """测试启动流程是否输出"✓ 到达首页" - Validates: Requirement 4.5"""
        log_callback = Mock()
        
        mock_detect_result = Mock()
        mock_detect_result.state = PageState.HOME
        mock_detect_result.confidence = 0.95
        
        automation.integrated_detector.detect_page = AsyncMock(return_value=mock_detect_result)
        
        result = await automation.handle_startup_flow_integrated(
            device_id="test_device",
            log_callback=log_callback
        )
        
        assert result is True
        
        log_calls = [call[0][0] for call in log_callback.call_args_list]
        assert any("到达首页" in log for log in log_calls), "应该输出'到达首页'"


class TestStartupFlowLogFormatting:
    """启动流程日志格式测试类"""
    
    def test_step_log_format(self):
        """测试步骤日志格式 - Validates: Requirement 4.1"""
        from zdqd.src.concise_logger import LogFormatter
        
        result = LogFormatter.format_step(1, "启动应用")
        assert result == "步骤1: 启动应用"
        
        result = LogFormatter.format_step(2, "登录账号")
        assert result == "步骤2: 登录账号"
    
    def test_action_log_format(self):
        """测试操作日志格式 - Validates: Requirements 4.2, 4.3, 4.4"""
        from zdqd.src.concise_logger import LogFormatter
        
        # 测试关闭服务弹窗
        result = LogFormatter.format_action("关闭服务弹窗")
        assert result == "  → 关闭服务弹窗", f"Expected '  → 关闭服务弹窗', got '{result}'"
        
        # 测试等待广告
        result = LogFormatter.format_action("等待广告")
        assert result == "  → 等待广告", f"Expected '  → 等待广告', got '{result}'"
        
        # 测试关闭首页广告
        result = LogFormatter.format_action("关闭首页广告")
        assert result == "  → 关闭首页广告", f"Expected '  → 关闭首页广告', got '{result}'"
    
    def test_success_log_format(self):
        """测试成功日志格式 - Validates: Requirement 4.5"""
        from zdqd.src.concise_logger import LogFormatter
        
        # 测试到达首页
        result = LogFormatter.format_success("到达首页")
        assert result == "  ✓ 到达首页", f"Expected '  ✓ 到达首页', got '{result}'"
        
        # 测试登录成功
        result = LogFormatter.format_success("登录成功")
        assert result == "  ✓ 登录成功", f"Expected '  ✓ 登录成功', got '{result}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
