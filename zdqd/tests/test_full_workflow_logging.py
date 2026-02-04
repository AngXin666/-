"""
完整工作流日志输出集成测试

测试 run_full_workflow 函数的简洁日志输出功能。
验证完整工作流是否按照设计要求输出正确的日志序列和格式。

**Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5**
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch, call

from zdqd.src.ximeng_automation import XimengAutomation
from zdqd.src.page_detector_hybrid import PageState
from zdqd.src.models.models import Account


class TestFullWorkflowLogging:
    """完整工作流日志输出测试类"""
    
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
    async def test_workflow_starts_with_begin_message(self, automation):
        """测试工作流开始时输出"开始执行完整流程" - Validates: Requirement 10.4"""
        log_callback = Mock()
        
        # 设置 log_callback 到 automation 实例
        automation._log_callback = log_callback
        
        # 模拟页面检测
        mock_detect_result = Mock()
        mock_detect_result.state = PageState.HOME
        mock_detect_result.confidence = 0.95
        automation.integrated_detector.detect_page = AsyncMock(return_value=mock_detect_result)
        
        # 模拟 profile_reader
        automation.profile_reader = Mock()
        automation.profile_reader.get_full_profile_with_retry = AsyncMock(return_value={
            'balance': 100.00,
            'points': 500,
            'nickname': '测试用户',
            'user_id': 'test_id_123',
            'vouchers': 0,
            'coupons': 0
        })
        automation.profile_reader.get_full_profile_parallel = AsyncMock(return_value={
            'balance': 100.00,
            'points': 500,
            'nickname': '测试用户',
            'vouchers': 0,
            'coupons': 0
        })
        
        # 模拟 daily_checkin
        automation.daily_checkin = Mock()
        automation.daily_checkin.do_checkin = AsyncMock(return_value={
            'success': True,
            'checkin_count': 1,
            'total_times': 5,
            'reward_amount': 1.0,
            'balance_after': 101.00
        })
        
        # 模拟 auto_login
        automation.auto_login = Mock()
        automation.auto_login.enable_cache = False
        automation.auto_login.logout = AsyncMock()
        
        # 模拟账号
        mock_account = Account(phone="test_user", password="test_pass")
        
        result = await automation.run_full_workflow(
            device_id="test_device",
            account=mock_account,
            skip_login=True  # 跳过登录以简化测试
        )
        
        log_calls = [call[0][0] for call in log_callback.call_args_list]
        assert any("开始执行完整流程" in log for log in log_calls), "应该输出'开始执行完整流程'"
    
    @pytest.mark.asyncio
    async def test_workflow_ends_with_complete_message(self, automation):
        """测试工作流结束时输出"流程执行完成" - Validates: Requirement 10.5"""
        log_callback = Mock()
        
        # 设置 log_callback 到 automation 实例
        automation._log_callback = log_callback
        
        # 模拟页面检测
        mock_detect_result = Mock()
        mock_detect_result.state = PageState.HOME
        mock_detect_result.confidence = 0.95
        automation.integrated_detector.detect_page = AsyncMock(return_value=mock_detect_result)
        
        # 模拟 profile_reader
        automation.profile_reader = Mock()
        automation.profile_reader.get_full_profile_with_retry = AsyncMock(return_value={
            'balance': 100.00,
            'points': 500,
            'nickname': '测试用户',
            'user_id': 'test_id_123',
            'vouchers': 0,
            'coupons': 0
        })
        automation.profile_reader.get_full_profile_parallel = AsyncMock(return_value={
            'balance': 100.00,
            'points': 500,
            'nickname': '测试用户',
            'vouchers': 0,
            'coupons': 0
        })
        
        # 模拟 daily_checkin
        automation.daily_checkin = Mock()
        automation.daily_checkin.do_checkin = AsyncMock(return_value={
            'success': True,
            'checkin_count': 1,
            'total_times': 5,
            'reward_amount': 1.0,
            'balance_after': 101.00
        })
        
        # 模拟 auto_login
        automation.auto_login = Mock()
        automation.auto_login.enable_cache = False
        automation.auto_login.logout = AsyncMock()
        
        # 模拟账号
        mock_account = Account(phone="test_user", password="test_pass")
        
        result = await automation.run_full_workflow(
            device_id="test_device",
            account=mock_account,
            skip_login=True  # 跳过登录以简化测试
        )
        
        log_calls = [call[0][0] for call in log_callback.call_args_list]
        assert any("流程执行完成" in log for log in log_calls), "应该输出'流程执行完成'"
    
    @pytest.mark.asyncio
    async def test_workflow_outputs_all_steps_in_sequence(self, automation):
        """测试工作流按顺序输出所有步骤 - Validates: Requirement 10.1"""
        log_callback = Mock()
        
        # 设置 log_callback 到 automation 实例
        automation._log_callback = log_callback
        
        # 模拟页面检测
        mock_detect_result = Mock()
        mock_detect_result.state = PageState.HOME
        mock_detect_result.confidence = 0.95
        automation.integrated_detector.detect_page = AsyncMock(return_value=mock_detect_result)
        
        # 模拟 profile_reader
        automation.profile_reader = Mock()
        automation.profile_reader.get_full_profile_with_retry = AsyncMock(return_value={
            'balance': 100.00,
            'points': 500,
            'nickname': '测试用户',
            'user_id': 'test_id_123',
            'vouchers': 0,
            'coupons': 0
        })
        automation.profile_reader.get_full_profile_parallel = AsyncMock(return_value={
            'balance': 100.00,
            'points': 500,
            'nickname': '测试用户',
            'vouchers': 0,
            'coupons': 0
        })
        
        # 模拟 daily_checkin
        automation.daily_checkin = Mock()
        automation.daily_checkin.do_checkin = AsyncMock(return_value={
            'success': True,
            'checkin_count': 1,
            'total_times': 5,
            'reward_amount': 1.0,
            'balance_after': 101.00
        })
        
        # 模拟 auto_login
        automation.auto_login = Mock()
        automation.auto_login.enable_cache = False
        automation.auto_login.logout = AsyncMock()
        
        # 模拟账号
        mock_account = Account(phone="test_user", password="test_pass")
        
        result = await automation.run_full_workflow(
            device_id="test_device",
            account=mock_account,
            skip_login=True  # 跳过登录以简化测试
        )
        
        log_calls = [call[0][0] for call in log_callback.call_args_list]
        
        # 验证步骤按顺序输出
        step_logs = [log for log in log_calls if "步骤" in log]
        
        # 应该至少有步骤1（登录账号）
        assert any("步骤1: 登录账号" in log for log in step_logs), "应该输出'步骤1: 登录账号'"
        
        # 如果有多个步骤，验证步骤编号是递增的
        if len(step_logs) > 1:
            step_numbers = []
            for log in step_logs:
                if "步骤" in log:
                    try:
                        # 提取步骤编号
                        num_str = log.split("步骤")[1].split(":")[0].strip()
                        step_numbers.append(int(num_str))
                    except:
                        pass
            
            # 验证步骤编号是递增的
            if len(step_numbers) > 1:
                for i in range(len(step_numbers) - 1):
                    assert step_numbers[i] < step_numbers[i + 1], f"步骤编号应该递增，但发现 {step_numbers[i]} >= {step_numbers[i + 1]}"
    
    @pytest.mark.asyncio
    async def test_workflow_each_step_has_completion_status(self, automation):
        """测试每个步骤都有完成状态 - Validates: Requirement 10.2"""
        log_callback = Mock()
        
        # 设置 log_callback 到 automation 实例
        automation._log_callback = log_callback
        
        # 模拟页面检测
        mock_detect_result = Mock()
        mock_detect_result.state = PageState.HOME
        mock_detect_result.confidence = 0.95
        automation.integrated_detector.detect_page = AsyncMock(return_value=mock_detect_result)
        
        # 模拟 profile_reader
        automation.profile_reader = Mock()
        automation.profile_reader.get_full_profile_with_retry = AsyncMock(return_value={
            'balance': 100.00,
            'points': 500,
            'nickname': '测试用户',
            'user_id': 'test_id_123',
            'vouchers': 0,
            'coupons': 0
        })
        automation.profile_reader.get_full_profile_parallel = AsyncMock(return_value={
            'balance': 100.00,
            'points': 500,
            'nickname': '测试用户',
            'vouchers': 0,
            'coupons': 0
        })
        
        # 模拟 daily_checkin
        automation.daily_checkin = Mock()
        automation.daily_checkin.do_checkin = AsyncMock(return_value={
            'success': True,
            'checkin_count': 1,
            'total_times': 5,
            'reward_amount': 1.0,
            'balance_after': 101.00
        })
        
        # 模拟 auto_login
        automation.auto_login = Mock()
        automation.auto_login.enable_cache = False
        automation.auto_login.logout = AsyncMock()
        
        # 模拟账号
        mock_account = Account(phone="test_user", password="test_pass")
        
        result = await automation.run_full_workflow(
            device_id="test_device",
            account=mock_account,
            skip_login=True  # 跳过登录以简化测试
        )
        
        log_calls = [call[0][0] for call in log_callback.call_args_list]
        
        # 验证至少有一个完成状态（✓）
        success_logs = [log for log in log_calls if "✓" in log]
        assert len(success_logs) > 0, "应该至少有一个完成状态日志"
        
        # 验证有登录成功或资料获取完成的状态
        assert any("登录成功" in log or "资料获取完成" in log for log in success_logs), "应该有登录或资料获取的完成状态"
    
    @pytest.mark.asyncio
    async def test_workflow_summary_at_end(self, automation):
        """测试工作流结束时输出最终汇总信息 - Validates: Requirement 10.3"""
        log_callback = Mock()
        
        # 设置 log_callback 到 automation 实例
        automation._log_callback = log_callback
        
        # 模拟页面检测
        mock_detect_result = Mock()
        mock_detect_result.state = PageState.HOME
        mock_detect_result.confidence = 0.95
        automation.integrated_detector.detect_page = AsyncMock(return_value=mock_detect_result)
        
        # 模拟 profile_reader
        automation.profile_reader = Mock()
        automation.profile_reader.get_full_profile_with_retry = AsyncMock(return_value={
            'balance': 100.00,
            'points': 500,
            'nickname': '测试用户',
            'user_id': 'test_id_123',
            'vouchers': 0,
            'coupons': 0
        })
        automation.profile_reader.get_full_profile_parallel = AsyncMock(return_value={
            'balance': 100.00,
            'points': 500,
            'nickname': '测试用户',
            'vouchers': 0,
            'coupons': 0
        })
        
        # 模拟 daily_checkin
        automation.daily_checkin = Mock()
        automation.daily_checkin.do_checkin = AsyncMock(return_value={
            'success': True,
            'checkin_count': 1,
            'total_times': 5,
            'reward_amount': 1.0,
            'balance_after': 101.00
        })
        
        # 模拟 auto_login
        automation.auto_login = Mock()
        automation.auto_login.enable_cache = False
        automation.auto_login.logout = AsyncMock()
        
        # 模拟账号
        mock_account = Account(phone="test_user", password="test_pass")
        
        result = await automation.run_full_workflow(
            device_id="test_device",
            account=mock_account,
            skip_login=True  # 跳过登录以简化测试
        )
        
        log_calls = [call[0][0] for call in log_callback.call_args_list]
        
        # 验证有汇总信息（流程执行完成）
        assert any("流程执行完成" in log for log in log_calls), "应该输出最终汇总信息"


class TestWorkflowLogFormatConsistency:
    """工作流日志格式一致性测试类"""
    
    def test_workflow_log_format_consistency(self):
        """测试工作流日志格式一致性"""
        from zdqd.src.concise_logger import LogFormatter
        
        # 测试步骤格式
        step1 = LogFormatter.format_step(1, "启动应用")
        step2 = LogFormatter.format_step(2, "登录账号")
        step3 = LogFormatter.format_step(3, "获取资料")
        
        assert step1.startswith("步骤1:")
        assert step2.startswith("步骤2:")
        assert step3.startswith("步骤3:")
        
        # 测试操作格式
        action1 = LogFormatter.format_action("关闭服务弹窗")
        action2 = LogFormatter.format_action("导航到登录页")
        action3 = LogFormatter.format_action("进入个人页")
        
        assert action1.startswith("  → ")
        assert action2.startswith("  → ")
        assert action3.startswith("  → ")
        
        # 测试成功格式
        success1 = LogFormatter.format_success("到达首页")
        success2 = LogFormatter.format_success("登录成功")
        success3 = LogFormatter.format_success("资料获取完成")
        
        assert success1.startswith("  ✓ ")
        assert success2.startswith("  ✓ ")
        assert success3.startswith("  ✓ ")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
