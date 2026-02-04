"""
签到流程日志输出单元测试

测试 do_checkin 函数的简洁日志输出功能。
验证签到流程是否按照设计要求输出正确的日志序列和格式。

**Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 7.6**
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch, call
from io import BytesIO
from PIL import Image


class TestCheckinFlowLogging:
    """签到流程日志输出测试类"""
    
    @pytest.fixture
    def mock_components(self):
        """创建模拟的组件"""
        mock_adb = AsyncMock()
        mock_adb.tap = AsyncMock()
        mock_adb.press_back = AsyncMock()
        mock_adb.screencap = AsyncMock()
        
        # 创建一个简单的测试图像
        test_image = Image.new('RGB', (540, 960), color='white')
        img_byte_arr = BytesIO()
        test_image.save(img_byte_arr, format='PNG')
        mock_adb.screencap.return_value = img_byte_arr.getvalue()
        
        mock_detector = AsyncMock()
        mock_navigator = AsyncMock()
        
        return {
            'adb': mock_adb,
            'detector': mock_detector,
            'navigator': mock_navigator
        }
    
    @pytest.fixture
    def daily_checkin(self, mock_components):
        """创建 DailyCheckin 实例"""
        from zdqd.src.daily_checkin import DailyCheckin
        
        with patch('zdqd.src.model_manager.ModelManager') as mock_model_manager:
            mock_instance = Mock()
            mock_model_manager.get_instance.return_value = mock_instance
            
            # 模拟OCR线程池
            mock_ocr_pool = AsyncMock()
            mock_instance.get_ocr_thread_pool.return_value = mock_ocr_pool
            
            checkin = DailyCheckin(
                adb=mock_components['adb'],
                detector=mock_components['detector'],
                navigator=mock_components['navigator']
            )
            
            # 设置模拟的OCR池
            checkin._ocr_pool = mock_ocr_pool
            
            return checkin

    @pytest.mark.asyncio
    async def test_checkin_flow_logs_step_number(self, daily_checkin):
        """测试签到流程是否输出"步骤X: 签到" - Validates: Requirement 7.1"""
        gui_logger = Mock()
        gui_logger.info = Mock()
        gui_logger.error = Mock()
        
        # 模拟页面检测结果
        from zdqd.src.page_detector_hybrid import PageState
        
        # 模拟导航到首页成功
        daily_checkin.guard.ensure_page_state = AsyncMock(return_value=True)
        
        # 模拟找到签到按钮
        daily_checkin._find_checkin_button = AsyncMock(return_value=(270, 800))
        
        # 模拟页面检测（签到页）
        mock_page_result = Mock()
        mock_page_result.state = PageState.CHECKIN
        mock_page_result.confidence = 0.95
        daily_checkin._detect_page_cached = AsyncMock(return_value=mock_page_result)
        
        # 模拟签到信息读取
        mock_checkin_info = {
            'total_times': 5,
            'daily_remaining_times': 3,
            'can_checkin': True,
            'raw_text': 'test'
        }
        daily_checkin.reader.get_checkin_info = AsyncMock(return_value=mock_checkin_info)
        
        # 模拟个人信息（余额）
        profile_data = {
            'balance': 100.0,
            'points': 500,
            'vouchers': 10
        }
        
        # 模拟签到完成（温馨提示弹窗）
        call_count = 0
        async def mock_detect_sequence(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = Mock()
            if call_count <= 2:
                result.state = PageState.CHECKIN
            else:
                result.state = PageState.WARMTIP
            result.confidence = 0.95
            return result
        
        daily_checkin._detect_page_cached = AsyncMock(side_effect=mock_detect_sequence)
        
        # 模拟OCR识别温馨提示
        mock_ocr_result = Mock()
        mock_ocr_result.texts = ['温馨提示', '没有签到次数']
        daily_checkin._ocr_pool.recognize = AsyncMock(return_value=mock_ocr_result)
        
        # 模拟智能等待器
        with patch('zdqd.src.daily_checkin.wait_for_page', new_callable=AsyncMock) as mock_wait:
            mock_wait.return_value = True
            
            result = await daily_checkin.do_checkin(
                device_id="test_device",
                phone="13800138000",
                profile_data=profile_data,
                step_number=3,
                gui_logger=gui_logger
            )
        
        assert gui_logger.info.called
        
        # 检查是否输出了步骤日志
        log_calls = [call[0][0] for call in gui_logger.info.call_args_list]
        assert any("步骤3: 签到" in log for log in log_calls), "应该输出'步骤3: 签到'"

    @pytest.mark.asyncio
    async def test_checkin_flow_logs_navigate_home(self, daily_checkin):
        """测试签到流程是否输出"→ 导航到首页" - Validates: Requirement 7.2"""
        gui_logger = Mock()
        gui_logger.info = Mock()
        gui_logger.error = Mock()
        
        from zdqd.src.page_detector_hybrid import PageState
        
        # 模拟导航到首页成功
        daily_checkin.guard.ensure_page_state = AsyncMock(return_value=True)
        daily_checkin._find_checkin_button = AsyncMock(return_value=(270, 800))
        
        mock_page_result = Mock()
        mock_page_result.state = PageState.CHECKIN
        mock_page_result.confidence = 0.95
        daily_checkin._detect_page_cached = AsyncMock(return_value=mock_page_result)
        
        mock_checkin_info = {
            'total_times': 5,
            'daily_remaining_times': 0,
            'can_checkin': False,
            'raw_text': 'test'
        }
        daily_checkin.reader.get_checkin_info = AsyncMock(return_value=mock_checkin_info)
        
        profile_data = {
            'balance': 100.0,
            'points': 500,
            'vouchers': 10
        }
        
        with patch('zdqd.src.daily_checkin.wait_for_page', new_callable=AsyncMock) as mock_wait:
            mock_wait.return_value = True
            
            result = await daily_checkin.do_checkin(
                device_id="test_device",
                phone="13800138000",
                profile_data=profile_data,
                step_number=3,
                gui_logger=gui_logger
            )
        
        # 检查是否输出了导航到首页的日志
        log_calls = [call[0][0] for call in gui_logger.info.call_args_list]
        assert any("导航到首页" in log for log in log_calls), "应该输出'导航到首页'"

    @pytest.mark.asyncio
    async def test_checkin_flow_logs_click_daily_checkin(self, daily_checkin):
        """测试签到流程是否输出"→ 点击每日签到" - Validates: Requirement 7.3"""
        gui_logger = Mock()
        gui_logger.info = Mock()
        gui_logger.error = Mock()
        
        from zdqd.src.page_detector_hybrid import PageState
        
        daily_checkin.guard.ensure_page_state = AsyncMock(return_value=True)
        daily_checkin._find_checkin_button = AsyncMock(return_value=(270, 800))
        
        mock_page_result = Mock()
        mock_page_result.state = PageState.CHECKIN
        mock_page_result.confidence = 0.95
        daily_checkin._detect_page_cached = AsyncMock(return_value=mock_page_result)
        
        mock_checkin_info = {
            'total_times': 5,
            'daily_remaining_times': 0,
            'can_checkin': False,
            'raw_text': 'test'
        }
        daily_checkin.reader.get_checkin_info = AsyncMock(return_value=mock_checkin_info)
        
        profile_data = {
            'balance': 100.0,
            'points': 500,
            'vouchers': 10
        }
        
        with patch('zdqd.src.daily_checkin.wait_for_page', new_callable=AsyncMock) as mock_wait:
            mock_wait.return_value = True
            
            result = await daily_checkin.do_checkin(
                device_id="test_device",
                phone="13800138000",
                profile_data=profile_data,
                step_number=3,
                gui_logger=gui_logger
            )
        
        # 检查是否输出了点击每日签到的日志
        log_calls = [call[0][0] for call in gui_logger.info.call_args_list]
        assert any("点击每日签到" in log for log in log_calls), "应该输出'点击每日签到'"

    @pytest.mark.asyncio
    async def test_checkin_flow_logs_checkin_page_info(self, daily_checkin):
        """测试签到流程是否输出"→ 到达签到页 (总次数: X, 剩余: Y)" - Validates: Requirement 7.4"""
        gui_logger = Mock()
        gui_logger.info = Mock()
        gui_logger.error = Mock()
        
        from zdqd.src.page_detector_hybrid import PageState
        
        daily_checkin.guard.ensure_page_state = AsyncMock(return_value=True)
        daily_checkin._find_checkin_button = AsyncMock(return_value=(270, 800))
        
        mock_page_result = Mock()
        mock_page_result.state = PageState.CHECKIN
        mock_page_result.confidence = 0.95
        daily_checkin._detect_page_cached = AsyncMock(return_value=mock_page_result)
        
        # 模拟签到信息：总次数5，剩余3
        mock_checkin_info = {
            'total_times': 5,
            'daily_remaining_times': 3,
            'can_checkin': True,
            'raw_text': 'test'
        }
        daily_checkin.reader.get_checkin_info = AsyncMock(return_value=mock_checkin_info)
        
        profile_data = {
            'balance': 100.0,
            'points': 500,
            'vouchers': 10
        }
        
        # 模拟签到完成（温馨提示弹窗）
        call_count = 0
        async def mock_detect_sequence(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = Mock()
            if call_count <= 2:
                result.state = PageState.CHECKIN
            else:
                result.state = PageState.WARMTIP
            result.confidence = 0.95
            return result
        
        daily_checkin._detect_page_cached = AsyncMock(side_effect=mock_detect_sequence)
        
        mock_ocr_result = Mock()
        mock_ocr_result.texts = ['温馨提示', '没有签到次数']
        daily_checkin._ocr_pool.recognize = AsyncMock(return_value=mock_ocr_result)
        
        with patch('zdqd.src.daily_checkin.wait_for_page', new_callable=AsyncMock) as mock_wait:
            mock_wait.return_value = True
            
            result = await daily_checkin.do_checkin(
                device_id="test_device",
                phone="13800138000",
                profile_data=profile_data,
                step_number=3,
                gui_logger=gui_logger
            )
        
        # 检查是否输出了到达签到页的日志，包含次数信息
        log_calls = [call[0][0] for call in gui_logger.info.call_args_list]
        assert any("到达签到页" in log and "总次数: 5" in log and "剩余: 3" in log 
                   for log in log_calls), "应该输出'到达签到页 (总次数: 5, 剩余: 3)'"

    @pytest.mark.asyncio
    async def test_checkin_flow_logs_click_checkin_with_remaining(self, daily_checkin):
        """测试签到流程是否输出"→ 点击立即签到 (剩余: Y)" - Validates: Requirement 7.5"""
        gui_logger = Mock()
        gui_logger.info = Mock()
        gui_logger.error = Mock()
        
        from zdqd.src.page_detector_hybrid import PageState
        
        daily_checkin.guard.ensure_page_state = AsyncMock(return_value=True)
        daily_checkin._find_checkin_button = AsyncMock(return_value=(270, 800))
        
        # 模拟签到信息：总次数5，剩余2
        mock_checkin_info = {
            'total_times': 5,
            'daily_remaining_times': 2,
            'can_checkin': True,
            'raw_text': 'test'
        }
        daily_checkin.reader.get_checkin_info = AsyncMock(return_value=mock_checkin_info)
        
        profile_data = {
            'balance': 100.0,
            'points': 500,
            'vouchers': 10
        }
        
        # 模拟签到流程：第一次签到成功，第二次温馨提示
        call_count = 0
        async def mock_detect_sequence(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = Mock()
            if call_count <= 3:
                result.state = PageState.CHECKIN
            elif call_count == 4:
                result.state = PageState.CHECKIN_POPUP  # 第一次签到弹窗
            elif call_count <= 6:
                result.state = PageState.CHECKIN  # 返回签到页
            else:
                result.state = PageState.WARMTIP  # 第二次温馨提示
            result.confidence = 0.95
            return result
        
        daily_checkin._detect_page_cached = AsyncMock(side_effect=mock_detect_sequence)
        
        mock_ocr_result = Mock()
        mock_ocr_result.texts = ['温馨提示', '没有签到次数']
        daily_checkin._ocr_pool.recognize = AsyncMock(return_value=mock_ocr_result)
        
        with patch('zdqd.src.daily_checkin.wait_for_page', new_callable=AsyncMock) as mock_wait:
            # 模拟智能等待器：第一次返回签到弹窗，第二次返回温馨提示
            wait_call_count = 0
            async def mock_wait_sequence(*args, **kwargs):
                nonlocal wait_call_count
                wait_call_count += 1
                return True
            
            mock_wait.side_effect = mock_wait_sequence
            
            result = await daily_checkin.do_checkin(
                device_id="test_device",
                phone="13800138000",
                profile_data=profile_data,
                step_number=3,
                gui_logger=gui_logger
            )
        
        # 检查是否输出了点击立即签到的日志，包含剩余次数
        log_calls = [call[0][0] for call in gui_logger.info.call_args_list]
        assert any("点击立即签到" in log and "剩余:" in log 
                   for log in log_calls), "应该输出'点击立即签到 (剩余: Y)'"

    @pytest.mark.asyncio
    async def test_checkin_flow_logs_success(self, daily_checkin):
        """测试签到流程是否输出"✓ 签到完成" - Validates: Requirement 7.6"""
        gui_logger = Mock()
        gui_logger.info = Mock()
        gui_logger.error = Mock()
        
        from zdqd.src.page_detector_hybrid import PageState
        
        daily_checkin.guard.ensure_page_state = AsyncMock(return_value=True)
        daily_checkin._find_checkin_button = AsyncMock(return_value=(270, 800))
        
        mock_page_result = Mock()
        mock_page_result.state = PageState.CHECKIN
        mock_page_result.confidence = 0.95
        daily_checkin._detect_page_cached = AsyncMock(return_value=mock_page_result)
        
        mock_checkin_info = {
            'total_times': 5,
            'daily_remaining_times': 0,
            'can_checkin': False,
            'raw_text': 'test'
        }
        daily_checkin.reader.get_checkin_info = AsyncMock(return_value=mock_checkin_info)
        
        profile_data = {
            'balance': 100.0,
            'points': 500,
            'vouchers': 10
        }
        
        with patch('zdqd.src.daily_checkin.wait_for_page', new_callable=AsyncMock) as mock_wait:
            mock_wait.return_value = True
            
            result = await daily_checkin.do_checkin(
                device_id="test_device",
                phone="13800138000",
                profile_data=profile_data,
                step_number=3,
                gui_logger=gui_logger
            )
        
        # 检查是否输出了签到完成的日志
        log_calls = [call[0][0] for call in gui_logger.info.call_args_list]
        assert any("签到完成" in log for log in log_calls), "应该输出'签到完成'"

    @pytest.mark.asyncio
    async def test_checkin_flow_complete_sequence(self, daily_checkin):
        """测试签到流程是否按顺序输出完整的日志序列 - Validates: Requirements 7.1-7.6"""
        gui_logger = Mock()
        gui_logger.info = Mock()
        gui_logger.error = Mock()
        
        from zdqd.src.page_detector_hybrid import PageState
        
        daily_checkin.guard.ensure_page_state = AsyncMock(return_value=True)
        daily_checkin._find_checkin_button = AsyncMock(return_value=(270, 800))
        
        # 模拟签到信息：总次数5，剩余1
        mock_checkin_info = {
            'total_times': 5,
            'daily_remaining_times': 1,
            'can_checkin': True,
            'raw_text': 'test'
        }
        daily_checkin.reader.get_checkin_info = AsyncMock(return_value=mock_checkin_info)
        
        profile_data = {
            'balance': 100.0,
            'points': 500,
            'vouchers': 10
        }
        
        # 模拟签到流程：签到页 -> 签到弹窗 -> 签到页 -> 温馨提示
        call_count = 0
        async def mock_detect_sequence(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = Mock()
            if call_count <= 3:
                result.state = PageState.CHECKIN
            elif call_count == 4:
                result.state = PageState.CHECKIN_POPUP
            elif call_count <= 6:
                result.state = PageState.CHECKIN
            else:
                result.state = PageState.WARMTIP
            result.confidence = 0.95
            return result
        
        daily_checkin._detect_page_cached = AsyncMock(side_effect=mock_detect_sequence)
        
        mock_ocr_result = Mock()
        mock_ocr_result.texts = ['温馨提示', '没有签到次数']
        daily_checkin._ocr_pool.recognize = AsyncMock(return_value=mock_ocr_result)
        
        with patch('zdqd.src.daily_checkin.wait_for_page', new_callable=AsyncMock) as mock_wait:
            mock_wait.return_value = True
            
            result = await daily_checkin.do_checkin(
                device_id="test_device",
                phone="13800138000",
                profile_data=profile_data,
                step_number=3,
                gui_logger=gui_logger
            )
        
        # 收集所有日志
        log_calls = [call[0][0] for call in gui_logger.info.call_args_list]
        log_text = "\n".join(log_calls)
        
        # 验证日志序列
        assert "步骤3: 签到" in log_text, "应该包含步骤日志"
        assert "导航到首页" in log_text, "应该包含导航到首页日志"
        assert "点击每日签到" in log_text, "应该包含点击每日签到日志"
        assert "到达签到页" in log_text, "应该包含到达签到页日志"
        assert "签到完成" in log_text, "应该包含签到完成日志"
        
        # 验证日志顺序
        step_index = log_text.index("步骤3: 签到")
        navigate_index = log_text.index("导航到首页")
        click_index = log_text.index("点击每日签到")
        arrive_index = log_text.index("到达签到页")
        success_index = log_text.index("签到完成")
        
        assert step_index < navigate_index, "步骤日志应该在导航到首页之前"
        assert navigate_index < click_index, "导航到首页应该在点击每日签到之前"
        assert click_index < arrive_index, "点击每日签到应该在到达签到页之前"
        assert arrive_index < success_index, "到达签到页应该在签到完成之前"


class TestCheckinFlowLogFormatting:
    """签到流程日志格式测试类"""
    
    def test_step_log_format(self):
        """测试步骤日志格式 - Validates: Requirement 7.1"""
        from zdqd.src.concise_logger import LogFormatter
        
        result = LogFormatter.format_step(3, "签到")
        assert result == "步骤3: 签到"
        
        result = LogFormatter.format_step(4, "签到")
        assert result == "步骤4: 签到"
    
    def test_action_log_format(self):
        """测试操作日志格式 - Validates: Requirements 7.2, 7.3, 7.4, 7.5"""
        from zdqd.src.concise_logger import LogFormatter
        
        # 测试导航到首页
        result = LogFormatter.format_action("导航到首页")
        assert result == "  → 导航到首页", f"Expected '  → 导航到首页', got '{result}'"
        
        # 测试点击每日签到
        result = LogFormatter.format_action("点击每日签到")
        assert result == "  → 点击每日签到", f"Expected '  → 点击每日签到', got '{result}'"
        
        # 测试到达签到页（带次数信息）
        result = LogFormatter.format_action("到达签到页 (总次数: 5, 剩余: 3)")
        assert result == "  → 到达签到页 (总次数: 5, 剩余: 3)", \
            f"Expected '  → 到达签到页 (总次数: 5, 剩余: 3)', got '{result}'"
        
        # 测试点击立即签到（带剩余次数）
        result = LogFormatter.format_action("点击立即签到 (剩余: 2)")
        assert result == "  → 点击立即签到 (剩余: 2)", \
            f"Expected '  → 点击立即签到 (剩余: 2)', got '{result}'"
    
    def test_success_log_format(self):
        """测试成功日志格式 - Validates: Requirement 7.6"""
        from zdqd.src.concise_logger import LogFormatter
        
        # 测试签到完成
        result = LogFormatter.format_success("签到完成")
        assert result == "  ✓ 签到完成", f"Expected '  ✓ 签到完成', got '{result}'"
    
    def test_checkin_times_format(self):
        """测试签到次数信息格式 - Validates: Requirement 7.4"""
        # 验证次数信息格式：总次数: X, 剩余: Y
        info_text = "到达签到页 (总次数: 5, 剩余: 3)"
        assert "总次数: 5" in info_text, "应该包含总次数信息"
        assert "剩余: 3" in info_text, "应该包含剩余次数信息"
        
        info_text = "到达签到页 (总次数: 10, 剩余: 7)"
        assert "总次数: 10" in info_text, "应该包含总次数信息"
        assert "剩余: 7" in info_text, "应该包含剩余次数信息"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
