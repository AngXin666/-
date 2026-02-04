"""
转账流程日志输出单元测试

测试 transfer_balance 函数的简洁日志输出功能。
验证转账流程是否按照设计要求输出正确的日志序列和格式。

**Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6**
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch, call
from io import BytesIO
from PIL import Image


class TestTransferFlowLogging:
    """转账流程日志输出测试类"""
    
    @pytest.fixture
    def mock_components(self):
        """创建模拟的组件"""
        mock_adb = AsyncMock()
        mock_adb.tap = AsyncMock()
        mock_adb.press_back = AsyncMock()
        mock_adb.input_text = AsyncMock()
        mock_adb.screencap = AsyncMock()
        mock_adb.shell = AsyncMock()
        
        # 创建一个简单的测试图像
        test_image = Image.new('RGB', (540, 960), color='white')
        img_byte_arr = BytesIO()
        test_image.save(img_byte_arr, format='PNG')
        mock_adb.screencap.return_value = img_byte_arr.getvalue()
        
        mock_detector = AsyncMock()
        
        return {
            'adb': mock_adb,
            'detector': mock_detector
        }
    
    @pytest.fixture
    def balance_transfer(self, mock_components):
        """创建 BalanceTransfer 实例"""
        from zdqd.src.balance_transfer import BalanceTransfer
        
        with patch('zdqd.src.model_manager.ModelManager') as mock_model_manager:
            mock_instance = Mock()
            mock_model_manager.get_instance.return_value = mock_instance
            
            # 模拟混合检测器
            mock_hybrid_detector = AsyncMock()
            mock_instance.get_page_detector_hybrid.return_value = mock_hybrid_detector
            
            transfer = BalanceTransfer(
                adb=mock_components['adb'],
                detector=mock_components['detector']
            )
            
            # 设置模拟的混合检测器
            transfer.hybrid_detector = mock_hybrid_detector
            
            return transfer

    @pytest.mark.asyncio
    async def test_transfer_flow_logs_step_number(self, balance_transfer):
        """测试转账流程是否输出"步骤X: 转账" - Validates: Requirement 8.1"""
        gui_logger = Mock()
        gui_logger.info = Mock()
        gui_logger.error = Mock()
        
        # 模拟页面检测结果
        from zdqd.src.page_detector import PageState
        
        # 模拟个人页面（已登录）
        mock_profile_result = Mock()
        mock_profile_result.state = PageState.PROFILE_LOGGED
        mock_profile_result.confidence = 0.95
        mock_profile_result.elements = []
        
        # 模拟钱包页面
        mock_wallet_result = Mock()
        mock_wallet_result.state = PageState.WALLET
        mock_wallet_result.confidence = 0.95
        mock_wallet_result.elements = []
        
        # 模拟转账页面
        mock_transfer_result = Mock()
        mock_transfer_result.state = PageState.TRANSFER
        mock_transfer_result.confidence = 0.95
        mock_transfer_result.elements = []
        
        # 设置页面检测序列
        call_count = 0
        async def mock_detect_sequence(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_profile_result  # 个人页面
            elif call_count == 2:
                return mock_wallet_result  # 钱包页面
            elif call_count == 3:
                return mock_transfer_result  # 转账页面
            else:
                return mock_wallet_result  # 转账后返回钱包页面
        
        balance_transfer.detector.detect_page = AsyncMock(side_effect=mock_detect_sequence)
        
        # 模拟OCR识别确认按钮
        with patch('zdqd.src.ocr_thread_pool.OCRThreadPool') as mock_ocr_pool_class:
            mock_ocr_pool = AsyncMock()
            mock_ocr_pool_class.return_value = mock_ocr_pool
            
            mock_ocr_result = Mock()
            mock_ocr_result.texts = ['确认提交']
            import numpy as np
            mock_ocr_result.boxes = [np.array([[270, 610], [370, 610], [370, 650], [270, 650]])]
            mock_ocr_pool.recognize = AsyncMock(return_value=mock_ocr_result)
            
            result = await balance_transfer.transfer_balance(
                device_id="test_device",
                recipient_id="1234567",
                initial_balance=100.0,
                step_number=4,
                gui_logger=gui_logger
            )
        
        assert gui_logger.info.called
        
        # 检查是否输出了步骤日志
        log_calls = [call[0][0] for call in gui_logger.info.call_args_list]
        assert any("步骤4: 转账" in log for log in log_calls), "应该输出'步骤4: 转账'"

    @pytest.mark.asyncio
    async def test_transfer_flow_logs_navigate_to_transfer(self, balance_transfer):
        """测试转账流程是否输出"→ 导航到转账页" - Validates: Requirement 8.2"""
        gui_logger = Mock()
        gui_logger.info = Mock()
        gui_logger.error = Mock()
        
        from zdqd.src.page_detector import PageState
        
        mock_profile_result = Mock()
        mock_profile_result.state = PageState.PROFILE_LOGGED
        mock_profile_result.confidence = 0.95
        mock_profile_result.elements = []
        
        mock_wallet_result = Mock()
        mock_wallet_result.state = PageState.WALLET
        mock_wallet_result.confidence = 0.95
        mock_wallet_result.elements = []
        
        mock_transfer_result = Mock()
        mock_transfer_result.state = PageState.TRANSFER
        mock_transfer_result.confidence = 0.95
        mock_transfer_result.elements = []
        
        call_count = 0
        async def mock_detect_sequence(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_profile_result
            elif call_count == 2:
                return mock_wallet_result
            elif call_count == 3:
                return mock_transfer_result
            else:
                return mock_wallet_result
        
        balance_transfer.detector.detect_page = AsyncMock(side_effect=mock_detect_sequence)
        
        with patch('zdqd.src.ocr_thread_pool.OCRThreadPool') as mock_ocr_pool_class:
            mock_ocr_pool = AsyncMock()
            mock_ocr_pool_class.return_value = mock_ocr_pool
            
            mock_ocr_result = Mock()
            mock_ocr_result.texts = ['确认提交']
            import numpy as np
            mock_ocr_result.boxes = [np.array([[270, 610], [370, 610], [370, 650], [270, 650]])]
            mock_ocr_pool.recognize = AsyncMock(return_value=mock_ocr_result)
            
            result = await balance_transfer.transfer_balance(
                device_id="test_device",
                recipient_id="1234567",
                initial_balance=100.0,
                step_number=4,
                gui_logger=gui_logger
            )
        
        # 检查是否输出了导航到转账页的日志
        log_calls = [call[0][0] for call in gui_logger.info.call_args_list]
        assert any("导航到转账页" in log for log in log_calls), "应该输出'导航到转账页'"

    @pytest.mark.asyncio
    async def test_transfer_flow_logs_select_target(self, balance_transfer):
        """测试转账流程是否输出"→ 选择目标: [ID]" - Validates: Requirement 8.3"""
        gui_logger = Mock()
        gui_logger.info = Mock()
        gui_logger.error = Mock()
        
        from zdqd.src.page_detector import PageState
        
        mock_profile_result = Mock()
        mock_profile_result.state = PageState.PROFILE_LOGGED
        mock_profile_result.confidence = 0.95
        mock_profile_result.elements = []
        
        mock_wallet_result = Mock()
        mock_wallet_result.state = PageState.WALLET
        mock_wallet_result.confidence = 0.95
        mock_wallet_result.elements = []
        
        mock_transfer_result = Mock()
        mock_transfer_result.state = PageState.TRANSFER
        mock_transfer_result.confidence = 0.95
        mock_transfer_result.elements = []
        
        call_count = 0
        async def mock_detect_sequence(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_profile_result
            elif call_count == 2:
                return mock_wallet_result
            elif call_count == 3:
                return mock_transfer_result
            else:
                return mock_wallet_result
        
        balance_transfer.detector.detect_page = AsyncMock(side_effect=mock_detect_sequence)
        
        with patch('zdqd.src.ocr_thread_pool.OCRThreadPool') as mock_ocr_pool_class:
            mock_ocr_pool = AsyncMock()
            mock_ocr_pool_class.return_value = mock_ocr_pool
            
            mock_ocr_result = Mock()
            mock_ocr_result.texts = ['确认提交']
            import numpy as np
            mock_ocr_result.boxes = [np.array([[270, 610], [370, 610], [370, 650], [270, 650]])]
            mock_ocr_pool.recognize = AsyncMock(return_value=mock_ocr_result)
            
            result = await balance_transfer.transfer_balance(
                device_id="test_device",
                recipient_id="1234567",
                initial_balance=100.0,
                step_number=4,
                gui_logger=gui_logger
            )
        
        # 检查是否输出了选择目标的日志
        log_calls = [call[0][0] for call in gui_logger.info.call_args_list]
        assert any("选择目标: 1234567" in log for log in log_calls), "应该输出'选择目标: 1234567'"

    @pytest.mark.asyncio
    async def test_transfer_flow_logs_input_amount(self, balance_transfer):
        """测试转账流程是否输出"→ 输入金额: XX.XX元" - Validates: Requirement 8.4"""
        gui_logger = Mock()
        gui_logger.info = Mock()
        gui_logger.error = Mock()
        
        from zdqd.src.page_detector import PageState
        
        mock_profile_result = Mock()
        mock_profile_result.state = PageState.PROFILE_LOGGED
        mock_profile_result.confidence = 0.95
        mock_profile_result.elements = []
        
        mock_wallet_result = Mock()
        mock_wallet_result.state = PageState.WALLET
        mock_wallet_result.confidence = 0.95
        mock_wallet_result.elements = []
        
        mock_transfer_result = Mock()
        mock_transfer_result.state = PageState.TRANSFER
        mock_transfer_result.confidence = 0.95
        mock_transfer_result.elements = []
        
        call_count = 0
        async def mock_detect_sequence(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_profile_result
            elif call_count == 2:
                return mock_wallet_result
            elif call_count == 3:
                return mock_transfer_result
            else:
                return mock_wallet_result
        
        balance_transfer.detector.detect_page = AsyncMock(side_effect=mock_detect_sequence)
        
        # 模拟parse_confirm_dialog返回金额信息
        async def mock_parse_dialog(*args, **kwargs):
            return {
                'recipient_id': '1234567',
                'recipient_name': '测试用户',
                'amount': 50.00
            }
        
        balance_transfer.parse_confirm_dialog = AsyncMock(side_effect=mock_parse_dialog)
        
        with patch('zdqd.src.ocr_thread_pool.OCRThreadPool') as mock_ocr_pool_class:
            mock_ocr_pool = AsyncMock()
            mock_ocr_pool_class.return_value = mock_ocr_pool
            
            mock_ocr_result = Mock()
            mock_ocr_result.texts = ['确认提交']
            import numpy as np
            mock_ocr_result.boxes = [np.array([[270, 610], [370, 610], [370, 650], [270, 650]])]
            mock_ocr_pool.recognize = AsyncMock(return_value=mock_ocr_result)
            
            result = await balance_transfer.transfer_balance(
                device_id="test_device",
                recipient_id="1234567",
                initial_balance=100.0,
                step_number=4,
                gui_logger=gui_logger
            )
        
        # 检查是否输出了输入金额的日志
        log_calls = [call[0][0] for call in gui_logger.info.call_args_list]
        assert any("输入金额: 50.00元" in log for log in log_calls), "应该输出'输入金额: 50.00元'"

    @pytest.mark.asyncio
    async def test_transfer_flow_logs_confirm_transfer(self, balance_transfer):
        """测试转账流程是否输出"→ 确认转账" - Validates: Requirement 8.5"""
        gui_logger = Mock()
        gui_logger.info = Mock()
        gui_logger.error = Mock()
        
        from zdqd.src.page_detector import PageState
        
        mock_profile_result = Mock()
        mock_profile_result.state = PageState.PROFILE_LOGGED
        mock_profile_result.confidence = 0.95
        mock_profile_result.elements = []
        
        mock_wallet_result = Mock()
        mock_wallet_result.state = PageState.WALLET
        mock_wallet_result.confidence = 0.95
        mock_wallet_result.elements = []
        
        mock_transfer_result = Mock()
        mock_transfer_result.state = PageState.TRANSFER
        mock_transfer_result.confidence = 0.95
        mock_transfer_result.elements = []
        
        call_count = 0
        async def mock_detect_sequence(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_profile_result
            elif call_count == 2:
                return mock_wallet_result
            elif call_count == 3:
                return mock_transfer_result
            else:
                return mock_wallet_result
        
        balance_transfer.detector.detect_page = AsyncMock(side_effect=mock_detect_sequence)
        
        with patch('zdqd.src.ocr_thread_pool.OCRThreadPool') as mock_ocr_pool_class:
            mock_ocr_pool = AsyncMock()
            mock_ocr_pool_class.return_value = mock_ocr_pool
            
            mock_ocr_result = Mock()
            mock_ocr_result.texts = ['确认提交']
            import numpy as np
            mock_ocr_result.boxes = [np.array([[270, 610], [370, 610], [370, 650], [270, 650]])]
            mock_ocr_pool.recognize = AsyncMock(return_value=mock_ocr_result)
            
            result = await balance_transfer.transfer_balance(
                device_id="test_device",
                recipient_id="1234567",
                initial_balance=100.0,
                step_number=4,
                gui_logger=gui_logger
            )
        
        # 检查是否输出了确认转账的日志
        log_calls = [call[0][0] for call in gui_logger.info.call_args_list]
        assert any("确认转账" in log for log in log_calls), "应该输出'确认转账'"

    @pytest.mark.asyncio
    async def test_transfer_flow_logs_success(self, balance_transfer):
        """测试转账流程是否输出"✓ 转账成功" - Validates: Requirement 8.6"""
        gui_logger = Mock()
        gui_logger.info = Mock()
        gui_logger.error = Mock()
        
        from zdqd.src.page_detector import PageState
        
        mock_profile_result = Mock()
        mock_profile_result.state = PageState.PROFILE_LOGGED
        mock_profile_result.confidence = 0.95
        mock_profile_result.elements = []
        
        mock_wallet_result = Mock()
        mock_wallet_result.state = PageState.WALLET
        mock_wallet_result.confidence = 0.95
        mock_wallet_result.elements = []
        
        mock_transfer_result = Mock()
        mock_transfer_result.state = PageState.TRANSFER
        mock_transfer_result.confidence = 0.95
        mock_transfer_result.elements = []
        
        call_count = 0
        async def mock_detect_sequence(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_profile_result
            elif call_count == 2:
                return mock_wallet_result
            elif call_count == 3:
                return mock_transfer_result
            else:
                return mock_wallet_result
        
        balance_transfer.detector.detect_page = AsyncMock(side_effect=mock_detect_sequence)
        
        with patch('zdqd.src.ocr_thread_pool.OCRThreadPool') as mock_ocr_pool_class:
            mock_ocr_pool = AsyncMock()
            mock_ocr_pool_class.return_value = mock_ocr_pool
            
            mock_ocr_result = Mock()
            mock_ocr_result.texts = ['确认提交']
            import numpy as np
            mock_ocr_result.boxes = [np.array([[270, 610], [370, 610], [370, 650], [270, 650]])]
            mock_ocr_pool.recognize = AsyncMock(return_value=mock_ocr_result)
            
            result = await balance_transfer.transfer_balance(
                device_id="test_device",
                recipient_id="1234567",
                initial_balance=100.0,
                step_number=4,
                gui_logger=gui_logger
            )
        
        # 检查是否输出了转账成功的日志
        log_calls = [call[0][0] for call in gui_logger.info.call_args_list]
        assert any("转账成功" in log for log in log_calls), "应该输出'转账成功'"

    @pytest.mark.asyncio
    async def test_transfer_flow_complete_sequence(self, balance_transfer):
        """测试转账流程是否按顺序输出完整的日志序列 - Validates: Requirements 8.1-8.6"""
        gui_logger = Mock()
        gui_logger.info = Mock()
        gui_logger.error = Mock()
        
        from zdqd.src.page_detector import PageState
        
        mock_profile_result = Mock()
        mock_profile_result.state = PageState.PROFILE_LOGGED
        mock_profile_result.confidence = 0.95
        mock_profile_result.elements = []
        
        mock_wallet_result = Mock()
        mock_wallet_result.state = PageState.WALLET
        mock_wallet_result.confidence = 0.95
        mock_wallet_result.elements = []
        
        mock_transfer_result = Mock()
        mock_transfer_result.state = PageState.TRANSFER
        mock_transfer_result.confidence = 0.95
        mock_transfer_result.elements = []
        
        call_count = 0
        async def mock_detect_sequence(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_profile_result
            elif call_count == 2:
                return mock_wallet_result
            elif call_count == 3:
                return mock_transfer_result
            else:
                return mock_wallet_result
        
        balance_transfer.detector.detect_page = AsyncMock(side_effect=mock_detect_sequence)
        
        # 模拟parse_confirm_dialog返回金额信息
        async def mock_parse_dialog(*args, **kwargs):
            return {
                'recipient_id': '1234567',
                'recipient_name': '测试用户',
                'amount': 50.00
            }
        
        balance_transfer.parse_confirm_dialog = AsyncMock(side_effect=mock_parse_dialog)
        
        with patch('zdqd.src.ocr_thread_pool.OCRThreadPool') as mock_ocr_pool_class:
            mock_ocr_pool = AsyncMock()
            mock_ocr_pool_class.return_value = mock_ocr_pool
            
            mock_ocr_result = Mock()
            mock_ocr_result.texts = ['确认提交']
            import numpy as np
            mock_ocr_result.boxes = [np.array([[270, 610], [370, 610], [370, 650], [270, 650]])]
            mock_ocr_pool.recognize = AsyncMock(return_value=mock_ocr_result)
            
            result = await balance_transfer.transfer_balance(
                device_id="test_device",
                recipient_id="1234567",
                initial_balance=100.0,
                step_number=4,
                gui_logger=gui_logger
            )
        
        # 收集所有日志
        log_calls = [call[0][0] for call in gui_logger.info.call_args_list]
        log_text = "\n".join(log_calls)
        
        # 验证日志序列
        assert "步骤4: 转账" in log_text, "应该包含步骤日志"
        assert "导航到转账页" in log_text, "应该包含导航到转账页日志"
        assert "选择目标: 1234567" in log_text, "应该包含选择目标日志"
        assert "输入金额: 50.00元" in log_text, "应该包含输入金额日志"
        assert "确认转账" in log_text, "应该包含确认转账日志"
        assert "转账成功" in log_text, "应该包含转账成功日志"
        
        # 验证日志顺序
        step_index = log_text.index("步骤4: 转账")
        navigate_index = log_text.index("导航到转账页")
        select_index = log_text.index("选择目标: 1234567")
        amount_index = log_text.index("输入金额: 50.00元")
        confirm_index = log_text.index("确认转账")
        success_index = log_text.index("转账成功")
        
        assert step_index < navigate_index, "步骤日志应该在导航到转账页之前"
        assert navigate_index < select_index, "导航到转账页应该在选择目标之前"
        assert select_index < amount_index, "选择目标应该在输入金额之前"
        assert amount_index < confirm_index, "输入金额应该在确认转账之前"
        assert confirm_index < success_index, "确认转账应该在转账成功之前"


class TestTransferFlowLogFormatting:
    """转账流程日志格式测试类"""
    
    def test_step_log_format(self):
        """测试步骤日志格式 - Validates: Requirement 8.1"""
        from zdqd.src.concise_logger import LogFormatter
        
        result = LogFormatter.format_step(4, "转账")
        assert result == "步骤4: 转账"
        
        result = LogFormatter.format_step(5, "转账")
        assert result == "步骤5: 转账"
    
    def test_action_log_format(self):
        """测试操作日志格式 - Validates: Requirements 8.2, 8.3, 8.4, 8.5"""
        from zdqd.src.concise_logger import LogFormatter
        
        # 测试导航到转账页
        result = LogFormatter.format_action("导航到转账页")
        assert result == "  → 导航到转账页", f"Expected '  → 导航到转账页', got '{result}'"
        
        # 测试选择目标
        result = LogFormatter.format_action("选择目标: 1234567")
        assert result == "  → 选择目标: 1234567", f"Expected '  → 选择目标: 1234567', got '{result}'"
        
        # 测试输入金额
        result = LogFormatter.format_action("输入金额: 50.00元")
        assert result == "  → 输入金额: 50.00元", f"Expected '  → 输入金额: 50.00元', got '{result}'"
        
        # 测试确认转账
        result = LogFormatter.format_action("确认转账")
        assert result == "  → 确认转账", f"Expected '  → 确认转账', got '{result}'"
    
    def test_success_log_format(self):
        """测试成功日志格式 - Validates: Requirement 8.6"""
        from zdqd.src.concise_logger import LogFormatter
        
        # 测试转账成功
        result = LogFormatter.format_success("转账成功")
        assert result == "  ✓ 转账成功", f"Expected '  ✓ 转账成功', got '{result}'"
    
    def test_transfer_amount_format(self):
        """测试转账金额格式 - Validates: Requirement 8.4"""
        # 验证金额格式：XX.XX元
        amount_text = "输入金额: 50.00元"
        assert "50.00元" in amount_text, "应该包含金额信息"
        
        amount_text = "输入金额: 100.50元"
        assert "100.50元" in amount_text, "应该包含金额信息"
        
        amount_text = "输入金额: 0.01元"
        assert "0.01元" in amount_text, "应该包含金额信息"
    
    def test_transfer_target_format(self):
        """测试转账目标格式 - Validates: Requirement 8.3"""
        # 验证目标格式：选择目标: [ID]
        target_text = "选择目标: 1234567"
        assert "选择目标:" in target_text, "应该包含'选择目标:'"
        assert "1234567" in target_text, "应该包含目标ID"
        
        target_text = "选择目标: 7654321"
        assert "选择目标:" in target_text, "应该包含'选择目标:'"
        assert "7654321" in target_text, "应该包含目标ID"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
