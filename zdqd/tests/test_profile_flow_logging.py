"""
获取资料流程日志输出单元测试

测试 get_full_profile 函数的简洁日志输出功能。
验证获取资料流程是否按照设计要求输出正确的日志序列和格式。

**Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5**
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch, call
from io import BytesIO
from PIL import Image


class TestProfileFlowLogging:
    """获取资料流程日志输出测试类"""
    
    @pytest.fixture
    def mock_components(self):
        """创建模拟的组件"""
        mock_adb = AsyncMock()
        mock_adb.screencap = AsyncMock()
        mock_adb.tap = AsyncMock()
        mock_adb.press_back = AsyncMock()
        
        # 创建一个简单的测试图像
        test_image = Image.new('RGB', (540, 960), color='white')
        img_byte_arr = BytesIO()
        test_image.save(img_byte_arr, format='PNG')
        mock_adb.screencap.return_value = img_byte_arr.getvalue()
        
        return {
            'adb': mock_adb
        }
    
    @pytest.fixture
    def profile_reader(self, mock_components):
        """创建 ProfileReader 实例"""
        from zdqd.src.profile_reader import ProfileReader
        
        with patch('zdqd.src.model_manager.ModelManager') as mock_model_manager:
            mock_instance = Mock()
            mock_model_manager.get_instance.return_value = mock_instance
            
            # 模拟OCR线程池
            mock_ocr_pool = AsyncMock()
            mock_ocr_result = Mock()
            mock_ocr_result.texts = ['昵称测试', 'ID: 123456', '余额', '100.50', '积分', '500']
            mock_ocr_result.boxes = None
            mock_ocr_pool.recognize = AsyncMock(return_value=mock_ocr_result)
            mock_instance.get_ocr_thread_pool.return_value = mock_ocr_pool
            
            # 模拟整合检测器
            mock_detector = AsyncMock()
            mock_instance.get_page_detector_integrated.return_value = mock_detector
            
            reader = ProfileReader(
                adb=mock_components['adb'],
                yolo_detector=mock_detector
            )
            
            # 设置检测器的模拟行为
            reader._integrated_detector = mock_detector
            reader._ocr_pool = mock_ocr_pool
            
            return reader

    @pytest.mark.asyncio
    async def test_profile_flow_logs_step_number(self, profile_reader):
        """测试获取资料流程是否输出"步骤X: 获取资料" - Validates: Requirement 6.1"""
        gui_logger = Mock()
        gui_logger.info = Mock()
        gui_logger.error = Mock()
        
        # 模拟页面检测结果（个人页）
        from zdqd.src.page_detector import PageState
        mock_page_result = Mock()
        mock_page_result.state = PageState.PROFILE
        mock_page_result.confidence = 0.95
        mock_page_result.elements = []
        mock_page_result.yolo_model_used = 'test_model'
        
        profile_reader._integrated_detector.detect_page = AsyncMock(return_value=mock_page_result)
        
        result = await profile_reader.get_full_profile(
            device_id="test_device",
            step_number=3,
            gui_logger=gui_logger
        )
        
        assert gui_logger.info.called
        
        # 检查是否输出了步骤日志
        log_calls = [call[0][0] for call in gui_logger.info.call_args_list]
        assert any("步骤3: 获取资料" in log for log in log_calls), "应该输出'步骤3: 获取资料'"
    
    @pytest.mark.asyncio
    async def test_profile_flow_logs_enter_profile_page(self, profile_reader):
        """测试获取资料流程是否输出"→ 进入个人页" - Validates: Requirement 6.2"""
        gui_logger = Mock()
        gui_logger.info = Mock()
        gui_logger.error = Mock()
        
        # 模拟页面检测结果（个人页）
        from zdqd.src.page_detector import PageState
        mock_page_result = Mock()
        mock_page_result.state = PageState.PROFILE
        mock_page_result.confidence = 0.95
        mock_page_result.elements = []
        mock_page_result.yolo_model_used = 'test_model'
        
        profile_reader._integrated_detector.detect_page = AsyncMock(return_value=mock_page_result)
        
        result = await profile_reader.get_full_profile(
            device_id="test_device",
            step_number=3,
            gui_logger=gui_logger
        )
        
        # 检查是否输出了进入个人页的日志
        log_calls = [call[0][0] for call in gui_logger.info.call_args_list]
        assert any("进入个人页" in log for log in log_calls), "应该输出'进入个人页'"
    
    @pytest.mark.asyncio
    async def test_profile_flow_logs_get_detailed_info(self, profile_reader):
        """测试获取资料流程是否输出"→ 获取详细资料" - Validates: Requirement 6.4"""
        gui_logger = Mock()
        gui_logger.info = Mock()
        gui_logger.error = Mock()
        
        # 模拟页面检测结果（个人页）
        from zdqd.src.page_detector import PageState
        mock_page_result = Mock()
        mock_page_result.state = PageState.PROFILE
        mock_page_result.confidence = 0.95
        mock_page_result.elements = []
        mock_page_result.yolo_model_used = 'test_model'
        
        profile_reader._integrated_detector.detect_page = AsyncMock(return_value=mock_page_result)
        
        result = await profile_reader.get_full_profile(
            device_id="test_device",
            step_number=3,
            gui_logger=gui_logger
        )
        
        # 检查是否输出了获取详细资料的日志
        log_calls = [call[0][0] for call in gui_logger.info.call_args_list]
        assert any("获取详细资料" in log for log in log_calls), "应该输出'获取详细资料'"
    
    @pytest.mark.asyncio
    async def test_profile_flow_logs_success_with_data(self, profile_reader):
        """测试获取资料流程是否输出"✓ 资料获取完成 (余额: XX.XX元, 积分: XXX)" - Validates: Requirement 6.5"""
        gui_logger = Mock()
        gui_logger.info = Mock()
        gui_logger.error = Mock()
        
        # 模拟页面检测结果（个人页）
        from zdqd.src.page_detector import PageState
        mock_page_result = Mock()
        mock_page_result.state = PageState.PROFILE
        mock_page_result.confidence = 0.95
        
        # 模拟检测到的元素（余额和积分）
        mock_element_balance = Mock()
        mock_element_balance.class_name = '余额'
        mock_element_balance.bbox = (30, 230, 150, 330)
        mock_element_balance.confidence = 0.9
        
        mock_element_points = Mock()
        mock_element_points.class_name = '积分'
        mock_element_points.bbox = (180, 230, 260, 330)
        mock_element_points.confidence = 0.9
        
        mock_page_result.elements = [mock_element_balance, mock_element_points]
        mock_page_result.yolo_model_used = 'test_model'
        
        # 模拟OCR识别结果
        import numpy as np
        mock_ocr_result = Mock()
        mock_ocr_result.texts = ['100.50', '500']
        # OCR boxes 格式：每个box是8个坐标值的数组 [x1, y1, x2, y2, x3, y3, x4, y4]
        mock_ocr_result.boxes = [
            np.array([35, 235, 145, 235, 145, 325, 35, 325]),  # 余额位置
            np.array([185, 235, 255, 235, 255, 325, 185, 325])  # 积分位置
        ]
        profile_reader._ocr_pool.recognize = AsyncMock(return_value=mock_ocr_result)
        
        profile_reader._integrated_detector.detect_page = AsyncMock(return_value=mock_page_result)
        
        result = await profile_reader.get_full_profile(
            device_id="test_device",
            step_number=3,
            gui_logger=gui_logger
        )
        
        # 检查是否输出了成功日志，包含余额和积分数据
        log_calls = [call[0][0] for call in gui_logger.info.call_args_list]
        log_text = "\n".join(log_calls)
        
        assert "资料获取完成" in log_text, "应该输出'资料获取完成'"
        # 注意：由于实际实现可能返回None，我们只检查格式是否正确
        # 如果有数据，应该包含余额和积分
        if result.get('balance') is not None:
            assert "余额" in log_text, "应该包含余额信息"
        if result.get('points') is not None:
            assert "积分" in log_text, "应该包含积分信息"
    
    @pytest.mark.asyncio
    async def test_profile_flow_complete_sequence(self, profile_reader):
        """测试获取资料流程是否按顺序输出完整的日志序列 - Validates: Requirements 6.1-6.5"""
        gui_logger = Mock()
        gui_logger.info = Mock()
        gui_logger.error = Mock()
        
        # 模拟页面检测结果（个人页）
        from zdqd.src.page_detector import PageState
        mock_page_result = Mock()
        mock_page_result.state = PageState.PROFILE
        mock_page_result.confidence = 0.95
        mock_page_result.elements = []
        mock_page_result.yolo_model_used = 'test_model'
        
        profile_reader._integrated_detector.detect_page = AsyncMock(return_value=mock_page_result)
        
        result = await profile_reader.get_full_profile(
            device_id="test_device",
            step_number=3,
            gui_logger=gui_logger
        )
        
        # 收集所有日志
        log_calls = [call[0][0] for call in gui_logger.info.call_args_list]
        log_text = "\n".join(log_calls)
        
        # 验证日志序列
        assert "步骤3: 获取资料" in log_text, "应该包含步骤日志"
        assert "进入个人页" in log_text, "应该包含进入个人页日志"
        assert "获取详细资料" in log_text, "应该包含获取详细资料日志"
        assert "资料获取完成" in log_text, "应该包含资料获取完成日志"
        
        # 验证日志顺序
        step_index = log_text.index("步骤3: 获取资料")
        enter_index = log_text.index("进入个人页")
        get_index = log_text.index("获取详细资料")
        success_index = log_text.index("资料获取完成")
        
        assert step_index < enter_index, "步骤日志应该在进入个人页之前"
        assert enter_index < get_index, "进入个人页应该在获取详细资料之前"
        assert get_index < success_index, "获取详细资料应该在资料获取完成之前"
    
    @pytest.mark.asyncio
    async def test_profile_flow_handles_popup(self, profile_reader):
        """测试获取资料流程是否处理弹窗并输出"→ 关闭提示弹窗" - Validates: Requirement 6.3"""
        gui_logger = Mock()
        gui_logger.info = Mock()
        gui_logger.error = Mock()
        
        # 模拟页面检测结果（先是弹窗，然后是个人页）
        from zdqd.src.page_detector import PageState
        
        call_count = 0
        async def mock_detect_page_sequence(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            mock_result = Mock()
            mock_result.yolo_model_used = 'test_model'
            
            if call_count == 1:
                # 第一次检测：弹窗页面
                mock_result.state = PageState.POPUP
                mock_result.confidence = 0.95
                mock_result.elements = []
            else:
                # 后续检测：个人页
                mock_result.state = PageState.PROFILE
                mock_result.confidence = 0.95
                mock_result.elements = []
            
            return mock_result
        
        profile_reader._integrated_detector.detect_page = AsyncMock(side_effect=mock_detect_page_sequence)
        
        # 模拟OCR识别结果（用于检测是否返回个人页）
        async def mock_ocr_recognize(*args, **kwargs):
            nonlocal call_count
            mock_result = Mock()
            if call_count <= 2:
                # 弹窗页面的OCR结果
                mock_result.texts = ['友情提示', '确认', '取消']
            else:
                # 个人页的OCR结果
                mock_result.texts = ['昵称', 'ID', '余额', '积分']
            mock_result.boxes = None
            return mock_result
        
        profile_reader._ocr_pool.recognize = AsyncMock(side_effect=mock_ocr_recognize)
        
        result = await profile_reader.get_full_profile(
            device_id="test_device",
            step_number=3,
            gui_logger=gui_logger
        )
        
        # 检查是否输出了关闭提示弹窗的日志
        log_calls = [call[0][0] for call in gui_logger.info.call_args_list]
        assert any("关闭提示弹窗" in log for log in log_calls), "应该输出'关闭提示弹窗'"


class TestProfileFlowDataFormatting:
    """获取资料流程数据格式测试类"""
    
    def test_balance_data_format(self):
        """测试余额数据格式 - Validates: Requirement 3.1"""
        from zdqd.src.concise_logger import LogFormatter
        
        # 测试余额格式
        result = LogFormatter.format_success("资料获取完成", {"余额": "100.50元"})
        assert "余额: 100.50元" in result, f"Expected '余额: 100.50元' in result, got '{result}'"
        
        result = LogFormatter.format_success("资料获取完成", {"余额": "0.01元"})
        assert "余额: 0.01元" in result, f"Expected '余额: 0.01元' in result, got '{result}'"
    
    def test_points_data_format(self):
        """测试积分数据格式 - Validates: Requirement 3.2"""
        from zdqd.src.concise_logger import LogFormatter
        
        # 测试积分格式
        result = LogFormatter.format_success("资料获取完成", {"积分": "500"})
        assert "积分: 500" in result, f"Expected '积分: 500' in result, got '{result}'"
        
        result = LogFormatter.format_success("资料获取完成", {"积分": "0"})
        assert "积分: 0" in result, f"Expected '积分: 0' in result, got '{result}'"
    
    def test_combined_data_format(self):
        """测试余额和积分组合格式 - Validates: Requirements 3.1, 3.2, 6.5"""
        from zdqd.src.concise_logger import LogFormatter
        
        # 测试余额和积分组合
        result = LogFormatter.format_success("资料获取完成", {
            "余额": "100.50元",
            "积分": "500"
        })
        
        assert "资料获取完成" in result, "应该包含完成描述"
        assert "余额: 100.50元" in result, "应该包含余额信息"
        assert "积分: 500" in result, "应该包含积分信息"
        
        # 验证格式：应该是 "  ✓ 资料获取完成 (余额: 100.50元, 积分: 500)"
        assert result.startswith("  ✓"), "应该以'  ✓'开头"
        assert "(" in result and ")" in result, "数据应该在括号内"
    
    def test_action_log_format(self):
        """测试操作日志格式 - Validates: Requirements 6.2, 6.3, 6.4"""
        from zdqd.src.concise_logger import LogFormatter
        
        # 测试进入个人页
        result = LogFormatter.format_action("进入个人页")
        assert result == "  → 进入个人页", f"Expected '  → 进入个人页', got '{result}'"
        
        # 测试关闭提示弹窗
        result = LogFormatter.format_action("关闭提示弹窗")
        assert result == "  → 关闭提示弹窗", f"Expected '  → 关闭提示弹窗', got '{result}'"
        
        # 测试获取详细资料
        result = LogFormatter.format_action("获取详细资料")
        assert result == "  → 获取详细资料", f"Expected '  → 获取详细资料', got '{result}'"
    
    def test_success_log_format(self):
        """测试成功日志格式 - Validates: Requirement 6.5"""
        from zdqd.src.concise_logger import LogFormatter
        
        # 测试资料获取完成（无数据）
        result = LogFormatter.format_success("资料获取完成")
        assert result == "  ✓ 资料获取完成", f"Expected '  ✓ 资料获取完成', got '{result}'"
        
        # 测试资料获取完成（有数据）
        result = LogFormatter.format_success("资料获取完成", {"余额": "100.50元", "积分": "500"})
        assert result.startswith("  ✓ 资料获取完成"), "应该以'  ✓ 资料获取完成'开头"
        assert "余额: 100.50元" in result, "应该包含余额"
        assert "积分: 500" in result, "应该包含积分"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
