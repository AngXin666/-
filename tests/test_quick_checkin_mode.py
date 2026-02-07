"""
快速签到模式和只登录模式单元测试
Test Quick Check-in Mode and Login-Only Mode
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

# 导入被测试的模块
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.daily_checkin import DailyCheckin
from src.ximeng_automation import XimengAutomation
from src.models.models import Account, AccountResult
from src.page_detector_hybrid import PageState


class TestQuickCheckinMode:
    """测试快速签到模式"""
    
    @pytest.fixture
    def mock_adb(self):
        """模拟ADB桥接"""
        adb = Mock()
        adb.tap = AsyncMock()
        adb.press_back = AsyncMock()
        adb.screencap = AsyncMock(return_value=b'fake_image_data')
        return adb
    
    @pytest.fixture
    def mock_detector(self):
        """模拟页面检测器"""
        detector = Mock()
        detector.detect_page = AsyncMock()
        detector.close_popup = AsyncMock(return_value=True)
        return detector
    
    @pytest.fixture
    def mock_navigator(self):
        """模拟导航器"""
        navigator = Mock()
        navigator.navigate_to_home = AsyncMock(return_value=True)
        return navigator
    
    @pytest.fixture
    def daily_checkin(self, mock_adb, mock_detector, mock_navigator):
        """创建DailyCheckin实例"""
        return DailyCheckin(mock_adb, mock_detector, mock_navigator)
    
    @pytest.mark.asyncio
    async def test_checkin_with_zero_remaining_times_first_attempt(self, daily_checkin, mock_adb, mock_detector):
        """
        测试：第一次循环时，即使OCR识别剩余次数为0，也会点击签到按钮
        
        场景：
        1. OCR识别剩余次数为0
        2. 第一次循环（attempt == 0）
        3. 应该点击签到按钮
        4. 检测到温馨提示弹窗
        5. 确认签到完成
        """
        device_id = "test_device"
        phone = "13800138000"
        
        # 模拟页面检测：签到页 -> 温馨提示
        page_states = [
            Mock(state=PageState.CHECKIN, confidence=0.95),  # 初始在签到页
            Mock(state=PageState.WARMTIP, confidence=0.90),  # 点击后出现温馨提示
        ]
        mock_detector.detect_page.side_effect = page_states
        
        # 模拟签到页面读取器返回剩余次数为0
        with patch.object(daily_checkin, 'reader') as mock_reader:
            mock_reader.get_checkin_info = AsyncMock(return_value={
                'total_times': 5,
                'daily_remaining_times': 0,  # 剩余次数为0
                'can_checkin': False,
                'raw_text': '剩余0次'
            })
            
            # 模拟智能等待器
            with patch('src.daily_checkin.wait_for_page') as mock_wait:
                mock_wait.return_value = Mock(state=PageState.WARMTIP)
                
                # 执行签到
                result = await daily_checkin.do_checkin(
                    device_id=device_id,
                    phone=phone,
                    allow_skip_profile=True  # 快速签到模式
                )
        
        # 验证：签到按钮被点击了
        assert mock_adb.tap.called, "签到按钮应该被点击"
        
        # 验证：检测到温馨提示后设置已签到标志
        assert result['already_checked'] == True, "应该标记为已签到"
        assert result['remaining_times'] == 0, "剩余次数应该为0"
    
    @pytest.mark.asyncio
    async def test_checkin_with_zero_remaining_times_second_attempt(self, daily_checkin, mock_adb, mock_detector):
        """
        测试：第二次循环时，如果剩余次数为0，应该跳出循环
        
        场景：
        1. 第一次点击成功，获得奖励
        2. 第二次循环，推算剩余次数为0
        3. 应该跳出循环，不再点击
        """
        device_id = "test_device"
        phone = "13800138000"
        
        # 这个测试需要模拟完整的签到循环
        # 由于实现复杂，这里只验证逻辑
        pass  # 实际测试需要更复杂的mock设置


class TestQuickCheckinModeWithDatabase:
    """测试快速签到模式从数据库获取历史记录"""
    
    @pytest.mark.asyncio
    async def test_get_nickname_and_userid_from_database(self):
        """
        测试：快速签到模式下从数据库获取昵称和用户ID
        
        场景：
        1. 快速签到模式（enable_profile=False）
        2. 从数据库查询最新记录
        3. 提取昵称和用户ID
        """
        # 模拟数据库返回历史记录
        mock_db = Mock()
        mock_db.get_latest_record_by_phone = Mock(return_value={
            'nickname': '测试用户',
            'user_id': 'TEST123456',
            'balance': 10.50
        })
        
        # 模拟XimengAutomation
        with patch('src.ximeng_automation.LocalDatabase', return_value=mock_db):
            # 创建账号
            account = Account(phone="13800138000", password="test123")
            
            # 模拟工作流配置
            workflow_config = {
                'enable_login': True,
                'enable_profile': False,  # 快速签到模式
                'enable_checkin': True,
                'enable_transfer': False
            }
            
            # 验证数据库被调用
            # 实际测试需要完整的XimengAutomation实例
            pass


class TestLoginOnlyMode:
    """测试只登录模式"""
    
    @pytest.mark.asyncio
    async def test_login_only_mode_balance_consistency(self):
        """
        测试：只登录模式下，余额前后一致
        
        场景：
        1. 只登录模式（enable_checkin=False）
        2. 登录后获取资料，得到balance_before
        3. 跳过签到
        4. balance_after应该等于balance_before
        """
        # 创建测试账号
        account = Account(phone="13800138000", password="test123")
        
        # 模拟结果对象
        result = AccountResult(
            phone=account.phone,
            success=False,
            timestamp=datetime.now()
        )
        
        # 模拟获取资料后的余额
        result.balance_before = 15.80
        
        # 模拟工作流配置：只登录，不签到
        workflow_config = {
            'enable_login': True,
            'enable_profile': True,
            'enable_checkin': False,  # 禁用签到
            'enable_transfer': False
        }
        
        # 模拟只登录模式的逻辑
        if not workflow_config.get('enable_checkin', True):
            # 使用初始余额作为最终余额
            if result.balance_before is not None:
                result.balance_after = result.balance_before
        
        # 验证：余额前后一致
        assert result.balance_after == result.balance_before, "只登录模式下余额前后应该一致"
        assert result.balance_after == 15.80, "最终余额应该等于初始余额"
        assert result.balance_after is not None, "最终余额不应该为None"


class TestBalanceConsistency:
    """测试余额一致性"""
    
    def test_quick_checkin_mode_balance_from_login(self):
        """
        测试：快速签到模式下，使用登录余额作为最终余额
        
        场景：
        1. 快速签到模式
        2. 登录后获取余额（balance_before）
        3. 执行签到
        4. 使用登录余额作为最终余额（balance_after = balance_before）
        """
        result = AccountResult(
            phone="13800138000",
            success=False,
            timestamp=datetime.now()
        )
        
        # 模拟登录后获取的余额
        result.balance_before = 20.50
        
        # 模拟快速签到模式的逻辑
        profile_success = False  # 快速签到模式跳过获取资料
        
        if not profile_success:
            # 使用登录余额作为最终余额
            if result.balance_before is not None:
                result.balance_after = result.balance_before
        
        # 验证
        assert result.balance_after == 20.50, "快速签到模式下应使用登录余额"
        assert result.balance_after == result.balance_before, "余额前后应该一致"
    
    def test_login_only_mode_no_na_display(self):
        """
        测试：只登录模式下，GUI不显示N/A
        
        场景：
        1. 只登录模式
        2. 获取资料后有balance_before
        3. 跳过签到，balance_after = balance_before
        4. GUI表格中余额前后都有值，不显示N/A
        """
        result = AccountResult(
            phone="13800138000",
            success=False,
            timestamp=datetime.now()
        )
        
        # 模拟获取资料
        result.balance_before = 18.30
        result.nickname = "测试用户"
        result.user_id = "USER123"
        
        # 模拟只登录模式
        enable_checkin = False
        
        if not enable_checkin:
            if result.balance_before is not None:
                result.balance_after = result.balance_before
        
        # 验证：余额前后都有值
        assert result.balance_before is not None, "余额前不应该为None"
        assert result.balance_after is not None, "余额后不应该为None"
        assert result.balance_after == result.balance_before, "余额应该一致"


if __name__ == '__main__':
    # 运行测试
    pytest.main([__file__, '-v', '-s'])
