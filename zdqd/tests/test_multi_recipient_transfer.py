"""
多收款人转账集成测试
Multi-Recipient Transfer Integration Tests
"""

import pytest
import asyncio
from pathlib import Path
from src.user_manager import UserManager, User
from src.transfer_config import TransferConfig
from src.recipient_selector import RecipientSelector


class TestMultiRecipientTransfer:
    """多收款人转账集成测试类"""
    
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self, manager):
        """每个测试前后的设置和清理"""
        # 测试前清理
        test_user_ids = ["test_user_001", "test_user_002", "test_user_003"]
        for user_id in test_user_ids:
            if manager.get_user(user_id):
                manager.delete_user(user_id)
        
        yield
        
        # 测试后清理
        for user_id in test_user_ids:
            if manager.get_user(user_id):
                manager.delete_user(user_id)
    
    @pytest.fixture
    def manager(self):
        """创建用户管理器实例"""
        return UserManager()
    
    @pytest.fixture
    def transfer_config(self):
        """创建转账配置实例"""
        config = TransferConfig()
        # 启用用户管理收款人配置
        config.use_user_manager_recipients = True
        # 清空转账配置的收款人列表（避免干扰测试）
        original_recipients = config.level_recipients.copy()
        config.level_recipients = {1: [], 2: [], 3: []}
        config.recipient_ids = []
        
        yield config
        
        # 恢复原始配置
        config.level_recipients = original_recipients
        config.recipient_ids = original_recipients[1]
        config.save()
    
    @pytest.fixture
    def selector(self):
        """创建收款人选择器实例"""
        selector = RecipientSelector(strategy="rotation")
        selector.reset_rotation()
        return selector
    
    @pytest.fixture
    def test_user(self):
        """创建测试用户"""
        return User(
            user_id="test_user_001",
            user_name="测试用户",
            transfer_recipients=["13800138000", "13900139000", "14000140000"],
            enabled=True
        )
    
    def test_complete_flow(self, manager, transfer_config, selector, test_user):
        """测试完整转账流程（用户管理 → 选择器 → 转账）"""
        # 1. 添加用户到用户管理
        assert manager.add_user(test_user)
        
        # 2. 分配账号
        test_phone = "18888888888"
        assert manager.assign_account(test_phone, test_user.user_id)
        
        # 3. 从用户管理获取收款人
        recipient1 = transfer_config.get_transfer_recipient_from_user_manager(
            test_phone,
            selector
        )
        assert recipient1 == "13800138000"
        
        # 4. 再次获取（应该轮询到下一个）
        recipient2 = transfer_config.get_transfer_recipient_from_user_manager(
            test_phone,
            selector
        )
        assert recipient2 == "13900139000"
        
        # 5. 第三次获取
        recipient3 = transfer_config.get_transfer_recipient_from_user_manager(
            test_phone,
            selector
        )
        assert recipient3 == "14000140000"
        
        # 6. 第四次获取（应该循环回第一个）
        recipient4 = transfer_config.get_transfer_recipient_from_user_manager(
            test_phone,
            selector
        )
        assert recipient4 == "13800138000"
        
        # 清理
        manager.delete_user(test_user.user_id)
    
    def test_fallback_mechanism(self, manager, transfer_config, selector):
        """测试降级机制（用户管理不可用时）"""
        # 1. 配置转账配置的收款人
        transfer_config.add_recipient("fallback_recipient", level=1)
        
        # 2. 尝试从用户管理获取（应该失败，因为没有配置）
        recipient = transfer_config.get_transfer_recipient_enhanced(
            phone="19999999999",  # 未分配的手机号
            user_id="unknown_user",
            current_level=0,
            selector=selector
        )
        
        # 3. 应该降级到转账配置
        assert recipient == "fallback_recipient"
        
        # 清理
        transfer_config.remove_recipient("fallback_recipient", level=1)
    
    def test_multi_level_compatibility(self, manager, transfer_config, selector, test_user):
        """测试多级转账兼容性"""
        # 1. 添加用户
        assert manager.add_user(test_user)
        test_phone = "18888888888"
        assert manager.assign_account(test_phone, test_user.user_id)
        
        # 2. 配置多级转账
        transfer_config.multi_level_enabled = True
        transfer_config.max_transfer_level = 2
        transfer_config.add_recipient("level2_recipient", level=2)
        
        # 3. 初始账号转账（level=0）
        recipient1 = transfer_config.get_transfer_recipient_enhanced(
            phone=test_phone,
            user_id=test_user.user_id,
            current_level=0,
            selector=selector
        )
        # 应该从用户管理获取
        assert recipient1 in test_user.transfer_recipients
        
        # 4. 1级收款账号转账（level=1）
        recipient2 = transfer_config.get_transfer_recipient_enhanced(
            phone=recipient1,
            user_id=recipient1,
            current_level=1,
            selector=selector
        )
        # 应该转给2级收款账号
        assert recipient2 == "level2_recipient"
        
        # 清理
        manager.delete_user(test_user.user_id)
        transfer_config.remove_recipient("level2_recipient", level=2)
        transfer_config.multi_level_enabled = False
    
    def test_rotation_persistence(self, manager, transfer_config, test_user):
        """测试轮询状态持久化（重启后继续）"""
        # 1. 添加用户
        assert manager.add_user(test_user)
        test_phone = "18888888888"
        assert manager.assign_account(test_phone, test_user.user_id)
        
        # 2. 第一个选择器选择两次（先重置状态）
        selector1 = RecipientSelector(strategy="rotation")
        selector1.reset_rotation(test_user.user_id)  # 重置状态
        
        recipient1 = transfer_config.get_transfer_recipient_from_user_manager(
            test_phone,
            selector1
        )
        assert recipient1 == "13800138000"
        
        recipient2 = transfer_config.get_transfer_recipient_from_user_manager(
            test_phone,
            selector1
        )
        assert recipient2 == "13900139000"
        
        # 3. 创建新的选择器（模拟重启）
        selector2 = RecipientSelector(strategy="rotation")
        recipient3 = transfer_config.get_transfer_recipient_from_user_manager(
            test_phone,
            selector2
        )
        # 应该继续上次的索引
        assert recipient3 == "14000140000"
        
        # 清理
        selector2.reset_rotation(test_user.user_id)
    
    def test_disabled_user_manager(self, manager, transfer_config, selector, test_user):
        """测试禁用用户管理收款人配置"""
        # 1. 添加用户
        assert manager.add_user(test_user)
        test_phone = "18888888888"
        assert manager.assign_account(test_phone, test_user.user_id)
        
        # 2. 配置转账配置的收款人
        transfer_config.add_recipient("config_recipient", level=1)
        
        # 3. 禁用用户管理收款人配置
        transfer_config.use_user_manager_recipients = False
        
        # 4. 获取收款人
        recipient = transfer_config.get_transfer_recipient_enhanced(
            phone=test_phone,
            user_id=test_user.user_id,
            current_level=0,
            selector=selector
        )
        
        # 5. 应该使用转账配置的收款人
        assert recipient == "config_recipient"
        
        # 清理
        manager.delete_user(test_user.user_id)
        transfer_config.remove_recipient("config_recipient", level=1)
        transfer_config.use_user_manager_recipients = True
    
    def test_self_filtering(self, manager, transfer_config, selector):
        """测试自我过滤（避免转给自己）"""
        # 1. 创建用户，收款人包括自己
        user = User(
            user_id="test_user_002",
            user_name="测试用户2",
            transfer_recipients=["18888888888", "13900139000", "14000140000"],
            enabled=True
        )
        assert manager.add_user(user)
        
        test_phone = "18888888888"
        assert manager.assign_account(test_phone, user.user_id)
        
        # 2. 获取收款人
        recipient = transfer_config.get_transfer_recipient_from_user_manager(
            test_phone,
            selector
        )
        
        # 3. 应该跳过自己，选择下一个
        assert recipient == "13900139000"
        
        # 清理
        manager.delete_user(user.user_id)
    
    def test_empty_recipients(self, manager, transfer_config, selector):
        """测试空收款人列表"""
        # 1. 创建没有收款人的用户
        user = User(
            user_id="test_user_003",
            user_name="测试用户3",
            transfer_recipients=[],
            enabled=True
        )
        assert manager.add_user(user)
        
        test_phone = "18888888888"
        assert manager.assign_account(test_phone, user.user_id)
        
        # 2. 获取收款人
        recipient = transfer_config.get_transfer_recipient_from_user_manager(
            test_phone,
            selector
        )
        
        # 3. 应该返回None
        assert recipient is None
        
        # 清理
        manager.delete_user(user.user_id)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
