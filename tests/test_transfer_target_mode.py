"""
转账目标模式单元测试
Unit Tests for Transfer Target Mode
"""

import unittest
import sys
import os
import json
import tempfile
import shutil
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from src.transfer_config import TransferConfig
from src.recipient_selector import RecipientSelector
from src.user_manager import UserManager, User


class TestTransferTargetMode(unittest.TestCase):
    """转账目标模式测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建临时目录
        self.test_dir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.test_dir)
        
        # 创建测试用的配置文件
        self.config_file = Path("transfer_config.json")
        self.user_file = Path("users.json")
        self.accounts_file = Path("user_accounts.json")
        
        # 初始化配置
        self.config = TransferConfig()
        self.config.enabled = True
        self.config.min_transfer_amount = 30.0
        self.config.min_balance = 0.0
        self.config.recipient_ids = ["15000150000", "16000160000"]
        self.config.level_recipients[1] = ["15000150000", "16000160000"]
        self.config.save()
        
        # 初始化用户管理器
        self.user_manager = UserManager()
        
        # 创建测试用户
        self.test_user = User(
            user_id="user_001",
            user_name="测试管理员",
            transfer_recipients=["13900139000", "14000140000"],
            description="测试用户",
            enabled=True
        )
        self.user_manager.add_user(self.test_user)
        
        # 为测试用户分配账号
        test_accounts = ["13800138000", "13800138001", "13800138002"]
        for phone in test_accounts:
            self.user_manager.assign_account(phone, self.test_user.user_id)
        
        # 创建选择器
        self.selector = RecipientSelector(strategy="rotation")
    
    def tearDown(self):
        """测试后清理"""
        os.chdir(self.original_dir)
        shutil.rmtree(self.test_dir)
    
    def test_config_load_and_save(self):
        """测试配置加载和保存"""
        # 设置模式
        self.config.set_transfer_target_mode("manager_account")
        self.assertEqual(self.config.transfer_target_mode, "manager_account")
        
        # 重新加载配置
        new_config = TransferConfig()
        self.assertEqual(new_config.transfer_target_mode, "manager_account")
    
    def test_invalid_mode(self):
        """测试无效模式"""
        with self.assertRaises(ValueError):
            self.config.set_transfer_target_mode("invalid_mode")
    
    def test_mode_display_names(self):
        """测试模式显示名称"""
        self.config.set_transfer_target_mode("manager_account")
        self.assertEqual(self.config.get_transfer_target_mode_display(), "转给管理员自己")
        
        self.config.set_transfer_target_mode("manager_recipients")
        self.assertEqual(self.config.get_transfer_target_mode_display(), "转给管理员的收款人")
        
        self.config.set_transfer_target_mode("system_recipients")
        self.assertEqual(self.config.get_transfer_target_mode_display(), "转给系统配置收款人")
    
    def test_manager_account_mode(self):
        """测试模式1：转给管理员自己"""
        self.config.set_transfer_target_mode("manager_account")
        
        # 测试账号1转账
        recipient = self.config.get_transfer_recipient_enhanced(
            phone="13800138000",
            user_id="test_user_id",
            current_level=0,
            selector=self.selector
        )
        
        # 应该选择管理员的其他账号
        self.assertIsNotNone(recipient)
        self.assertIn(recipient, ["13800138001", "13800138002"])
        self.assertNotEqual(recipient, "13800138000")  # 不应该是自己
    
    def test_manager_recipients_mode(self):
        """测试模式2：转给管理员的收款人"""
        self.config.set_transfer_target_mode("manager_recipients")
        
        # 测试账号1转账
        recipient = self.config.get_transfer_recipient_enhanced(
            phone="13800138000",
            user_id="test_user_id",
            current_level=0,
            selector=self.selector
        )
        
        # 应该选择管理员配置的收款人
        self.assertIsNotNone(recipient)
        self.assertIn(recipient, ["13900139000", "14000140000"])
    
    def test_system_recipients_mode(self):
        """测试模式3：转给系统配置收款人"""
        self.config.set_transfer_target_mode("system_recipients")
        
        # 测试账号1转账
        recipient = self.config.get_transfer_recipient_enhanced(
            phone="13800138000",
            user_id="test_user_id",
            current_level=0,
            selector=self.selector
        )
        
        # 应该选择系统配置的收款人
        self.assertIsNotNone(recipient)
        self.assertIn(recipient, ["15000150000", "16000160000"])
    
    def test_fallback_to_system_recipients(self):
        """测试降级到系统配置收款人"""
        # 创建一个没有收款人的用户
        user_no_recipients = User(
            user_id="user_002",
            user_name="无收款人管理员",
            transfer_recipients=[],  # 没有收款人
            description="测试用户",
            enabled=True
        )
        self.user_manager.add_user(user_no_recipients)
        self.user_manager.assign_account("13800138003", user_no_recipients.user_id)
        
        # 设置为管理员收款人模式
        self.config.set_transfer_target_mode("manager_recipients")
        
        # 测试转账
        recipient = self.config.get_transfer_recipient_enhanced(
            phone="13800138003",
            user_id="test_user_id",
            current_level=0,
            selector=self.selector
        )
        
        # 应该降级到系统配置收款人
        self.assertIsNotNone(recipient)
        self.assertIn(recipient, ["15000150000", "16000160000"])
    
    def test_manager_account_mode_single_account(self):
        """测试管理员只有一个账号的情况"""
        # 创建只有一个账号的管理员
        user_single = User(
            user_id="user_003",
            user_name="单账号管理员",
            transfer_recipients=["13900139000"],
            description="测试用户",
            enabled=True
        )
        self.user_manager.add_user(user_single)
        self.user_manager.assign_account("13800138004", user_single.user_id)
        
        # 设置为管理员账号模式
        self.config.set_transfer_target_mode("manager_account")
        
        # 测试转账
        recipient = self.config.get_transfer_recipient_enhanced(
            phone="13800138004",
            user_id="test_user_id",
            current_level=0,
            selector=self.selector
        )
        
        # 应该降级到系统配置收款人（因为管理员没有其他账号）
        self.assertIsNotNone(recipient)
        self.assertIn(recipient, ["15000150000", "16000160000"])
    
    def test_rotation_selection(self):
        """测试轮询选择"""
        self.config.set_transfer_target_mode("manager_recipients")
        
        # 重置轮询状态
        self.selector.reset_rotation()
        
        # 多次选择，应该轮询
        recipients = []
        for _ in range(4):
            recipient = self.config.get_transfer_recipient_enhanced(
                phone="13800138000",
                user_id="test_user_id",
                current_level=0,
                selector=self.selector
            )
            recipients.append(recipient)
        
        # 应该包含两个收款人，且轮询
        self.assertIn("13900139000", recipients)
        self.assertIn("14000140000", recipients)
        # 前两个应该不同
        self.assertNotEqual(recipients[0], recipients[1])
        # 第3个应该等于第1个（轮询）
        self.assertEqual(recipients[0], recipients[2])
    
    def test_disabled_user_manager_recipients(self):
        """测试禁用用户管理收款人配置"""
        self.config.use_user_manager_recipients = False
        self.config.save()
        
        # 即使设置为管理员收款人模式，也应该使用系统配置
        self.config.set_transfer_target_mode("manager_recipients")
        
        recipient = self.config.get_transfer_recipient_enhanced(
            phone="13800138000",
            user_id="test_user_id",
            current_level=0,
            selector=self.selector
        )
        
        # 应该使用系统配置收款人
        self.assertIsNotNone(recipient)
        self.assertIn(recipient, ["15000150000", "16000160000"])
    
    def test_no_manager_assigned(self):
        """测试账号未分配管理员"""
        self.config.set_transfer_target_mode("manager_recipients")
        
        # 测试未分配管理员的账号
        recipient = self.config.get_transfer_recipient_enhanced(
            phone="19999999999",  # 未分配管理员的账号
            user_id="test_user_id",
            current_level=0,
            selector=self.selector
        )
        
        # 应该降级到系统配置收款人
        self.assertIsNotNone(recipient)
        self.assertIn(recipient, ["15000150000", "16000160000"])
    
    def test_multi_level_transfer(self):
        """测试多级转账不受模式影响"""
        self.config.set_transfer_target_mode("manager_recipients")
        self.config.multi_level_enabled = True
        self.config.max_transfer_level = 2
        self.config.level_recipients[2] = ["17000170000"]
        self.config.save()
        
        # 测试1级收款账号转账（current_level=1）
        recipient = self.config.get_transfer_recipient_enhanced(
            phone="15000150000",
            user_id="15000150000",
            current_level=1,  # 1级收款账号
            selector=self.selector
        )
        
        # 应该使用原有逻辑，转给2级收款账号
        self.assertEqual(recipient, "17000170000")
    
    def test_self_filtering(self):
        """测试自动过滤自己"""
        self.config.set_transfer_target_mode("manager_account")
        
        # 管理员只有两个账号
        user_two = User(
            user_id="user_004",
            user_name="双账号管理员",
            transfer_recipients=[],
            description="测试用户",
            enabled=True
        )
        self.user_manager.add_user(user_two)
        self.user_manager.assign_account("13800138005", user_two.user_id)
        self.user_manager.assign_account("13800138006", user_two.user_id)
        
        # 账号1转账
        recipient1 = self.config.get_transfer_recipient_enhanced(
            phone="13800138005",
            user_id="test_user_id",
            current_level=0,
            selector=self.selector
        )
        
        # 应该选择账号2
        self.assertEqual(recipient1, "13800138006")
        
        # 账号2转账
        recipient2 = self.config.get_transfer_recipient_enhanced(
            phone="13800138006",
            user_id="test_user_id",
            current_level=0,
            selector=self.selector
        )
        
        # 应该选择账号1
        self.assertEqual(recipient2, "13800138005")
    
    def test_config_persistence(self):
        """测试配置持久化"""
        # 设置各种配置
        self.config.set_transfer_target_mode("manager_account")
        self.config.set_enabled(True)
        self.config.set_min_balance(5.0)
        
        # 创建新实例加载配置
        new_config = TransferConfig()
        
        # 验证配置已保存
        self.assertEqual(new_config.transfer_target_mode, "manager_account")
        self.assertTrue(new_config.enabled)
        self.assertEqual(new_config.min_balance, 5.0)
    
    def test_default_mode(self):
        """测试默认模式"""
        # 创建新配置（不设置模式）
        config = TransferConfig()
        
        # 默认应该是 manager_recipients
        self.assertEqual(config.transfer_target_mode, "manager_recipients")


class TestRecipientSelector(unittest.TestCase):
    """收款人选择器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.test_dir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.test_dir)
        
        self.selector = RecipientSelector(strategy="rotation")
    
    def tearDown(self):
        """测试后清理"""
        os.chdir(self.original_dir)
        shutil.rmtree(self.test_dir)
    
    def test_rotation_strategy(self):
        """测试轮询策略"""
        recipients = ["13900139000", "14000140000", "15000150000"]
        
        # 重置轮询状态
        self.selector.reset_rotation("test_key")
        
        # 选择3次
        selected = []
        for _ in range(3):
            recipient = self.selector.select_recipient(
                recipients,
                sender_phone="13800138000",
                key="test_key"
            )
            selected.append(recipient)
        
        # 应该按顺序选择
        self.assertEqual(selected, recipients)
        
        # 第4次应该回到第1个
        recipient = self.selector.select_recipient(
            recipients,
            sender_phone="13800138000",
            key="test_key"
        )
        self.assertEqual(recipient, recipients[0])
    
    def test_random_strategy(self):
        """测试随机策略"""
        selector = RecipientSelector(strategy="random")
        recipients = ["13900139000", "14000140000", "15000150000"]
        
        # 选择多次
        selected = []
        for _ in range(10):
            recipient = selector.select_recipient(
                recipients,
                sender_phone="13800138000"
            )
            selected.append(recipient)
        
        # 应该都在收款人列表中
        for s in selected:
            self.assertIn(s, recipients)
    
    def test_self_filtering(self):
        """测试自动过滤发送人"""
        recipients = ["13900139000", "14000140000", "13800138000"]
        
        # 发送人是13800138000
        recipient = self.selector.select_recipient(
            recipients,
            sender_phone="13800138000",
            key="test_key"
        )
        
        # 不应该选择发送人自己
        self.assertNotEqual(recipient, "13800138000")
        self.assertIn(recipient, ["13900139000", "14000140000"])
    
    def test_all_recipients_are_sender(self):
        """测试所有收款人都是发送人"""
        recipients = ["13800138000"]
        
        # 发送人是13800138000
        recipient = self.selector.select_recipient(
            recipients,
            sender_phone="13800138000",
            key="test_key"
        )
        
        # 应该返回None
        self.assertIsNone(recipient)
    
    def test_empty_recipients(self):
        """测试空收款人列表"""
        with self.assertRaises(ValueError):
            self.selector.select_recipient(
                [],
                sender_phone="13800138000",
                key="test_key"
            )


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestTransferTargetMode))
    suite.addTests(loader.loadTestsFromTestCase(TestRecipientSelector))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 返回测试结果
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
