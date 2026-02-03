"""
转账目标模式完整单元测试（无数据库依赖）
Complete Unit Tests for Transfer Target Mode (No Database Dependencies)
"""

import unittest
import sys
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from src.transfer_config import TransferConfig


class TestTransferConfigBasic(unittest.TestCase):
    """转账配置基础功能测试（不涉及数据库）"""
    
    def setUp(self):
        """测试前准备"""
        # 创建临时配置文件
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        self.temp_file.close()
        
        # 创建配置实例
        self.config = TransferConfig()
        self.config.config_file = Path(self.temp_file.name)
        self.config.enabled = True
        self.config.min_transfer_amount = 30.0
        self.config.min_balance = 0.0
        self.config.recipient_ids = ["15000150000", "16000160000"]
        self.config.level_recipients[1] = ["15000150000", "16000160000"]
        self.config.save()
    
    def tearDown(self):
        """测试后清理"""
        try:
            os.unlink(self.temp_file.name)
        except:
            pass
    
    def test_default_mode(self):
        """测试默认模式"""
        config = TransferConfig()
        config.config_file = Path(self.temp_file.name)
        config.load()
        
        # 默认应该是 manager_recipients
        self.assertEqual(config.transfer_target_mode, "manager_recipients")
    
    def test_set_mode_manager_account(self):
        """测试设置模式：转给管理员自己"""
        self.config.set_transfer_target_mode("manager_account")
        self.assertEqual(self.config.transfer_target_mode, "manager_account")
        
        # 验证保存
        new_config = TransferConfig()
        new_config.config_file = Path(self.temp_file.name)
        new_config.load()
        self.assertEqual(new_config.transfer_target_mode, "manager_account")
    
    def test_set_mode_manager_recipients(self):
        """测试设置模式：转给管理员的收款人"""
        self.config.set_transfer_target_mode("manager_recipients")
        self.assertEqual(self.config.transfer_target_mode, "manager_recipients")
    
    def test_set_mode_system_recipients(self):
        """测试设置模式：转给系统配置收款人"""
        self.config.set_transfer_target_mode("system_recipients")
        self.assertEqual(self.config.transfer_target_mode, "system_recipients")
    
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
    
    def test_config_persistence(self):
        """测试配置持久化"""
        # 设置各种配置
        self.config.set_transfer_target_mode("manager_account")
        self.config.set_enabled(True)
        self.config.set_min_balance(5.0)
        
        # 创建新实例加载配置
        new_config = TransferConfig()
        new_config.config_file = Path(self.temp_file.name)
        new_config.load()
        
        # 验证配置已保存
        self.assertEqual(new_config.transfer_target_mode, "manager_account")
        self.assertTrue(new_config.enabled)
        self.assertEqual(new_config.min_balance, 5.0)
    
    def test_config_json_structure(self):
        """测试配置JSON结构"""
        self.config.set_transfer_target_mode("system_recipients")
        self.config.save()
        
        # 读取JSON文件
        with open(self.temp_file.name, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 验证字段存在
        self.assertIn('transfer_target_mode', data)
        self.assertEqual(data['transfer_target_mode'], 'system_recipients')
        self.assertIn('enabled', data)
        self.assertIn('min_balance', data)
        self.assertIn('min_transfer_amount', data)
        self.assertIn('recipient_ids', data)
    
    def test_backward_compatibility(self):
        """测试向后兼容性"""
        # 创建旧版本配置（没有transfer_target_mode字段）
        old_config = {
            'min_balance': 0.0,
            'min_transfer_amount': 30.0,
            'recipient_ids': ['15000150000'],
            'enabled': True
        }
        
        with open(self.temp_file.name, 'w', encoding='utf-8') as f:
            json.dump(old_config, f)
        
        # 加载配置
        config = TransferConfig()
        config.config_file = Path(self.temp_file.name)
        config.load()
        
        # 应该使用默认模式
        self.assertEqual(config.transfer_target_mode, "manager_recipients")
    
    def test_mode_validation_on_load(self):
        """测试加载时的模式验证"""
        # 创建包含无效模式的配置
        invalid_config = {
            'min_balance': 0.0,
            'min_transfer_amount': 30.0,
            'recipient_ids': ['15000150000'],
            'enabled': True,
            'transfer_target_mode': 'invalid_mode'
        }
        
        with open(self.temp_file.name, 'w', encoding='utf-8') as f:
            json.dump(invalid_config, f)
        
        # 加载配置
        config = TransferConfig()
        config.config_file = Path(self.temp_file.name)
        config.load()
        
        # 应该降级到默认模式
        self.assertEqual(config.transfer_target_mode, "manager_recipients")


class TestTransferModeLogic(unittest.TestCase):
    """转账模式逻辑测试（使用Mock）"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        self.temp_file.close()
        
        self.config = TransferConfig()
        self.config.config_file = Path(self.temp_file.name)
        self.config.enabled = True
        self.config.min_transfer_amount = 30.0
        self.config.min_balance = 0.0
        self.config.recipient_ids = ["15000150000", "16000160000"]
        self.config.level_recipients[1] = ["15000150000", "16000160000"]
        self.config.use_user_manager_recipients = True
        self.config.save()
    
    def tearDown(self):
        """测试后清理"""
        try:
            os.unlink(self.temp_file.name)
        except:
            pass
    
    @patch('src.transfer_config.TransferConfig.get_transfer_recipient')
    def test_system_recipients_mode(self, mock_get_recipient):
        """测试模式3：转给系统配置收款人"""
        mock_get_recipient.return_value = "15000150000"
        
        self.config.set_transfer_target_mode("system_recipients")
        
        # 调用增强方法
        recipient = self.config.get_transfer_recipient_enhanced(
            phone="13800138000",
            user_id="test_user_id",
            current_level=0,
            selector=None
        )
        
        # 应该调用原有方法
        mock_get_recipient.assert_called_once_with("test_user_id", 0)
        self.assertEqual(recipient, "15000150000")
    
    def test_disabled_user_manager_recipients(self):
        """测试禁用用户管理收款人配置"""
        self.config.use_user_manager_recipients = False
        self.config.save()
        
        # 即使设置为管理员收款人模式，也应该使用系统配置
        self.config.set_transfer_target_mode("manager_recipients")
        
        with patch.object(self.config, 'get_transfer_recipient', return_value="15000150000") as mock_method:
            recipient = self.config.get_transfer_recipient_enhanced(
                phone="13800138000",
                user_id="test_user_id",
                current_level=0,
                selector=None
            )
            
            # 应该调用原有方法
            mock_method.assert_called_once()
            self.assertEqual(recipient, "15000150000")
    
    def test_multi_level_transfer_not_affected(self):
        """测试多级转账不受模式影响"""
        self.config.set_transfer_target_mode("manager_recipients")
        self.config.multi_level_enabled = True
        self.config.max_transfer_level = 2
        self.config.level_recipients[2] = ["17000170000"]
        self.config.save()
        
        # 测试1级收款账号转账（current_level=1）
        with patch.object(self.config, 'get_transfer_recipient', return_value="17000170000") as mock_method:
            recipient = self.config.get_transfer_recipient_enhanced(
                phone="15000150000",
                user_id="15000150000",
                current_level=1,  # 1级收款账号
                selector=None
            )
            
            # 应该使用原有逻辑
            mock_method.assert_called_once_with("15000150000", 1)
            self.assertEqual(recipient, "17000170000")
    
    def test_manager_account_mode_with_multiple_accounts(self):
        """测试模式1：管理员有多个账号"""
        # 创建Mock用户管理器
        mock_manager = MagicMock()
        mock_user_manager_class.return_value = mock_manager
        
        # 创建Mock用户
        mock_user = MagicMock()
        mock_user.user_id = "user_001"
        mock_user.user_name = "测试管理员"
        mock_user.enabled = True
        
        mock_manager.get_account_user.return_value = mock_user
        mock_manager.get_user_accounts.return_value = ["13800138000", "13800138001", "13800138002"]
        
        self.config.set_transfer_target_mode("manager_account")
        
        # 测试账号1转账
        recipient = self.config.get_transfer_recipient_enhanced(
            phone="13800138000",
            user_id="test_user_id",
            current_level=0,
            selector=None
        )
        
        # 应该选择管理员的其他账号
        self.assertIsNotNone(recipient)
        self.assertIn(recipient, ["13800138001", "13800138002"])
        self.assertNotEqual(recipient, "13800138000")  # 不应该是自己
    
    @patch('src.transfer_config.UserManager')
    def test_manager_account_mode_single_account_fallback(self, mock_user_manager_class):
        """测试模式1：管理员只有一个账号（降级）"""
        # 创建Mock用户管理器
        mock_manager = MagicMock()
        mock_user_manager_class.return_value = mock_manager
        
        # 创建Mock用户
        mock_user = MagicMock()
        mock_user.user_id = "user_001"
        mock_user.user_name = "单账号管理员"
        mock_user.enabled = True
        
        mock_manager.get_account_user.return_value = mock_user
        mock_manager.get_user_accounts.return_value = ["13800138000"]  # 只有一个账号
        
        self.config.set_transfer_target_mode("manager_account")
        
        with patch.object(self.config, 'get_transfer_recipient', return_value="15000150000") as mock_method:
            # 测试转账
            recipient = self.config.get_transfer_recipient_enhanced(
                phone="13800138000",
                user_id="test_user_id",
                current_level=0,
                selector=None
            )
            
            # 应该降级到系统配置收款人
            mock_method.assert_called_once()
            self.assertEqual(recipient, "15000150000")
    
    @patch('src.transfer_config.UserManager')
    def test_manager_recipients_mode_with_recipients(self, mock_user_manager_class):
        """测试模式2：管理员配置了收款人"""
        # 创建Mock用户管理器
        mock_manager = MagicMock()
        mock_user_manager_class.return_value = mock_manager
        
        # 创建Mock用户
        mock_user = MagicMock()
        mock_user.user_id = "user_001"
        mock_user.user_name = "测试管理员"
        mock_user.enabled = True
        mock_user.transfer_recipients = ["13900139000", "14000140000"]
        
        mock_manager.get_account_user.return_value = mock_user
        
        # 创建Mock选择器
        mock_selector = MagicMock()
        mock_selector.select_recipient.return_value = "13900139000"
        
        self.config.set_transfer_target_mode("manager_recipients")
        
        # 测试转账
        recipient = self.config.get_transfer_recipient_enhanced(
            phone="13800138000",
            user_id="test_user_id",
            current_level=0,
            selector=mock_selector
        )
        
        # 应该选择管理员配置的收款人
        self.assertEqual(recipient, "13900139000")
        mock_selector.select_recipient.assert_called_once()
    
    @patch('src.transfer_config.UserManager')
    def test_manager_recipients_mode_no_recipients_fallback(self, mock_user_manager_class):
        """测试模式2：管理员未配置收款人（降级）"""
        # 创建Mock用户管理器
        mock_manager = MagicMock()
        mock_user_manager_class.return_value = mock_manager
        
        # 创建Mock用户（没有收款人）
        mock_user = MagicMock()
        mock_user.user_id = "user_001"
        mock_user.user_name = "无收款人管理员"
        mock_user.enabled = True
        mock_user.transfer_recipients = []  # 没有收款人
        
        mock_manager.get_account_user.return_value = mock_user
        
        # 创建Mock选择器
        mock_selector = MagicMock()
        
        self.config.set_transfer_target_mode("manager_recipients")
        
        with patch.object(self.config, 'get_transfer_recipient', return_value="15000150000") as mock_method:
            # 测试转账
            recipient = self.config.get_transfer_recipient_enhanced(
                phone="13800138000",
                user_id="test_user_id",
                current_level=0,
                selector=mock_selector
            )
            
            # 应该降级到系统配置收款人
            mock_method.assert_called_once()
            self.assertEqual(recipient, "15000150000")
    
    @patch('src.transfer_config.UserManager')
    def test_no_manager_assigned_fallback(self, mock_user_manager_class):
        """测试账号未分配管理员（降级）"""
        # 创建Mock用户管理器
        mock_manager = MagicMock()
        mock_user_manager_class.return_value = mock_manager
        
        # 账号未分配管理员
        mock_manager.get_account_user.return_value = None
        
        self.config.set_transfer_target_mode("manager_recipients")
        
        with patch.object(self.config, 'get_transfer_recipient', return_value="15000150000") as mock_method:
            # 测试转账
            recipient = self.config.get_transfer_recipient_enhanced(
                phone="19999999999",
                user_id="test_user_id",
                current_level=0,
                selector=None
            )
            
            # 应该降级到系统配置收款人
            mock_method.assert_called_once()
            self.assertEqual(recipient, "15000150000")
    
    @patch('src.transfer_config.UserManager')
    def test_disabled_manager_fallback(self, mock_user_manager_class):
        """测试管理员已禁用（降级）"""
        # 创建Mock用户管理器
        mock_manager = MagicMock()
        mock_user_manager_class.return_value = mock_manager
        
        # 创建Mock用户（已禁用）
        mock_user = MagicMock()
        mock_user.user_id = "user_001"
        mock_user.user_name = "已禁用管理员"
        mock_user.enabled = False  # 已禁用
        
        mock_manager.get_account_user.return_value = mock_user
        
        self.config.set_transfer_target_mode("manager_recipients")
        
        with patch.object(self.config, 'get_transfer_recipient', return_value="15000150000") as mock_method:
            # 测试转账
            recipient = self.config.get_transfer_recipient_enhanced(
                phone="13800138000",
                user_id="test_user_id",
                current_level=0,
                selector=None
            )
            
            # 应该降级到系统配置收款人
            mock_method.assert_called_once()
            self.assertEqual(recipient, "15000150000")
    
    @patch('src.transfer_config.UserManager')
    def test_self_filtering_in_manager_account_mode(self, mock_user_manager_class):
        """测试模式1中自动过滤自己"""
        # 创建Mock用户管理器
        mock_manager = MagicMock()
        mock_user_manager_class.return_value = mock_manager
        
        # 创建Mock用户
        mock_user = MagicMock()
        mock_user.user_id = "user_001"
        mock_user.user_name = "双账号管理员"
        mock_user.enabled = True
        
        mock_manager.get_account_user.return_value = mock_user
        mock_manager.get_user_accounts.return_value = ["13800138005", "13800138006"]
        
        self.config.set_transfer_target_mode("manager_account")
        
        # 账号1转账
        recipient1 = self.config.get_transfer_recipient_enhanced(
            phone="13800138005",
            user_id="test_user_id",
            current_level=0,
            selector=None
        )
        
        # 应该选择账号2
        self.assertEqual(recipient1, "13800138006")
        
        # 账号2转账
        recipient2 = self.config.get_transfer_recipient_enhanced(
            phone="13800138006",
            user_id="test_user_id",
            current_level=0,
            selector=None
        )
        
        # 应该选择账号1
        self.assertEqual(recipient2, "13800138005")


class TestTransferConfigEdgeCases(unittest.TestCase):
    """转账配置边界情况测试"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        self.temp_file.close()
        
        self.config = TransferConfig()
        self.config.config_file = Path(self.temp_file.name)
        self.config.save()
    
    def tearDown(self):
        """测试后清理"""
        try:
            os.unlink(self.temp_file.name)
        except:
            pass
    
    def test_empty_config_file(self):
        """测试空配置文件"""
        # 创建空配置文件
        with open(self.temp_file.name, 'w', encoding='utf-8') as f:
            f.write("{}")
        
        config = TransferConfig()
        config.config_file = Path(self.temp_file.name)
        config.load()
        
        # 应该使用默认值
        self.assertEqual(config.transfer_target_mode, "manager_recipients")
        self.assertFalse(config.enabled)
    
    def test_corrupted_config_file(self):
        """测试损坏的配置文件"""
        # 创建损坏的配置文件
        with open(self.temp_file.name, 'w', encoding='utf-8') as f:
            f.write("{ invalid json }")
        
        config = TransferConfig()
        config.config_file = Path(self.temp_file.name)
        config.load()
        
        # 应该使用默认值（不崩溃）
        self.assertEqual(config.transfer_target_mode, "manager_recipients")
    
    def test_mode_switching(self):
        """测试模式切换"""
        # 切换到模式1
        self.config.set_transfer_target_mode("manager_account")
        self.assertEqual(self.config.transfer_target_mode, "manager_account")
        
        # 切换到模式2
        self.config.set_transfer_target_mode("manager_recipients")
        self.assertEqual(self.config.transfer_target_mode, "manager_recipients")
        
        # 切换到模式3
        self.config.set_transfer_target_mode("system_recipients")
        self.assertEqual(self.config.transfer_target_mode, "system_recipients")
        
        # 验证持久化
        new_config = TransferConfig()
        new_config.config_file = Path(self.temp_file.name)
        new_config.load()
        self.assertEqual(new_config.transfer_target_mode, "system_recipients")
    
    def test_concurrent_mode_changes(self):
        """测试并发模式更改"""
        # 模拟多次快速切换
        for _ in range(10):
            self.config.set_transfer_target_mode("manager_account")
            self.config.set_transfer_target_mode("manager_recipients")
            self.config.set_transfer_target_mode("system_recipients")
        
        # 最后的值应该被保存
        self.assertEqual(self.config.transfer_target_mode, "system_recipients")
        
        # 验证持久化
        new_config = TransferConfig()
        new_config.config_file = Path(self.temp_file.name)
        new_config.load()
        self.assertEqual(new_config.transfer_target_mode, "system_recipients")


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestTransferConfigBasic))
    suite.addTests(loader.loadTestsFromTestCase(TestTransferModeLogic))
    suite.addTests(loader.loadTestsFromTestCase(TestTransferConfigEdgeCases))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 打印总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    print(f"总测试数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print("=" * 70)
    
    # 返回测试结果
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
