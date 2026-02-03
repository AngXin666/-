"""
转账配置模式单元测试（简化版，避免数据库锁定）
Unit Tests for Transfer Config Modes (Simplified)
"""

import unittest
import sys
import os
import json
import tempfile
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from src.transfer_config import TransferConfig


class TestTransferConfigModes(unittest.TestCase):
    """转账配置模式测试类（不涉及数据库）"""
    
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


def run_tests():
    """运行所有测试"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestTransferConfigModes)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
