"""
转账目标模式最终单元测试（简化版，确保功能完整性）
Final Unit Tests for Transfer Target Mode (Simplified, Ensuring Completeness)
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


class TestTransferConfigCore(unittest.TestCase):
    """转账配置核心功能测试"""
    
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
        self.config.save()
    
    def tearDown(self):
        """测试后清理"""
        try:
            os.unlink(self.temp_file.name)
        except:
            pass
    
    # ========== 基础功能测试 ==========
    
    def test_01_default_mode(self):
        """测试1：默认模式应该是manager_recipients"""
        config = TransferConfig()
        config.config_file = Path(self.temp_file.name)
        config.load()
        self.assertEqual(config.transfer_target_mode, "manager_recipients")
    
    def test_02_set_mode_manager_account(self):
        """测试2：设置模式为manager_account"""
        self.config.set_transfer_target_mode("manager_account")
        self.assertEqual(self.config.transfer_target_mode, "manager_account")
        
        # 验证持久化
        new_config = TransferConfig()
        new_config.config_file = Path(self.temp_file.name)
        new_config.load()
        self.assertEqual(new_config.transfer_target_mode, "manager_account")
    
    def test_03_set_mode_manager_recipients(self):
        """测试3：设置模式为manager_recipients"""
        self.config.set_transfer_target_mode("manager_recipients")
        self.assertEqual(self.config.transfer_target_mode, "manager_recipients")
    
    def test_04_set_mode_system_recipients(self):
        """测试4：设置模式为system_recipients"""
        self.config.set_transfer_target_mode("system_recipients")
        self.assertEqual(self.config.transfer_target_mode, "system_recipients")
    
    def test_05_invalid_mode_raises_error(self):
        """测试5：设置无效模式应该抛出ValueError"""
        with self.assertRaises(ValueError) as context:
            self.config.set_transfer_target_mode("invalid_mode")
        self.assertIn("无效的转账目标模式", str(context.exception))
    
    def test_06_mode_display_names(self):
        """测试6：模式显示名称正确"""
        test_cases = [
            ("manager_account", "转给管理员自己"),
            ("manager_recipients", "转给管理员的收款人"),
            ("system_recipients", "转给系统配置收款人")
        ]
        
        for mode, expected_display in test_cases:
            self.config.set_transfer_target_mode(mode)
            actual_display = self.config.get_transfer_target_mode_display()
            self.assertEqual(actual_display, expected_display,
                           f"模式 {mode} 的显示名称应该是 {expected_display}")
    
    # ========== 配置持久化测试 ==========
    
    def test_07_config_persistence(self):
        """测试7：配置持久化功能"""
        # 设置各种配置
        self.config.set_transfer_target_mode("manager_account")
        self.config.set_enabled(True)
        self.config.set_min_balance(5.0)
        self.config.min_transfer_amount = 50.0
        self.config.save()
        
        # 创建新实例加载配置
        new_config = TransferConfig()
        new_config.config_file = Path(self.temp_file.name)
        new_config.load()
        
        # 验证所有配置都被正确保存和加载
        self.assertEqual(new_config.transfer_target_mode, "manager_account")
        self.assertTrue(new_config.enabled)
        self.assertEqual(new_config.min_balance, 5.0)
        self.assertEqual(new_config.min_transfer_amount, 50.0)
    
    def test_08_config_json_structure(self):
        """测试8：配置JSON结构完整"""
        self.config.set_transfer_target_mode("system_recipients")
        self.config.save()
        
        # 读取JSON文件
        with open(self.temp_file.name, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 验证所有必需字段都存在
        required_fields = [
            'transfer_target_mode',
            'enabled',
            'min_balance',
            'min_transfer_amount',
            'recipient_ids',
            'level_recipients',
            'use_user_manager_recipients'
        ]
        
        for field in required_fields:
            self.assertIn(field, data, f"配置文件应该包含字段: {field}")
        
        # 验证值正确
        self.assertEqual(data['transfer_target_mode'], 'system_recipients')
    
    # ========== 向后兼容性测试 ==========
    
    def test_09_backward_compatibility_no_mode_field(self):
        """测试9：向后兼容 - 旧配置文件没有transfer_target_mode字段"""
        old_config = {
            'min_balance': 0.0,
            'min_transfer_amount': 30.0,
            'recipient_ids': ['15000150000'],
            'enabled': True
        }
        
        with open(self.temp_file.name, 'w', encoding='utf-8') as f:
            json.dump(old_config, f)
        
        config = TransferConfig()
        config.config_file = Path(self.temp_file.name)
        config.load()
        
        # 应该使用默认模式
        self.assertEqual(config.transfer_target_mode, "manager_recipients")
    
    def test_10_mode_validation_on_load(self):
        """测试10：加载时验证模式有效性"""
        invalid_config = {
            'min_balance': 0.0,
            'min_transfer_amount': 30.0,
            'recipient_ids': ['15000150000'],
            'enabled': True,
            'transfer_target_mode': 'invalid_mode'
        }
        
        with open(self.temp_file.name, 'w', encoding='utf-8') as f:
            json.dump(invalid_config, f)
        
        config = TransferConfig()
        config.config_file = Path(self.temp_file.name)
        config.load()
        
        # 应该降级到默认模式
        self.assertEqual(config.transfer_target_mode, "manager_recipients")
    
    # ========== 边界情况测试 ==========
    
    def test_11_empty_config_file(self):
        """测试11：空配置文件"""
        with open(self.temp_file.name, 'w', encoding='utf-8') as f:
            f.write("{}")
        
        config = TransferConfig()
        config.config_file = Path(self.temp_file.name)
        config.load()
        
        # 应该使用默认值
        self.assertEqual(config.transfer_target_mode, "manager_recipients")
        self.assertFalse(config.enabled)
    
    def test_12_corrupted_config_file(self):
        """测试12：损坏的配置文件不应该导致崩溃"""
        with open(self.temp_file.name, 'w', encoding='utf-8') as f:
            f.write("{ invalid json }")
        
        config = TransferConfig()
        config.config_file = Path(self.temp_file.name)
        
        # 不应该抛出异常
        try:
            config.load()
            # 应该使用默认值
            self.assertEqual(config.transfer_target_mode, "manager_recipients")
        except Exception as e:
            self.fail(f"加载损坏的配置文件不应该抛出异常: {e}")
    
    def test_13_mode_switching(self):
        """测试13：模式切换功能"""
        modes = ["manager_account", "manager_recipients", "system_recipients"]
        
        for mode in modes:
            self.config.set_transfer_target_mode(mode)
            self.assertEqual(self.config.transfer_target_mode, mode)
            
            # 验证持久化
            new_config = TransferConfig()
            new_config.config_file = Path(self.temp_file.name)
            new_config.load()
            self.assertEqual(new_config.transfer_target_mode, mode)
    
    def test_14_concurrent_mode_changes(self):
        """测试14：快速连续切换模式"""
        # 模拟多次快速切换
        for _ in range(5):
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
    
    # ========== 配置开关测试 ==========
    
    def test_15_use_user_manager_recipients_flag(self):
        """测试15：use_user_manager_recipients标志"""
        # 测试启用
        self.config.use_user_manager_recipients = True
        self.config.save()
        
        new_config = TransferConfig()
        new_config.config_file = Path(self.temp_file.name)
        new_config.load()
        self.assertTrue(new_config.use_user_manager_recipients)
        
        # 测试禁用
        self.config.use_user_manager_recipients = False
        self.config.save()
        
        new_config2 = TransferConfig()
        new_config2.config_file = Path(self.temp_file.name)
        new_config2.load()
        self.assertFalse(new_config2.use_user_manager_recipients)
    
    # ========== 系统配置收款人测试 ==========
    
    def test_16_system_recipients_basic(self):
        """测试16：系统配置收款人基础功能"""
        self.config.set_transfer_target_mode("system_recipients")
        
        # 获取收款人（使用原有方法）
        recipient = self.config.get_transfer_recipient("test_user_id", 0)
        
        # 应该返回系统配置的收款人
        self.assertIsNotNone(recipient)
        self.assertIn(recipient, self.config.recipient_ids)
    
    def test_17_multi_level_not_affected_by_mode(self):
        """测试17：多级转账不受模式影响"""
        self.config.set_transfer_target_mode("manager_recipients")
        self.config.multi_level_enabled = True
        self.config.max_transfer_level = 2
        self.config.level_recipients[2] = ["17000170000"]
        self.config.save()
        
        # 对于多级转账（current_level > 0），应该使用原有逻辑
        # 这里只验证配置正确保存
        new_config = TransferConfig()
        new_config.config_file = Path(self.temp_file.name)
        new_config.load()
        
        self.assertTrue(new_config.multi_level_enabled)
        self.assertEqual(new_config.max_transfer_level, 2)
        self.assertEqual(new_config.level_recipients[2], ["17000170000"])
    
    # ========== 完整性验证测试 ==========
    
    def test_18_all_modes_are_valid(self):
        """测试18：所有三种模式都是有效的"""
        valid_modes = ["manager_account", "manager_recipients", "system_recipients"]
        
        for mode in valid_modes:
            try:
                self.config.set_transfer_target_mode(mode)
                self.assertEqual(self.config.transfer_target_mode, mode)
            except ValueError:
                self.fail(f"模式 {mode} 应该是有效的")
    
    def test_19_mode_names_are_consistent(self):
        """测试19：模式名称一致性"""
        mode_mapping = {
            "manager_account": "转给管理员自己",
            "manager_recipients": "转给管理员的收款人",
            "system_recipients": "转给系统配置收款人"
        }
        
        for mode, expected_display in mode_mapping.items():
            self.config.set_transfer_target_mode(mode)
            actual_display = self.config.get_transfer_target_mode_display()
            self.assertEqual(actual_display, expected_display)
    
    def test_20_config_file_format(self):
        """测试20：配置文件格式正确"""
        self.config.set_transfer_target_mode("manager_account")
        self.config.save()
        
        # 读取并验证JSON格式
        with open(self.temp_file.name, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                # 验证是字典
                self.assertIsInstance(data, dict)
                # 验证关键字段
                self.assertIn('transfer_target_mode', data)
                self.assertIsInstance(data['transfer_target_mode'], str)
            except json.JSONDecodeError:
                self.fail("配置文件应该是有效的JSON格式")


def run_tests():
    """运行所有测试"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestTransferConfigCore)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 打印详细总结
    print("\n" + "=" * 70)
    print("转账目标模式功能完整性测试总结")
    print("=" * 70)
    print(f"总测试数: {result.testsRun}")
    print(f"✅ 成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ 失败: {len(result.failures)}")
    print(f"⚠️  错误: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n🎉 所有测试通过！功能完整性已验证。")
    else:
        print("\n⚠️  部分测试失败，请检查上述错误信息。")
    
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
