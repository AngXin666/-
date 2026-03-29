"""
测试转账配置中的签到次数功能
"""

import sys
import os
import json
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_transfer_config_checkin_days():
    """测试签到次数配置的保存和加载"""
    print("=" * 60)
    print("测试：转账配置 - 签到次数功能")
    print("=" * 60)
    
    # 备份原配置文件（如果存在）
    config_file = Path("transfer_config.json")
    backup_file = Path("transfer_config.json.backup")
    
    if config_file.exists():
        import shutil
        shutil.copy(config_file, backup_file)
        print("✓ 已备份原配置文件")
    
    try:
        # 删除现有配置文件，从头开始测试
        if config_file.exists():
            config_file.unlink()
            print("✓ 已删除现有配置文件")
        
        # 1. 测试默认值
        print("\n[测试1] 检查默认值")
        from transfer_config import TransferConfig
        config = TransferConfig()
        
        assert hasattr(config, 'min_checkin_days_enabled'), "❌ 缺少 min_checkin_days_enabled 属性"
        assert hasattr(config, 'min_checkin_days'), "❌ 缺少 min_checkin_days 属性"
        assert config.min_checkin_days_enabled == False, f"❌ 默认开关应为 False，实际为 {config.min_checkin_days_enabled}"
        assert config.min_checkin_days == 7, f"❌ 默认次数应为 7，实际为 {config.min_checkin_days}"
        print("✓ 默认值正确：开关=False, 次数=7")
        
        # 2. 测试保存功能
        print("\n[测试2] 测试保存功能")
        config.min_checkin_days_enabled = True
        config.min_checkin_days = 10
        config.save()
        print("✓ 已保存配置：开关=True, 次数=10")
        
        # 验证文件是否创建
        assert config_file.exists(), "❌ 配置文件未创建"
        print("✓ 配置文件已创建")
        
        # 验证文件内容
        with open(config_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert 'min_checkin_days_enabled' in data, "❌ 配置文件中缺少 min_checkin_days_enabled"
        assert 'min_checkin_days' in data, "❌ 配置文件中缺少 min_checkin_days"
        assert data['min_checkin_days_enabled'] == True, f"❌ 保存的开关值错误：{data['min_checkin_days_enabled']}"
        assert data['min_checkin_days'] == 10, f"❌ 保存的次数值错误：{data['min_checkin_days']}"
        print("✓ 配置文件内容正确")
        
        # 3. 测试加载功能
        print("\n[测试3] 测试加载功能")
        config2 = TransferConfig()
        assert config2.min_checkin_days_enabled == True, f"❌ 加载的开关值错误：{config2.min_checkin_days_enabled}"
        assert config2.min_checkin_days == 10, f"❌ 加载的次数值错误：{config2.min_checkin_days}"
        print("✓ 配置加载正确：开关=True, 次数=10")
        
        # 4. 测试修改功能
        print("\n[测试4] 测试修改功能")
        config2.min_checkin_days_enabled = False
        config2.min_checkin_days = 15
        config2.save()
        print("✓ 已修改配置：开关=False, 次数=15")
        
        config3 = TransferConfig()
        assert config3.min_checkin_days_enabled == False, f"❌ 修改后的开关值错误：{config3.min_checkin_days_enabled}"
        assert config3.min_checkin_days == 15, f"❌ 修改后的次数值错误：{config3.min_checkin_days}"
        print("✓ 修改后的配置加载正确")
        
        # 5. 测试兼容性（加载旧配置文件）
        print("\n[测试5] 测试向后兼容性")
        # 创建一个不包含签到次数字段的旧配置文件
        old_config = {
            'min_balance': 5.0,
            'min_transfer_amount': 30.0,
            'enabled': True
        }
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(old_config, f, ensure_ascii=False, indent=2)
        print("✓ 已创建旧版本配置文件（不包含签到次数字段）")
        
        config4 = TransferConfig()
        assert config4.min_checkin_days_enabled == False, f"❌ 旧配置加载后开关应为默认值 False，实际为 {config4.min_checkin_days_enabled}"
        assert config4.min_checkin_days == 7, f"❌ 旧配置加载后次数应为默认值 7，实际为 {config4.min_checkin_days}"
        print("✓ 旧配置文件兼容性测试通过（使用默认值）")
        
        print("\n" + "=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)
        
        # 显示测试总结
        print("\n测试总结：")
        print("1. ✓ 默认值正确")
        print("2. ✓ 保存功能正常")
        print("3. ✓ 加载功能正常")
        print("4. ✓ 修改功能正常")
        print("5. ✓ 向后兼容性正常")
        
        return True
        
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        return False
        
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # 恢复原配置文件
        if backup_file.exists():
            import shutil
            shutil.copy(backup_file, config_file)
            backup_file.unlink()
            print("\n✓ 已恢复原配置文件")
        elif config_file.exists():
            config_file.unlink()
            print("\n✓ 已删除测试配置文件")


if __name__ == "__main__":
    success = test_transfer_config_checkin_days()
    sys.exit(0 if success else 1)
