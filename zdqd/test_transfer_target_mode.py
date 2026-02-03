"""
测试转账目标模式功能
Test Transfer Target Mode Feature
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.transfer_config import get_transfer_config
from src.recipient_selector import RecipientSelector
from src.user_manager import UserManager


def test_transfer_target_modes():
    """测试三种转账目标模式"""
    
    print("=" * 60)
    print("转账目标模式测试")
    print("=" * 60)
    
    # 获取配置
    config = get_transfer_config()
    selector = RecipientSelector(strategy="rotation")
    manager = UserManager()
    
    # 测试账号
    test_phone = "13800138000"
    test_user_id = "1234567"
    
    print(f"\n测试账号: {test_phone}")
    print(f"用户ID: {test_user_id}")
    
    # 获取账号的管理员信息
    user = manager.get_account_user(test_phone)
    if user:
        print(f"管理员: {user.user_name} ({user.user_id})")
        print(f"管理员的收款人: {', '.join(user.transfer_recipients) if user.transfer_recipients else '未配置'}")
        
        # 获取管理员的账号列表
        manager_accounts = manager.get_user_accounts(user.user_id)
        print(f"管理员的账号: {', '.join(manager_accounts) if manager_accounts else '无'}")
    else:
        print("未分配管理员")
    
    # 显示系统配置的收款人
    print(f"\n系统配置收款人: {', '.join(config.recipient_ids) if config.recipient_ids else '未配置'}")
    
    print("\n" + "=" * 60)
    print("测试三种模式")
    print("=" * 60)
    
    # 模式1：转给管理员自己
    print("\n【模式1】转给管理员自己的账号")
    print("-" * 60)
    config.set_transfer_target_mode("manager_account")
    print(f"当前模式: {config.get_transfer_target_mode_display()}")
    
    recipient = config.get_transfer_recipient_enhanced(
        phone=test_phone,
        user_id=test_user_id,
        current_level=0,
        selector=selector
    )
    print(f"选中的收款人: {recipient if recipient else '无'}")
    
    # 模式2：转给管理员的收款人（默认）
    print("\n【模式2】转给管理员配置的收款人（默认）")
    print("-" * 60)
    config.set_transfer_target_mode("manager_recipients")
    print(f"当前模式: {config.get_transfer_target_mode_display()}")
    
    recipient = config.get_transfer_recipient_enhanced(
        phone=test_phone,
        user_id=test_user_id,
        current_level=0,
        selector=selector
    )
    print(f"选中的收款人: {recipient if recipient else '无'}")
    
    # 模式3：转给系统配置收款人
    print("\n【模式3】转给系统配置的收款人")
    print("-" * 60)
    config.set_transfer_target_mode("system_recipients")
    print(f"当前模式: {config.get_transfer_target_mode_display()}")
    
    recipient = config.get_transfer_recipient_enhanced(
        phone=test_phone,
        user_id=test_user_id,
        current_level=0,
        selector=selector
    )
    print(f"选中的收款人: {recipient if recipient else '无'}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
    
    # 恢复默认模式
    config.set_transfer_target_mode("manager_recipients")
    print(f"\n已恢复默认模式: {config.get_transfer_target_mode_display()}")


def show_config_example():
    """显示配置文件示例"""
    print("\n" + "=" * 60)
    print("transfer_config.json 配置示例")
    print("=" * 60)
    
    example = """
{
  "min_balance": 0.0,
  "min_transfer_amount": 30.0,
  "recipient_ids": ["13900139000", "14000140000"],
  "enabled": true,
  "use_user_manager_recipients": true,
  "transfer_target_mode": "manager_recipients",
  
  "说明": {
    "transfer_target_mode": {
      "manager_account": "转给管理员自己的账号",
      "manager_recipients": "转给管理员配置的收款人（默认）",
      "system_recipients": "转给系统配置的收款人"
    }
  }
}
"""
    print(example)


if __name__ == "__main__":
    # 显示配置示例
    show_config_example()
    
    # 运行测试
    print("\n按回车键开始测试...")
    input()
    
    test_transfer_target_modes()
