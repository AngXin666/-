"""
多收款人转账功能演示
Multi-Recipient Transfer Demo

演示如何使用多收款人转账功能
"""

from src.user_manager import UserManager, User
from src.transfer_config import TransferConfig
from src.recipient_selector import RecipientSelector


def demo_multi_recipient_transfer():
    """演示多收款人转账功能"""
    print("=" * 70)
    print("多收款人转账功能演示")
    print("=" * 70)
    
    # 1. 创建用户管理器和转账配置
    print("\n[步骤1] 初始化组件...")
    manager = UserManager()
    transfer_config = TransferConfig()
    selector = RecipientSelector(strategy="rotation")
    print("✓ 组件初始化完成")
    
    # 2. 创建测试用户（带多个收款人）
    print("\n[步骤2] 创建测试用户...")
    test_user = User(
        user_id="demo_user_001",
        user_name="演示用户",
        transfer_recipients=[
            "13800138000",  # 收款人1
            "13900139000",  # 收款人2
            "14000140000"   # 收款人3
        ],
        enabled=True,
        description="演示多收款人功能"
    )
    
    # 清理旧数据
    if manager.get_user(test_user.user_id):
        manager.delete_user(test_user.user_id)
    
    if manager.add_user(test_user):
        print(f"✓ 用户创建成功: {test_user.user_name}")
        print(f"  用户ID: {test_user.user_id}")
        print(f"  收款人数量: {len(test_user.transfer_recipients)}")
        print(f"  收款人列表:")
        for i, recipient in enumerate(test_user.transfer_recipients, 1):
            print(f"    {i}. {recipient}")
    else:
        print("✗ 用户创建失败")
        return
    
    # 3. 分配账号
    print("\n[步骤3] 分配账号...")
    test_phone = "18888888888"
    if manager.assign_account(test_phone, test_user.user_id):
        print(f"✓ 账号分配成功: {test_phone} → {test_user.user_name}")
    else:
        print("✗ 账号分配失败")
        return
    
    # 4. 演示轮询选择
    print("\n[步骤4] 演示轮询选择...")
    print("连续选择5次，观察轮询效果：")
    
    # 重置轮询状态
    selector.reset_rotation(test_user.user_id)
    
    for i in range(5):
        recipient = transfer_config.get_transfer_recipient_from_user_manager(
            test_phone,
            selector
        )
        rotation_index = selector.get_rotation_index(test_user.user_id)
        print(f"  第{i+1}次选择: {recipient} (下次索引: {rotation_index})")
    
    # 5. 演示持久化
    print("\n[步骤5] 演示持久化（模拟重启）...")
    print("创建新的选择器实例（模拟程序重启）...")
    selector2 = RecipientSelector(strategy="rotation")
    recipient = transfer_config.get_transfer_recipient_from_user_manager(
        test_phone,
        selector2
    )
    print(f"✓ 重启后继续轮询: {recipient}")
    print("  说明：轮询状态已持久化，重启后从上次位置继续")
    
    # 6. 演示自我过滤
    print("\n[步骤6] 演示自我过滤...")
    print("创建一个收款人包括自己的用户...")
    test_user2 = User(
        user_id="demo_user_002",
        user_name="演示用户2",
        transfer_recipients=[
            "19999999999",  # 自己
            "13800138000",  # 收款人1
            "13900139000"   # 收款人2
        ],
        enabled=True
    )
    
    if manager.get_user(test_user2.user_id):
        manager.delete_user(test_user2.user_id)
    
    manager.add_user(test_user2)
    manager.assign_account("19999999999", test_user2.user_id)
    
    selector3 = RecipientSelector(strategy="rotation")
    selector3.reset_rotation(test_user2.user_id)
    
    recipient = transfer_config.get_transfer_recipient_from_user_manager(
        "19999999999",
        selector3
    )
    print(f"✓ 自动过滤自己: {recipient}")
    print("  说明：自动跳过发送人自己，避免循环转账")
    
    # 7. 演示降级机制
    print("\n[步骤7] 演示降级机制...")
    print("尝试获取未配置用户的收款人...")
    
    # 配置转账配置的收款人作为降级选项
    transfer_config.add_recipient("fallback_recipient", level=1)
    
    recipient = transfer_config.get_transfer_recipient_enhanced(
        phone="10000000000",  # 未分配的手机号
        user_id="unknown_user",
        current_level=0,
        selector=selector
    )
    print(f"✓ 降级到转账配置: {recipient}")
    print("  说明：用户管理未配置时，自动降级到转账配置")
    
    # 8. 演示随机选择
    print("\n[步骤8] 演示随机选择...")
    random_selector = RecipientSelector(strategy="random")
    print("连续随机选择5次：")
    for i in range(5):
        recipient = transfer_config.get_transfer_recipient_from_user_manager(
            test_phone,
            random_selector
        )
        print(f"  第{i+1}次选择: {recipient}")
    print("  说明：随机策略每次随机选择一个收款人")
    
    # 9. 清理测试数据
    print("\n[步骤9] 清理测试数据...")
    manager.delete_user(test_user.user_id)
    manager.delete_user(test_user2.user_id)
    transfer_config.remove_recipient("fallback_recipient", level=1)
    selector.reset_rotation()
    print("✓ 测试数据已清理")
    
    print("\n" + "=" * 70)
    print("✅ 演示完成")
    print("=" * 70)
    print("\n功能特性总结：")
    print("  1. ✓ 支持多个收款人配置")
    print("  2. ✓ 轮询选择策略（负载均衡）")
    print("  3. ✓ 随机选择策略（不可预测）")
    print("  4. ✓ 轮询状态持久化（重启后继续）")
    print("  5. ✓ 自我过滤（避免循环转账）")
    print("  6. ✓ 降级机制（用户管理不可用时）")
    print("  7. ✓ 多级转账兼容")
    print("  8. ✓ 配置开关（可回滚）")


if __name__ == '__main__':
    demo_multi_recipient_transfer()
