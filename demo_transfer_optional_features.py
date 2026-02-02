"""
多收款人转账可选功能演示
Demo for Multi-Recipient Transfer Optional Features
"""

import asyncio
from datetime import datetime


def demo_transfer_history():
    """演示转账历史记录功能"""
    print("=" * 70)
    print("演示 1: 转账历史记录功能")
    print("=" * 70)
    
    from src.transfer_history import get_transfer_history
    
    history = get_transfer_history()
    
    # 模拟保存几条转账记录
    print("\n[步骤 1] 保存转账记录...")
    
    test_records = [
        {
            'sender_phone': '13800138001',
            'sender_user_id': 'user_001',
            'sender_name': '张三',
            'recipient_phone': '13900139001',
            'recipient_name': '李四',
            'amount': 150.00,
            'strategy': 'rotation',
            'success': True,
            'owner': '团队A'
        },
        {
            'sender_phone': '13800138002',
            'sender_user_id': 'user_002',
            'sender_name': '王五',
            'recipient_phone': '13900139002',
            'recipient_name': '赵六',
            'amount': 200.50,
            'strategy': 'random',
            'success': True,
            'owner': '团队B'
        },
        {
            'sender_phone': '13800138001',
            'sender_user_id': 'user_001',
            'sender_name': '张三',
            'recipient_phone': '13900139003',
            'recipient_name': '孙七',
            'amount': 0.0,
            'strategy': 'rotation',
            'success': False,
            'error_message': '余额不足',
            'owner': '团队A'
        }
    ]
    
    for record in test_records:
        history.save_transfer_record(**record)
        status = '✓ 成功' if record['success'] else '✗ 失败'
        print(f"  {status}: {record['sender_name']} -> {record['recipient_name']} "
              f"({record['amount']:.2f}元)")
    
    # 查询记录
    print("\n[步骤 2] 查询转账记录...")
    records = history.get_transfer_records(limit=10)
    print(f"  查询到 {len(records)} 条最近的记录")
    
    # 获取统计信息
    print("\n[步骤 3] 获取统计信息...")
    stats = history.get_transfer_statistics(days=30)
    
    print(f"\n  📊 统计周期: 最近 30 天")
    print(f"  📈 总转账次数: {stats['total_count']} 次")
    print(f"  ✅ 成功次数: {stats['success_count']} 次")
    print(f"  ❌ 失败次数: {stats['failed_count']} 次")
    print(f"  📊 成功率: {stats['success_rate']:.1f}%")
    print(f"  💰 总金额: {stats['total_amount']:.2f} 元")
    
    if stats['recipient_stats']:
        print(f"\n  🏆 收款人排行榜 (Top 3):")
        for i, recipient_stat in enumerate(stats['recipient_stats'][:3], 1):
            print(f"    {i}. {recipient_stat['name']}: "
                  f"{recipient_stat['count']}次, "
                  f"{recipient_stat['amount']:.2f}元")
    
    print("\n✓ 转账历史记录功能演示完成")


def demo_transfer_retry():
    """演示转账重试机制"""
    print("\n" + "=" * 70)
    print("演示 2: 转账重试机制")
    print("=" * 70)
    
    from src.transfer_retry import get_transfer_retry
    
    retry_manager = get_transfer_retry(max_retries=3, retry_delay=1.0)
    
    # 场景1: 网络错误，重试后成功
    print("\n[场景 1] 网络错误，重试后成功")
    print("-" * 70)
    
    attempt_count = [0]
    
    async def mock_network_error_transfer(device_id, recipient_id, log_callback=None, **kwargs):
        """模拟网络错误的转账"""
        attempt_count[0] += 1
        
        if log_callback:
            log_callback(f"  尝试转账到 {recipient_id}（第{attempt_count[0]}次）")
        
        # 前2次失败，第3次成功
        if attempt_count[0] < 3:
            return {
                'success': False,
                'message': '网络连接超时',
                'amount': 0.0,
                'chain': []
            }
        else:
            return {
                'success': True,
                'message': '转账成功',
                'amount': 100.0,
                'chain': []
            }
    
    async def run_test():
        result = await retry_manager.transfer_with_retry(
            transfer_func=mock_network_error_transfer,
            device_id="test_device",
            recipient_id="13900139000",
            log_callback=print
        )
        return result
    
    result = asyncio.run(run_test())
    
    print(f"\n  结果: {'✓ 成功' if result['success'] else '✗ 失败'}")
    print(f"  尝试次数: {attempt_count[0]}")
    print(f"  消息: {result['message']}")
    
    # 场景2: 账号冻结，不重试
    print("\n[场景 2] 账号冻结，不重试")
    print("-" * 70)
    
    from src.models.error_types import ErrorType
    
    async def mock_frozen_account_transfer(device_id, recipient_id, log_callback=None, **kwargs):
        """模拟账号冻结的转账"""
        if log_callback:
            log_callback("  检测到账号已冻结")
        return {
            'success': False,
            'message': '账号已冻结，无法转账',
            'error_type': ErrorType.ACCOUNT_FROZEN,
            'amount': 0.0,
            'chain': []
        }
    
    async def run_frozen_test():
        attempt_count[0] = 0
        result = await retry_manager.transfer_with_retry(
            transfer_func=mock_frozen_account_transfer,
            device_id="test_device",
            recipient_id="13900139000",
            log_callback=print
        )
        return result
    
    result = asyncio.run(run_frozen_test())
    
    print(f"\n  结果: {'✓ 成功' if result['success'] else '✗ 失败'}")
    print(f"  消息: {result['message']}")
    print(f"  说明: 账号冻结错误不会重试，避免浪费时间")
    
    print("\n✓ 转账重试机制演示完成")


def demo_transfer_history_gui():
    """演示转账历史GUI"""
    print("\n" + "=" * 70)
    print("演示 3: 转账历史GUI界面")
    print("=" * 70)
    
    print("\n准备打开转账历史GUI窗口...")
    print("\n功能说明:")
    print("  1. 📋 记录列表 - 显示所有转账记录")
    print("  2. 🔍 筛选功能 - 按发送人、收款人、管理员、日期范围筛选")
    print("  3. 📊 统计信息 - 实时显示统计数据")
    print("  4. 👆 双击记录 - 查看详细信息")
    print("  5. 📤 导出CSV - 导出记录到CSV文件")
    print("  6. 🔄 刷新按钮 - 刷新数据")
    
    response = input("\n是否打开GUI窗口？(y/n): ").strip().lower()
    
    if response == 'y':
        try:
            from src.transfer_history_gui import TransferHistoryGUI
            
            print("\n正在打开GUI窗口...")
            gui = TransferHistoryGUI()
            
            print("✓ GUI窗口已打开")
            print("  请在窗口中测试各项功能")
            print("  关闭窗口后演示将继续...")
            
            gui.show()
            
            print("\n✓ 转账历史GUI演示完成")
            
        except Exception as e:
            print(f"\n✗ 打开GUI失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n跳过GUI演示")


def demo_integration():
    """演示完整集成流程"""
    print("\n" + "=" * 70)
    print("演示 4: 完整集成流程")
    print("=" * 70)
    
    print("\n在实际使用中，这些功能是如何协同工作的：")
    print()
    print("  1️⃣  用户配置多个收款人")
    print("      ↓")
    print("  2️⃣  系统使用轮询/随机策略选择收款人")
    print("      ↓")
    print("  3️⃣  执行转账（自动使用重试机制）")
    print("      ↓")
    print("  4️⃣  保存转账历史记录")
    print("      ↓")
    print("  5️⃣  更新统计信息")
    print("      ↓")
    print("  6️⃣  用户通过GUI查看历史和统计")
    
    print("\n关键特性:")
    print("  ✅ 自动重试 - 网络错误等临时问题自动重试")
    print("  ✅ 智能判断 - 账号冻结等永久错误不重试")
    print("  ✅ 完整记录 - 所有转账都有详细记录")
    print("  ✅ 多维统计 - 按发送人、收款人、时间等多维度统计")
    print("  ✅ 可视化界面 - 友好的GUI界面查看和导出")
    
    print("\n✓ 完整集成流程演示完成")


def main():
    """主演示函数"""
    print("\n" + "=" * 70)
    print("多收款人转账可选功能演示")
    print("=" * 70)
    print()
    print("本演示将展示以下功能:")
    print("  1. 转账历史记录")
    print("  2. 转账重试机制")
    print("  3. 转账历史GUI")
    print("  4. 完整集成流程")
    print()
    
    input("按回车键开始演示...")
    
    # 演示1: 转账历史记录
    demo_transfer_history()
    
    input("\n按回车键继续下一个演示...")
    
    # 演示2: 转账重试机制
    demo_transfer_retry()
    
    input("\n按回车键继续下一个演示...")
    
    # 演示3: 转账历史GUI
    demo_transfer_history_gui()
    
    input("\n按回车键继续下一个演示...")
    
    # 演示4: 完整集成流程
    demo_integration()
    
    print("\n" + "=" * 70)
    print("所有演示完成！")
    print("=" * 70)
    print()
    print("感谢使用多收款人转账功能！")
    print()


if __name__ == '__main__':
    main()
