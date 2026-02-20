"""
检查数据库中的异常签到奖励值

检查项：
1. 负值签到奖励（应该都是正值或0）
2. 异常大的签到奖励（超过合理范围）
3. 异常小的签到奖励（接近0但不为0）

运行方式：
    python dev_tools/check_abnormal_checkin_rewards.py
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.local_db import LocalDatabase


def check_abnormal_rewards():
    """检查异常的签到奖励值"""
    
    print("=" * 80)
    print("检查数据库中的异常签到奖励值")
    print("=" * 80)
    print()
    
    # 初始化数据库
    db = LocalDatabase()
    
    # 获取所有记录
    print("正在读取数据库记录...")
    all_records = db.get_all_history_records()
    
    if not all_records:
        print("❌ 数据库中没有记录")
        return
    
    print(f"✓ 共找到 {len(all_records)} 条记录")
    print()
    
    # 统计信息
    negative_rewards = []  # 负值奖励
    large_rewards = []     # 异常大的奖励（>50元）
    small_rewards = []     # 异常小的奖励（0.01-0.1元）
    zero_rewards = []      # 0元奖励
    normal_rewards = []    # 正常奖励
    
    # 遍历所有记录
    for record in all_records:
        phone = record.get('phone')
        run_date = record.get('run_date')
        checkin_reward = record.get('checkin_reward', 0.0) or 0.0
        balance_before = record.get('balance_before')
        balance_after = record.get('balance_after')
        transfer_amount = record.get('transfer_amount', 0.0) or 0.0
        
        # 跳过没有余额数据的记录
        if balance_after is None:
            continue
        
        # 分类
        if checkin_reward < 0:
            negative_rewards.append({
                'phone': phone,
                'date': run_date,
                'reward': checkin_reward,
                'balance_before': balance_before,
                'balance_after': balance_after,
                'transfer_amount': transfer_amount
            })
        elif checkin_reward > 50:
            large_rewards.append({
                'phone': phone,
                'date': run_date,
                'reward': checkin_reward,
                'balance_before': balance_before,
                'balance_after': balance_after,
                'transfer_amount': transfer_amount
            })
        elif 0 < checkin_reward < 0.1:
            small_rewards.append({
                'phone': phone,
                'date': run_date,
                'reward': checkin_reward,
                'balance_before': balance_before,
                'balance_after': balance_after,
                'transfer_amount': transfer_amount
            })
        elif checkin_reward == 0:
            zero_rewards.append({
                'phone': phone,
                'date': run_date,
                'reward': checkin_reward,
                'balance_before': balance_before,
                'balance_after': balance_after,
                'transfer_amount': transfer_amount
            })
        else:
            normal_rewards.append({
                'phone': phone,
                'date': run_date,
                'reward': checkin_reward
            })
    
    # 输出统计结果
    print("=" * 80)
    print("统计结果")
    print("=" * 80)
    print(f"总记录数: {len(all_records)}")
    print(f"正常奖励 (0.1-50元): {len(normal_rewards)} 条")
    print(f"0元奖励: {len(zero_rewards)} 条")
    print(f"异常小奖励 (0.01-0.1元): {len(small_rewards)} 条")
    print(f"异常大奖励 (>50元): {len(large_rewards)} 条")
    print(f"负值奖励 (<0元): {len(negative_rewards)} 条")
    print()
    
    # 显示负值奖励详情
    if negative_rewards:
        print("=" * 80)
        print("⚠️ 负值奖励详情（异常）")
        print("=" * 80)
        for item in negative_rewards[:20]:  # 只显示前20条
            print(f"账号: {item['phone']}")
            print(f"  日期: {item['date']}")
            print(f"  签到奖励: {item['reward']:.2f} 元")
            if item['balance_before'] is not None:
                print(f"  签到前余额: {item['balance_before']:.2f} 元")
            print(f"  签到后余额: {item['balance_after']:.2f} 元")
            if item['transfer_amount'] > 0:
                print(f"  转账金额: {item['transfer_amount']:.2f} 元")
            print()
        
        if len(negative_rewards) > 20:
            print(f"... 还有 {len(negative_rewards) - 20} 条负值奖励记录")
            print()
    
    # 显示异常大奖励详情
    if large_rewards:
        print("=" * 80)
        print("⚠️ 异常大奖励详情（>50元）")
        print("=" * 80)
        for item in large_rewards[:20]:  # 只显示前20条
            print(f"账号: {item['phone']}")
            print(f"  日期: {item['date']}")
            print(f"  签到奖励: {item['reward']:.2f} 元")
            if item['balance_before'] is not None:
                print(f"  签到前余额: {item['balance_before']:.2f} 元")
            print(f"  签到后余额: {item['balance_after']:.2f} 元")
            if item['transfer_amount'] > 0:
                print(f"  转账金额: {item['transfer_amount']:.2f} 元")
            print()
        
        if len(large_rewards) > 20:
            print(f"... 还有 {len(large_rewards) - 20} 条异常大奖励记录")
            print()
    
    # 显示异常小奖励详情（抽样显示）
    if small_rewards:
        print("=" * 80)
        print("ℹ️ 异常小奖励详情（0.01-0.1元，抽样显示）")
        print("=" * 80)
        for item in small_rewards[:10]:  # 只显示前10条
            print(f"账号: {item['phone']}")
            print(f"  日期: {item['date']}")
            print(f"  签到奖励: {item['reward']:.2f} 元")
            if item['balance_before'] is not None:
                print(f"  签到前余额: {item['balance_before']:.2f} 元")
            print(f"  签到后余额: {item['balance_after']:.2f} 元")
            if item['transfer_amount'] > 0:
                print(f"  转账金额: {item['transfer_amount']:.2f} 元")
            print()
        
        if len(small_rewards) > 10:
            print(f"... 还有 {len(small_rewards) - 10} 条异常小奖励记录")
            print()
    
    # 显示0元奖励详情（抽样显示）
    if zero_rewards:
        print("=" * 80)
        print("ℹ️ 0元奖励详情（抽样显示）")
        print("=" * 80)
        for item in zero_rewards[:10]:  # 只显示前10条
            print(f"账号: {item['phone']}")
            print(f"  日期: {item['date']}")
            print(f"  签到奖励: {item['reward']:.2f} 元")
            if item['balance_before'] is not None:
                print(f"  签到前余额: {item['balance_before']:.2f} 元")
            print(f"  签到后余额: {item['balance_after']:.2f} 元")
            if item['transfer_amount'] > 0:
                print(f"  转账金额: {item['transfer_amount']:.2f} 元")
            print()
        
        if len(zero_rewards) > 10:
            print(f"... 还有 {len(zero_rewards) - 10} 条0元奖励记录")
            print()
    
    # 总结
    print("=" * 80)
    print("检查完成")
    print("=" * 80)
    
    if negative_rewards or large_rewards:
        print("⚠️ 发现异常值，需要进一步检查")
    else:
        print("✅ 未发现明显异常值")
    
    print()


if __name__ == "__main__":
    try:
        check_abnormal_rewards()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
