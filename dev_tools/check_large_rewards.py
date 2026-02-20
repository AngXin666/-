"""
检查签到奖励大于10的记录

签到奖励不可能大于10元，所有大于10的都是错误数据
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.local_db import LocalDatabase


def check_large_rewards():
    """检查签到奖励大于10的记录"""
    
    print("=" * 80)
    print("检查签到奖励大于10的记录")
    print("=" * 80)
    print()
    
    # 初始化数据库
    db = LocalDatabase()
    
    # 获取所有记录
    all_records = db.get_all_history_records()
    
    # 找出大于10的奖励
    large_rewards = []
    for record in all_records:
        checkin_reward = record.get('checkin_reward', 0.0) or 0.0
        if checkin_reward > 10:
            large_rewards.append(record)
    
    print(f"共找到 {len(large_rewards)} 条签到奖励大于10的记录")
    print()
    
    if not large_rewards:
        print("✅ 没有找到签到奖励大于10的记录")
        return
    
    # 按账号分组
    by_phone = {}
    for record in large_rewards:
        phone = record.get('phone')
        if phone not in by_phone:
            by_phone[phone] = []
        by_phone[phone].append(record)
    
    print(f"涉及 {len(by_phone)} 个账号")
    print()
    
    # 显示每个账号的详情
    for phone, records in by_phone.items():
        print(f"账号: {phone} ({len(records)} 条记录)")
        print("-" * 80)
        
        for record in records:
            record_id = record.get('id')
            run_date = record.get('run_date')
            balance_before = record.get('balance_before')
            balance_after = record.get('balance_after')
            checkin_reward = record.get('checkin_reward', 0.0) or 0.0
            transfer_amount = record.get('transfer_amount', 0.0) or 0.0
            
            print(f"  [{run_date}] ID: {record_id}")
            print(f"    签到前余额: {balance_before}")
            print(f"    签到后余额: {balance_after}")
            print(f"    签到奖励: {checkin_reward:.2f} 元 ⚠️")
            if transfer_amount > 0:
                print(f"    转账金额: {transfer_amount:.2f} 元")
            
            # 判断余额是否变化
            if balance_before is not None and balance_after is not None:
                diff = abs(balance_after - balance_before)
                if diff < 0.001:
                    print(f"    ✓ 签到失败（余额没变化）-> 应该是 0 元")
                else:
                    print(f"    ✓ 签到成功（余额变化: {balance_after - balance_before:.2f} 元）")
            
            print()
        
        print()


if __name__ == "__main__":
    try:
        check_large_rewards()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
