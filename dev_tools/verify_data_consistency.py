"""
验证数据一致性

检查所有记录的余额计算是否正确

运行方式:
    python dev_tools/verify_data_consistency.py
"""

import sys
import os
from pathlib import Path

# 设置标准输出编码为 UTF-8
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.local_db import LocalDatabase


def verify_data_consistency():
    """验证数据一致性"""
    
    print("=" * 80)
    print("验证数据一致性")
    print("=" * 80)
    print()
    
    # 初始化数据库
    db = LocalDatabase()
    all_records = db.get_all_history_records()
    
    print(f"总记录数: {len(all_records)}")
    print()
    
    # 按账号分组
    records_by_phone = {}
    for record in all_records:
        phone = record.get('phone')
        if phone:
            if phone not in records_by_phone:
                records_by_phone[phone] = []
            records_by_phone[phone].append(record)
    
    # 对每个账号的记录按日期排序
    for phone in records_by_phone:
        records_by_phone[phone].sort(key=lambda r: r.get('run_date', ''))
    
    print(f"共 {len(records_by_phone)} 个账号")
    print()
    
    # 验证规则
    print("=" * 80)
    print("验证规则:")
    print("=" * 80)
    print("1. balance_before = 前一天的 balance_after（第一条记录除外）")
    print("2. checkin_balance_after = balance_before + checkin_reward")
    print("3. balance_after = checkin_balance_after - transfer_amount")
    print()
    
    # 统计
    rule1_violations = []
    rule2_violations = []
    rule3_violations = []
    
    for phone, records in records_by_phone.items():
        previous_balance_after = None
        
        for idx, record in enumerate(records):
            record_id = record.get('id')
            run_date = record.get('run_date')
            balance_before = record.get('balance_before')
            checkin_reward = record.get('checkin_reward', 0.0) or 0.0
            checkin_balance_after = record.get('checkin_balance_after')
            transfer_amount = record.get('transfer_amount', 0.0) or 0.0
            balance_after = record.get('balance_after')
            
            # 规则1: balance_before = 前一天的 balance_after
            if idx > 0 and previous_balance_after is not None and balance_before is not None:
                if abs(balance_before - previous_balance_after) > 0.01:
                    rule1_violations.append({
                        'phone': phone,
                        'date': run_date,
                        'balance_before': balance_before,
                        'previous_balance_after': previous_balance_after
                    })
            
            # 规则2: checkin_balance_after = balance_before + checkin_reward
            if balance_before is not None and checkin_balance_after is not None:
                expected_checkin_balance = balance_before + checkin_reward
                if abs(checkin_balance_after - expected_checkin_balance) > 0.01:
                    rule2_violations.append({
                        'phone': phone,
                        'date': run_date,
                        'balance_before': balance_before,
                        'checkin_reward': checkin_reward,
                        'checkin_balance_after': checkin_balance_after,
                        'expected': expected_checkin_balance
                    })
            
            # 规则3: balance_after = checkin_balance_after - transfer_amount
            if checkin_balance_after is not None and balance_after is not None:
                expected_balance_after = checkin_balance_after - transfer_amount
                if abs(balance_after - expected_balance_after) > 0.01:
                    rule3_violations.append({
                        'phone': phone,
                        'date': run_date,
                        'checkin_balance_after': checkin_balance_after,
                        'transfer_amount': transfer_amount,
                        'balance_after': balance_after,
                        'expected': expected_balance_after
                    })
            
            previous_balance_after = balance_after
    
    # 输出结果
    print("=" * 80)
    print("验证结果:")
    print("=" * 80)
    print()
    
    print(f"规则1违反: {len(rule1_violations)} 条")
    if rule1_violations:
        print("  (balance_before != 前一天的 balance_after)")
        for case in rule1_violations[:5]:
            print(f"    [{case['phone']}] [{case['date']}]")
            print(f"      balance_before: {case['balance_before']:.2f}")
            print(f"      前一天 balance_after: {case['previous_balance_after']:.2f}")
    print()
    
    print(f"规则2违反: {len(rule2_violations)} 条")
    if rule2_violations:
        print("  (checkin_balance_after != balance_before + checkin_reward)")
        for case in rule2_violations[:5]:
            print(f"    [{case['phone']}] [{case['date']}]")
            print(f"      balance_before: {case['balance_before']:.2f}")
            print(f"      checkin_reward: {case['checkin_reward']:.2f}")
            print(f"      checkin_balance_after: {case['checkin_balance_after']:.2f}")
            print(f"      期望值: {case['expected']:.2f}")
    print()
    
    print(f"规则3违反: {len(rule3_violations)} 条")
    if rule3_violations:
        print("  (balance_after != checkin_balance_after - transfer_amount)")
        for case in rule3_violations[:5]:
            print(f"    [{case['phone']}] [{case['date']}]")
            print(f"      checkin_balance_after: {case['checkin_balance_after']:.2f}")
            print(f"      transfer_amount: {case['transfer_amount']:.2f}")
            print(f"      balance_after: {case['balance_after']:.2f}")
            print(f"      期望值: {case['expected']:.2f}")
    print()
    
    # 总结
    print("=" * 80)
    print("总结:")
    print("=" * 80)
    
    if not rule1_violations and not rule2_violations and not rule3_violations:
        print("✓ 所有数据都符合一致性规则")
    else:
        print("⚠️ 发现数据不一致")
        print(f"  规则1违反: {len(rule1_violations)} 条")
        print(f"  规则2违反: {len(rule2_violations)} 条")
        print(f"  规则3违反: {len(rule3_violations)} 条")
    
    print()


if __name__ == "__main__":
    try:
        verify_data_consistency()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
