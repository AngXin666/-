"""
检查规则2违反的记录

规则2: checkin_balance_after = balance_before + checkin_reward

运行方式:
    python dev_tools/check_rule2_violations.py
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


def check_rule2_violations():
    """检查规则2违反的记录"""
    
    print("=" * 80)
    print("检查规则2违反的记录")
    print("=" * 80)
    print()
    
    # 初始化数据库
    db = LocalDatabase()
    all_records = db.get_all_history_records()
    
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
    
    # 找出规则2违反的记录
    violations = []
    
    for phone, records in records_by_phone.items():
        for record in records:
            balance_before = record.get('balance_before')
            checkin_reward = record.get('checkin_reward', 0.0) or 0.0
            checkin_balance_after = record.get('checkin_balance_after')
            
            if balance_before is not None and checkin_balance_after is not None:
                expected = balance_before + checkin_reward
                if abs(checkin_balance_after - expected) > 0.01:
                    violations.append({
                        'phone': phone,
                        'date': record.get('run_date'),
                        'balance_before': balance_before,
                        'checkin_reward': checkin_reward,
                        'checkin_balance_after': checkin_balance_after,
                        'expected': expected,
                        'balance_after': record.get('balance_after'),
                        'transfer_amount': record.get('transfer_amount', 0.0) or 0.0
                    })
    
    print(f"发现 {len(violations)} 条违反规则2的记录")
    print()
    
    # 显示所有违反记录
    for idx, v in enumerate(violations, 1):
        print(f"[{idx}] 账号: {v['phone']}, 日期: {v['date']}")
        print(f"  balance_before: {v['balance_before']:.2f}")
        print(f"  checkin_reward: {v['checkin_reward']:.2f}")
        print(f"  checkin_balance_after: {v['checkin_balance_after']:.2f}")
        print(f"  期望值: {v['expected']:.2f}")
        print(f"  差异: {v['checkin_balance_after'] - v['expected']:.2f}")
        balance_after_str = f"{v['balance_after']:.2f}" if v['balance_after'] is not None else 'None'
        print(f"  balance_after: {balance_after_str}")
        print(f"  transfer_amount: {v['transfer_amount']:.2f}")
        print()
    
    # 分析问题
    print("=" * 80)
    print("问题分析:")
    print("=" * 80)
    print()
    
    # 按账号分组统计
    by_phone = {}
    for v in violations:
        phone = v['phone']
        if phone not in by_phone:
            by_phone[phone] = []
        by_phone[phone].append(v)
    
    for phone, cases in by_phone.items():
        print(f"账号 {phone}: {len(cases)} 条违反记录")
        
        # 检查是否是余额异常增长
        for case in cases:
            diff = case['checkin_balance_after'] - case['expected']
            if diff > 100:
                print(f"  [{case['date']}] 余额异常增长: {diff:.2f} 元")
            elif case['checkin_balance_after'] == 0:
                print(f"  [{case['date']}] 签到后余额为0（可能是数据缺失）")
            else:
                print(f"  [{case['date']}] 差异: {diff:.2f} 元")
        print()


if __name__ == "__main__":
    try:
        check_rule2_violations()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
