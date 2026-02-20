"""
检查转账修复对第二天数据的影响

检查修复后的转账记录，看第二天的签到奖励计算是否正确

运行方式:
    python dev_tools/check_next_day_impact.py
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# 设置标准输出编码为 UTF-8
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.local_db import LocalDatabase


def check_next_day_impact():
    """检查转账修复对第二天数据的影响"""
    
    print("=" * 80)
    print("检查转账修复对第二天数据的影响")
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
    
    # 检查有问题的记录
    issues = []
    
    for phone, records in records_by_phone.items():
        for idx in range(len(records) - 1):
            current = records[idx]
            next_day = records[idx + 1]
            
            # 获取当前记录的数据
            current_date = current.get('run_date')
            current_balance_after = current.get('balance_after', 0.0) or 0.0
            current_checkin_balance_after = current.get('checkin_balance_after')
            current_transfer = current.get('transfer_amount', 0.0) or 0.0
            
            # 获取第二天的数据
            next_date = next_day.get('run_date')
            next_checkin_reward = next_day.get('checkin_reward', 0.0) or 0.0
            next_checkin_balance_after = next_day.get('checkin_balance_after')
            
            # 检查：第二天的签到奖励应该 = 第二天签到后余额 - 前一天最终余额
            if next_checkin_balance_after is not None:
                expected_reward = next_checkin_balance_after - current_balance_after
                
                # 如果差异超过0.01元，记录问题
                if abs(expected_reward - next_checkin_reward) > 0.01:
                    issues.append({
                        'phone': phone,
                        'current_date': current_date,
                        'next_date': next_date,
                        'current_balance_after': current_balance_after,
                        'current_checkin_balance_after': current_checkin_balance_after,
                        'current_transfer': current_transfer,
                        'next_checkin_reward': next_checkin_reward,
                        'next_checkin_balance_after': next_checkin_balance_after,
                        'expected_reward': expected_reward,
                        'difference': expected_reward - next_checkin_reward
                    })
    
    print(f"找到 {len(issues)} 条有问题的记录")
    print()
    
    if not issues:
        print("✓ 没有发现问题，第二天的数据都是正确的")
        return
    
    # 显示详情
    print("=" * 80)
    print("问题记录详情（前20条）:")
    print("=" * 80)
    
    for item in issues[:20]:
        print(f"\n账号: {item['phone']}")
        print(f"  当前日期: {item['current_date']}")
        print(f"    签到后余额: {item['current_checkin_balance_after']:.2f}" if item['current_checkin_balance_after'] is not None else "    签到后余额: None")
        print(f"    转账金额: {item['current_transfer']:.2f}")
        print(f"    最终余额: {item['current_balance_after']:.2f}")
        print(f"  第二天日期: {item['next_date']}")
        print(f"    当前签到奖励: {item['next_checkin_reward']:.2f}")
        print(f"    应该是: {item['expected_reward']:.2f}")
        print(f"    签到后余额: {item['next_checkin_balance_after']:.2f}")
        print(f"    差异: {item['difference']:.2f}")
    
    if len(issues) > 20:
        print(f"\n... 还有 {len(issues) - 20} 条记录")
    
    # 总结
    print()
    print("=" * 80)
    print("总结:")
    print("=" * 80)
    print(f"有问题的记录: {len(issues)}")
    print()
    print("这些记录的第二天签到奖励需要重新计算")
    print("建议: 运行签到奖励重新计算脚本")


if __name__ == "__main__":
    try:
        check_next_day_impact()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
