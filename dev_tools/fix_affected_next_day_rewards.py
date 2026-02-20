"""
修复受转账修复影响的第二天签到奖励

直接修复那137条受影响的记录

运行方式:
    python dev_tools/fix_affected_next_day_rewards.py
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


def fix_affected_next_day_rewards():
    """修复受转账修复影响的第二天签到奖励"""
    
    print("=" * 80)
    print("修复受转账修复影响的第二天签到奖励")
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
    
    # 找到需要修复的记录
    need_fix = []
    
    for phone, records in records_by_phone.items():
        for idx in range(len(records) - 1):
            current = records[idx]
            next_day = records[idx + 1]
            
            # 获取当前记录的数据
            current_balance_after = current.get('balance_after', 0.0) or 0.0
            
            # 获取第二天的数据
            next_id = next_day.get('id')
            next_date = next_day.get('run_date')
            next_checkin_reward = next_day.get('checkin_reward', 0.0) or 0.0
            next_checkin_balance_after = next_day.get('checkin_balance_after')
            
            # 检查：第二天的签到奖励应该 = 第二天签到后余额 - 前一天最终余额
            if next_checkin_balance_after is not None:
                expected_reward = next_checkin_balance_after - current_balance_after
                
                # 如果差异超过0.01元，需要修复
                if abs(expected_reward - next_checkin_reward) > 0.01:
                    need_fix.append({
                        'id': next_id,
                        'phone': phone,
                        'date': next_date,
                        'current_reward': next_checkin_reward,
                        'correct_reward': expected_reward
                    })
    
    print(f"找到 {len(need_fix)} 条需要修复的记录")
    print()
    
    if not need_fix:
        print("✓ 没有需要修复的记录")
        return
    
    # 修复记录
    fixed_count = 0
    
    for item in need_fix:
        try:
            db.update_checkin_reward(item['id'], item['correct_reward'])
            fixed_count += 1
            print(f"[{item['phone']}] [{item['date']}] {item['current_reward']:.2f} → {item['correct_reward']:.2f}")
        except Exception as e:
            print(f"❌ [{item['phone']}] [{item['date']}] 修复失败: {e}")
    
    # 总结
    print()
    print("=" * 80)
    print("修复完成")
    print("=" * 80)
    print(f"需要修复: {len(need_fix)}")
    print(f"已修复: {fixed_count}")
    print()


if __name__ == "__main__":
    try:
        fix_affected_next_day_rewards()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
