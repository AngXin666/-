"""
检查签到次数为0的记录

检查余额前后相同且签到次数为0的记录,看看这些账号整个历史中签到次数是否都是0

运行方式:
    python dev_tools/check_zero_checkin_times.py
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


def check_zero_checkin_times():
    """检查签到次数为0的记录"""
    
    print("=" * 80)
    print("检查签到次数为0的记录")
    print("=" * 80)
    print()
    
    # 初始化数据库
    db = LocalDatabase()
    all_records = db.get_all_history_records()
    
    # 找出余额前后相同且签到奖励为0的记录
    zero_reward_same_balance = []
    
    for record in all_records:
        checkin_reward = record.get('checkin_reward', 0.0) or 0.0
        balance_before = record.get('balance_before')
        checkin_balance_after = record.get('checkin_balance_after')
        
        if checkin_reward == 0 and balance_before is not None and checkin_balance_after is not None:
            if abs(balance_before - checkin_balance_after) < 0.01:
                zero_reward_same_balance.append(record)
    
    # 找出签到次数为0的记录
    zero_checkin_times_records = []
    for record in zero_reward_same_balance:
        checkin_total_times = record.get('checkin_total_times', 0) or 0
        if checkin_total_times == 0:
            zero_checkin_times_records.append(record)
    
    print(f"余额前后相同且签到奖励为0: {len(zero_reward_same_balance)} 条")
    print(f"其中签到次数为0: {len(zero_checkin_times_records)} 条")
    print()
    
    # 按账号分组
    records_by_phone = {}
    for record in zero_checkin_times_records:
        phone = record.get('phone')
        if phone not in records_by_phone:
            records_by_phone[phone] = []
        records_by_phone[phone].append(record)
    
    print(f"涉及账号数: {len(records_by_phone)}")
    print()
    
    # 检查这些账号的所有记录
    print("=" * 80)
    print("检查这些账号的所有历史记录:")
    print("=" * 80)
    print()
    
    all_zero_count = 0  # 所有记录签到次数都是0
    has_nonzero_count = 0  # 有记录签到次数不是0
    
    for phone in sorted(records_by_phone.keys()):
        # 获取该账号的所有记录
        phone_all_records = [r for r in all_records if r.get('phone') == phone]
        
        # 检查签到次数
        all_zero = True
        max_checkin_times = 0
        
        for record in phone_all_records:
            checkin_total_times = record.get('checkin_total_times', 0) or 0
            if checkin_total_times > 0:
                all_zero = False
                max_checkin_times = max(max_checkin_times, checkin_total_times)
        
        if all_zero:
            all_zero_count += 1
            print(f"账号 {phone}: 所有 {len(phone_all_records)} 条记录签到次数都是0")
        else:
            has_nonzero_count += 1
            print(f"账号 {phone}: 有 {len(phone_all_records)} 条记录, 最大签到次数: {max_checkin_times}")
    
    print()
    print("=" * 80)
    print("统计结果:")
    print("=" * 80)
    print(f"所有记录签到次数都是0的账号: {all_zero_count} 个")
    print(f"有记录签到次数不是0的账号: {has_nonzero_count} 个")


if __name__ == "__main__":
    try:
        check_zero_checkin_times()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
