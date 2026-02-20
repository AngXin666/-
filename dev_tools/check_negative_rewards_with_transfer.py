"""
检查负值奖励记录的转账情况
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.local_db import LocalDatabase


def check_negative_rewards():
    """检查负值奖励记录"""
    
    print("=" * 80)
    print("检查负值奖励记录的转账情况")
    print("=" * 80)
    print()
    
    # 初始化数据库
    db = LocalDatabase()
    
    # 获取所有记录
    all_records = db.get_all_history_records()
    
    # 找出负值奖励
    negative_rewards = []
    for record in all_records:
        checkin_reward = record.get('checkin_reward', 0.0) or 0.0
        if checkin_reward < 0:
            negative_rewards.append(record)
    
    print(f"共找到 {len(negative_rewards)} 条负值奖励记录")
    print()
    
    for record in negative_rewards:
        record_id = record.get('id')
        phone = record.get('phone')
        run_date = record.get('run_date')
        balance_before = record.get('balance_before')
        balance_after = record.get('balance_after')
        checkin_reward = record.get('checkin_reward', 0.0) or 0.0
        transfer_amount = record.get('transfer_amount', 0.0) or 0.0
        
        print(f"账号: {phone}, 日期: {run_date}, ID: {record_id}")
        print(f"  签到前余额: {balance_before}")
        print(f"  签到后余额: {balance_after}")
        print(f"  签到奖励: {checkin_reward:.2f} 元")
        print(f"  转账金额: {transfer_amount:.2f} 元")
        
        # 分析
        if balance_before is not None and balance_after == 0:
            if transfer_amount > 0:
                # 有转账记录
                # 推算：签到后余额（转账前）= 转账金额
                # 签到奖励 = 转账金额 - 签到前余额
                expected_reward = transfer_amount - balance_before
                print(f"  推算签到奖励: {expected_reward:.2f} 元")
                
                if expected_reward > 10:
                    print(f"  ⚠️ 推算奖励大于10，无法准确计算 -> 应设为 0")
                elif expected_reward < 0:
                    print(f"  ⚠️ 推算奖励为负值，可能是签到失败后转账 -> 应设为 0")
                else:
                    print(f"  ✓ 推算奖励合理")
            else:
                # 没有转账记录，但余额变为0
                print(f"  ⚠️ 没有转账记录，但余额变为0 -> 数据异常，应设为 0")
        
        print()


if __name__ == "__main__":
    try:
        check_negative_rewards()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
