"""
检查负值奖励的详细信息

检查负值奖励记录的 balance_before 字段是否为 None
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.local_db import LocalDatabase


def check_negative_rewards_detail():
    """检查负值奖励的详细信息"""
    
    print("=" * 80)
    print("检查负值奖励的详细信息")
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
    
    # 检查每条记录
    for i, record in enumerate(negative_rewards[:10], 1):  # 只显示前10条
        phone = record.get('phone')
        run_date = record.get('run_date')
        balance_before = record.get('balance_before')
        balance_after = record.get('balance_after')
        checkin_reward = record.get('checkin_reward', 0.0) or 0.0
        transfer_amount = record.get('transfer_amount', 0.0) or 0.0
        
        print(f"{i}. 账号: {phone}, 日期: {run_date}")
        print(f"   签到前余额: {balance_before} (类型: {type(balance_before).__name__})")
        print(f"   签到后余额: {balance_after} (类型: {type(balance_after).__name__})")
        print(f"   签到奖励: {checkin_reward}")
        print(f"   转账金额: {transfer_amount}")
        
        # 检查是否相等
        if balance_before is not None and balance_after is not None:
            diff = abs(balance_after - balance_before)
            print(f"   余额差异: {diff:.10f}")
            if diff < 0.001:
                print(f"   ✓ 签到失败（余额没变化）")
            else:
                print(f"   ✗ 签到成功（余额有变化）")
        else:
            print(f"   ⚠️ balance_before 为 None")
        
        print()


if __name__ == "__main__":
    try:
        check_negative_rewards_detail()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
