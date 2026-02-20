"""
详细检查0元奖励记录

查看这些记录的余额变化情况
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.local_db import LocalDatabase


def check_zero_rewards_detail():
    """详细检查0元奖励记录"""
    
    print("=" * 80)
    print("详细检查0元奖励记录")
    print("=" * 80)
    print()
    
    # 初始化数据库
    db = LocalDatabase()
    
    # 获取所有记录
    all_records = db.get_all_history_records()
    
    # 找出0元奖励
    zero_rewards = [r for r in all_records if (r.get('checkin_reward', 0.0) or 0.0) == 0]
    
    print(f"共找到 {len(zero_rewards)} 条0元奖励记录")
    print()
    
    # 分类统计
    balance_unchanged = 0  # 余额没变化（签到失败）
    balance_changed = 0    # 余额有变化（但奖励是0）
    no_balance_data = 0    # 没有余额数据
    
    # 抽样显示余额有变化的记录
    changed_samples = []
    
    for record in zero_rewards:
        balance_before = record.get('balance_before')
        balance_after = record.get('balance_after')
        
        if balance_after is None:
            no_balance_data += 1
        elif balance_before is not None and abs(balance_after - balance_before) < 0.001:
            balance_unchanged += 1
        else:
            balance_changed += 1
            if len(changed_samples) < 20:
                changed_samples.append(record)
    
    print("分类统计:")
    print(f"  余额没变化（签到失败）: {balance_unchanged} 条")
    print(f"  余额有变化（但奖励是0）: {balance_changed} 条 ⚠️")
    print(f"  没有余额数据: {no_balance_data} 条")
    print()
    
    if changed_samples:
        print("=" * 80)
        print("余额有变化但奖励是0的记录（抽样）")
        print("=" * 80)
        for record in changed_samples:
            phone = record.get('phone')
            run_date = record.get('run_date')
            balance_before = record.get('balance_before')
            balance_after = record.get('balance_after')
            
            print(f"[{phone}] {run_date}")
            print(f"  签到前余额: {balance_before}")
            print(f"  签到后余额: {balance_after}")
            if balance_before is not None:
                print(f"  余额变化: {balance_after - balance_before:.2f} 元")
            print()


if __name__ == "__main__":
    try:
        check_zero_rewards_detail()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
