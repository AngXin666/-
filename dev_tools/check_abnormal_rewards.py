"""
检查异常的签到奖励

检查签到奖励 > 10 或 <= 1 的记录

运行方式:
    python dev_tools/check_abnormal_rewards.py
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


def check_abnormal_rewards():
    """检查异常的签到奖励"""
    
    print("=" * 80)
    print("检查异常的签到奖励")
    print("=" * 80)
    print()
    
    # 初始化数据库
    db = LocalDatabase()
    all_records = db.get_all_history_records()
    
    # 分类统计
    large_rewards = []  # > 10
    small_rewards = []  # <= 1 (不包括0)
    zero_rewards = []   # = 0
    
    for record in all_records:
        checkin_reward = record.get('checkin_reward', 0.0) or 0.0
        
        if checkin_reward > 10:
            large_rewards.append(record)
        elif 0 < checkin_reward <= 1:
            small_rewards.append(record)
        elif checkin_reward == 0:
            zero_rewards.append(record)
    
    # 输出统计
    print(f"总记录数: {len(all_records)}")
    print(f"签到奖励 > 10元: {len(large_rewards)} 条")
    print(f"签到奖励 <= 1元 (不含0): {len(small_rewards)} 条")
    print(f"签到奖励 = 0元: {len(zero_rewards)} 条")
    print()
    
    # 显示大于10的记录
    if large_rewards:
        print("=" * 80)
        print(f"签到奖励 > 10元 的记录 (共{len(large_rewards)}条):")
        print("=" * 80)
        print()
        
        for record in large_rewards:
            phone = record.get('phone')
            run_date = record.get('run_date')
            checkin_reward = record.get('checkin_reward', 0.0) or 0.0
            balance_before = record.get('balance_before')
            balance_after = record.get('balance_after')
            checkin_balance_after = record.get('checkin_balance_after')
            
            print(f"账号: {phone}, 日期: {run_date}")
            print(f"  签到奖励: {checkin_reward:.2f} 元")
            print(f"  余额前: {balance_before:.2f if balance_before is not None else 'None'}")
            print(f"  签到后余额: {checkin_balance_after:.2f if checkin_balance_after is not None else 'None'}")
            print(f"  最终余额: {balance_after:.2f if balance_after is not None else 'None'}")
            print()
    
    # 显示小于等于1的记录
    if small_rewards:
        print("=" * 80)
        print(f"签到奖励 <= 1元 的记录 (共{len(small_rewards)}条):")
        print("=" * 80)
        print()
        
        for record in small_rewards:
            phone = record.get('phone')
            run_date = record.get('run_date')
            checkin_reward = record.get('checkin_reward', 0.0) or 0.0
            balance_before = record.get('balance_before')
            balance_after = record.get('balance_after')
            checkin_balance_after = record.get('checkin_balance_after')
            
            print(f"账号: {phone}, 日期: {run_date}")
            print(f"  签到奖励: {checkin_reward:.2f} 元")
            print(f"  余额前: {balance_before:.2f if balance_before is not None else 'None'}")
            print(f"  签到后余额: {checkin_balance_after:.2f if checkin_balance_after is not None else 'None'}")
            print(f"  最终余额: {balance_after:.2f if balance_after is not None else 'None'}")
            print()
    
    # 显示等于0的记录(分类统计)
    if zero_rewards:
        print("=" * 80)
        print(f"签到奖励 = 0元 的记录分析 (共{len(zero_rewards)}条):")
        print("=" * 80)
        print()
        
        # 分类统计
        no_balance_after = []  # 没有最终余额
        no_checkin_balance = []  # 没有签到后余额
        balance_same = []  # 余额前后相同
        other = []  # 其他情况
        
        for record in zero_rewards:
            balance_before = record.get('balance_before')
            balance_after = record.get('balance_after')
            checkin_balance_after = record.get('checkin_balance_after')
            
            if balance_after is None:
                no_balance_after.append(record)
            elif checkin_balance_after is None:
                no_checkin_balance.append(record)
            elif balance_before is not None and checkin_balance_after is not None:
                if abs(balance_before - checkin_balance_after) < 0.01:
                    balance_same.append(record)
                else:
                    other.append(record)
            else:
                other.append(record)
        
        print(f"分类统计:")
        print(f"  没有最终余额: {len(no_balance_after)} 条")
        print(f"  没有签到后余额: {len(no_checkin_balance)} 条")
        print(f"  余额前后相同(未签到): {len(balance_same)} 条")
        print(f"  其他情况: {len(other)} 条")
        print()
        
        # 显示余额前后相同的记录(前10条)
        if balance_same:
            print("-" * 80)
            print(f"余额前后相同的记录 (共{len(balance_same)}条,显示前10条):")
            print("-" * 80)
            print()
            
            # 统计签到总次数
            checkin_times_zero = 0
            checkin_times_not_zero = 0
            
            for record in balance_same:
                checkin_total_times = record.get('checkin_total_times', 0) or 0
                if checkin_total_times == 0:
                    checkin_times_zero += 1
                else:
                    checkin_times_not_zero += 1
            
            print(f"签到总次数统计:")
            print(f"  签到总次数 = 0: {checkin_times_zero} 条")
            print(f"  签到总次数 > 0: {checkin_times_not_zero} 条")
            print()
            
            for record in balance_same[:10]:
                phone = record.get('phone')
                run_date = record.get('run_date')
                balance_before = record.get('balance_before')
                checkin_balance_after = record.get('checkin_balance_after')
                balance_after = record.get('balance_after')
                checkin_total_times = record.get('checkin_total_times', 0) or 0
                
                print(f"账号: {phone}, 日期: {run_date}")
                print(f"  余额前: {balance_before:.2f}")
                print(f"  签到后余额: {checkin_balance_after:.2f}")
                print(f"  最终余额: {balance_after:.2f}")
                print(f"  签到总次数: {checkin_total_times}")
                print()
        
        # 显示其他情况(前10条)
        if other:
            print("-" * 80)
            print(f"其他情况 (共{len(other)}条,显示前10条):")
            print("-" * 80)
            print()
            
            for record in other[:10]:
                phone = record.get('phone')
                run_date = record.get('run_date')
                balance_before = record.get('balance_before')
                checkin_balance_after = record.get('checkin_balance_after')
                balance_after = record.get('balance_after')
                
                bal_before = f"{balance_before:.2f}" if balance_before is not None else 'None'
                checkin_bal = f"{checkin_balance_after:.2f}" if checkin_balance_after is not None else 'None'
                bal_after = f"{balance_after:.2f}" if balance_after is not None else 'None'
                
                print(f"账号: {phone}, 日期: {run_date}")
                print(f"  余额前: {bal_before}")
                print(f"  签到后余额: {checkin_bal}")
                print(f"  最终余额: {bal_after}")
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
