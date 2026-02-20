"""
详细分析签到奖励为0的记录

分析333条签到奖励为0的记录,特别关注余额前后相同但签到总次数>0的异常情况

运行方式:
    python dev_tools/analyze_zero_rewards.py
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


def analyze_zero_rewards():
    """详细分析签到奖励为0的记录"""
    
    print("=" * 80)
    print("详细分析签到奖励为0的记录")
    print("=" * 80)
    print()
    
    # 初始化数据库
    db = LocalDatabase()
    all_records = db.get_all_history_records()
    
    # 找出签到奖励为0的记录
    zero_rewards = [r for r in all_records if (r.get('checkin_reward', 0.0) or 0.0) == 0]
    
    print(f"总记录数: {len(all_records)}")
    print(f"签到奖励 = 0元: {len(zero_rewards)} 条")
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
    
    print("=" * 80)
    print("分类统计:")
    print("=" * 80)
    print(f"  没有最终余额: {len(no_balance_after)} 条")
    print(f"  没有签到后余额: {len(no_checkin_balance)} 条")
    print(f"  余额前后相同(未签到): {len(balance_same)} 条")
    print(f"  其他情况: {len(other)} 条")
    print()
    
    # 详细分析"余额前后相同"的记录
    if balance_same:
        print("=" * 80)
        print(f"余额前后相同的记录详细分析 (共{len(balance_same)}条):")
        print("=" * 80)
        print()
        
        # 按签到总次数分类
        checkin_times_zero = []
        checkin_times_not_zero = []
        
        for record in balance_same:
            checkin_total_times = record.get('checkin_total_times', 0) or 0
            if checkin_total_times == 0:
                checkin_times_zero.append(record)
            else:
                checkin_times_not_zero.append(record)
        
        print(f"签到总次数统计:")
        print(f"  签到总次数 = 0: {len(checkin_times_zero)} 条")
        print(f"  签到总次数 > 0: {len(checkin_times_not_zero)} 条")
        print()
        
        # 分析签到总次数=0的记录
        if checkin_times_zero:
            print("-" * 80)
            print(f"签到总次数 = 0 的记录 (共{len(checkin_times_zero)}条):")
            print("-" * 80)
            print()
            
            # 按账号分组
            records_by_phone = {}
            for record in checkin_times_zero:
                phone = record.get('phone')
                if phone not in records_by_phone:
                    records_by_phone[phone] = []
                records_by_phone[phone].append(record)
            
            print(f"涉及账号数: {len(records_by_phone)}")
            print()
            
            # 检查这些账号的所有记录
            all_zero_accounts = []  # 所有记录签到次数都是0
            has_nonzero_accounts = []  # 有记录签到次数不是0
            
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
                    all_zero_accounts.append(phone)
                else:
                    has_nonzero_accounts.append(phone)
            
            print(f"  所有历史记录签到次数都是0: {len(all_zero_accounts)} 个账号")
            if all_zero_accounts:
                for phone in all_zero_accounts:
                    print(f"    - {phone}")
            print()
            
            print(f"  有历史记录签到次数 > 0: {len(has_nonzero_accounts)} 个账号")
            if has_nonzero_accounts and len(has_nonzero_accounts) <= 10:
                for phone in has_nonzero_accounts:
                    print(f"    - {phone}")
            elif has_nonzero_accounts:
                print(f"    (账号数量较多,不全部显示)")
            print()
        
        # 分析签到总次数>0的记录 - 这些是有问题的!
        if checkin_times_not_zero:
            print("-" * 80)
            print(f"⚠️ 签到总次数 > 0 但余额前后相同的记录 (共{len(checkin_times_not_zero)}条):")
            print("-" * 80)
            print("这些记录有问题:签到了但余额没变化!")
            print()
            
            # 统计转账情况
            has_transfer = []
            no_transfer = []
            
            for record in checkin_times_not_zero:
                transfer_amount = record.get('transfer_amount', 0.0) or 0.0
                if transfer_amount > 0:
                    has_transfer.append(record)
                else:
                    no_transfer.append(record)
            
            print(f"转账情况统计:")
            print(f"  有转账记录: {len(has_transfer)} 条")
            print(f"  无转账记录: {len(no_transfer)} 条")
            print()
            
            # 显示有转账的记录(前10条)
            if has_transfer:
                print(f"有转账的记录 (共{len(has_transfer)}条,显示前10条):")
                print()
                
                for record in has_transfer[:10]:
                    phone = record.get('phone')
                    run_date = record.get('run_date')
                    balance_before = record.get('balance_before')
                    checkin_balance_after = record.get('checkin_balance_after')
                    balance_after = record.get('balance_after')
                    checkin_total_times = record.get('checkin_total_times', 0) or 0
                    transfer_amount = record.get('transfer_amount', 0.0) or 0.0
                    
                    bal_after = f"{balance_after:.2f}" if balance_after is not None else 'None'
                    
                    print(f"账号: {phone}, 日期: {run_date}")
                    print(f"  余额前: {balance_before:.2f}")
                    print(f"  签到后余额: {checkin_balance_after:.2f}")
                    print(f"  最终余额: {bal_after}")
                    print(f"  签到总次数: {checkin_total_times}")
                    print(f"  转账金额: {transfer_amount:.2f}")
                    print()
            
            # 显示无转账的记录(前10条)
            if no_transfer:
                print(f"无转账的记录 (共{len(no_transfer)}条,显示前10条):")
                print()
                
                # 需要获取前一天余额来判断
                for record in no_transfer[:10]:
                    phone = record.get('phone')
                    run_date = record.get('run_date')
                    balance_before = record.get('balance_before')
                    checkin_balance_after = record.get('checkin_balance_after')
                    balance_after = record.get('balance_after')
                    checkin_total_times = record.get('checkin_total_times', 0) or 0
                    
                    # 获取该账号的前一天记录
                    phone_records = [r for r in all_records if r.get('phone') == phone]
                    phone_records.sort(key=lambda r: r.get('run_date', ''))
                    
                    previous_balance_after = None
                    for i, r in enumerate(phone_records):
                        if r.get('run_date') == run_date and i > 0:
                            previous_balance_after = phone_records[i-1].get('balance_after')
                            break
                    
                    bal_after = f"{balance_after:.2f}" if balance_after is not None else 'None'
                    prev_bal = f"{previous_balance_after:.2f}" if previous_balance_after is not None else '0.00(第一条)'
                    
                    # 计算应该的签到奖励
                    if previous_balance_after is not None:
                        should_reward = checkin_balance_after - previous_balance_after
                    else:
                        should_reward = checkin_balance_after  # 第一条记录
                    
                    print(f"账号: {phone}, 日期: {run_date}")
                    print(f"  前一天余额: {prev_bal}")
                    print(f"  当天余额前: {balance_before:.2f}")
                    print(f"  签到后余额: {checkin_balance_after:.2f}")
                    print(f"  最终余额: {bal_after}")
                    print(f"  应得签到奖励: {should_reward:.2f} 元")
                    print(f"  签到总次数: {checkin_total_times}")
                    print()
    
    # 分析"其他情况"的记录
    if other:
        print("=" * 80)
        print(f"其他情况的记录 (共{len(other)}条):")
        print("=" * 80)
        print()
        
        # 显示前20条
        print(f"显示前20条:")
        print()
        
        for record in other[:20]:
            phone = record.get('phone')
            run_date = record.get('run_date')
            balance_before = record.get('balance_before')
            checkin_balance_after = record.get('checkin_balance_after')
            balance_after = record.get('balance_after')
            checkin_total_times = record.get('checkin_total_times', 0) or 0
            transfer_amount = record.get('transfer_amount', 0.0) or 0.0
            
            bal_before = f"{balance_before:.2f}" if balance_before is not None else 'None'
            checkin_bal = f"{checkin_balance_after:.2f}" if checkin_balance_after is not None else 'None'
            bal_after = f"{balance_after:.2f}" if balance_after is not None else 'None'
            
            # 计算余额差
            if balance_before is not None and checkin_balance_after is not None:
                diff = checkin_balance_after - balance_before
                diff_str = f"{diff:.2f}"
            else:
                diff_str = "无法计算"
            
            print(f"账号: {phone}, 日期: {run_date}")
            print(f"  余额前: {bal_before}")
            print(f"  签到后余额: {checkin_bal}")
            print(f"  最终余额: {bal_after}")
            print(f"  余额差: {diff_str}")
            print(f"  签到总次数: {checkin_total_times}")
            if transfer_amount > 0:
                print(f"  转账金额: {transfer_amount:.2f}")
            print()


if __name__ == "__main__":
    try:
        analyze_zero_rewards()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
