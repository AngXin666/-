"""
检查被修复转账金额的账号的余额情况

查看这些账号前一天的余额，分析余额变化

运行方式:
    python dev_tools/check_fixed_accounts_balance.py
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
from datetime import datetime, timedelta


def check_fixed_accounts_balance():
    """检查被修复转账金额的账号的余额情况"""
    
    print("=" * 80)
    print("检查被修复转账金额的账号的余额情况")
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
    
    # 找出有转账金额=0但签到后余额>0的记录（刚修复的）
    fixed_records = []
    for record in all_records:
        transfer_amount = record.get('transfer_amount', 0.0) or 0.0
        checkin_balance_after = record.get('checkin_balance_after')
        balance_after = record.get('balance_after')
        
        # 转账金额=0，但签到后余额>0，且最终余额>0
        if (transfer_amount == 0.0 and 
            checkin_balance_after is not None and checkin_balance_after > 0 and
            balance_after is not None and balance_after > 0):
            fixed_records.append(record)
    
    print(f"刚修复的记录（转账金额=0，签到后余额>0）: {len(fixed_records)} 条")
    print()
    
    # 按账号分组这些记录
    fixed_by_phone = {}
    for record in fixed_records:
        phone = record.get('phone')
        if phone not in fixed_by_phone:
            fixed_by_phone[phone] = []
        fixed_by_phone[phone].append(record)
    
    print(f"涉及账号数: {len(fixed_by_phone)}")
    print()
    
    # 分析每个账号的余额变化
    print("=" * 80)
    print("余额变化分析（显示前30个账号）:")
    print("=" * 80)
    print()
    
    problem_accounts = []
    
    for idx, (phone, fixed_recs) in enumerate(list(fixed_by_phone.items())[:30], 1):
        print(f"[{idx}] 账号: {phone}")
        
        # 获取该账号的所有记录
        all_phone_records = records_by_phone.get(phone, [])
        
        for fixed_rec in fixed_recs:
            run_date = fixed_rec.get('run_date')
            balance_before = fixed_rec.get('balance_before')
            checkin_balance_after = fixed_rec.get('checkin_balance_after')
            balance_after = fixed_rec.get('balance_after')
            
            # 找前一天的记录
            try:
                current_date = datetime.strptime(run_date, '%Y-%m-%d')
                previous_date = (current_date - timedelta(days=1)).strftime('%Y-%m-%d')
            except:
                previous_date = None
            
            previous_record = None
            if previous_date:
                for rec in all_phone_records:
                    if rec.get('run_date') == previous_date:
                        previous_record = rec
                        break
            
            previous_balance_after = None
            if previous_record:
                previous_balance_after = previous_record.get('balance_after')
            
            # 显示信息
            print(f"  日期: {run_date}")
            
            if previous_balance_after is not None:
                print(f"    前一天余额: {previous_balance_after:.2f}")
            else:
                print(f"    前一天余额: 无记录")
            
            bal_before_str = f"{balance_before:.2f}" if balance_before is not None else 'None'
            print(f"    余额前: {bal_before_str}")
            print(f"    签到后余额: {checkin_balance_after:.2f}")
            print(f"    最终余额: {balance_after:.2f}")
            
            # 检查是否有问题
            if previous_balance_after is not None:
                # 计算签到奖励
                calculated_reward = checkin_balance_after - previous_balance_after
                print(f"    计算签到奖励: {calculated_reward:.2f} 元")
                
                # 检查是否异常
                if calculated_reward < 0 or calculated_reward > 10:
                    print(f"    ⚠️ 异常: 签到奖励不合理")
                    problem_accounts.append((phone, run_date, calculated_reward))
            else:
                # 第一条记录，前一天余额默认为0
                calculated_reward = checkin_balance_after - 0.0
                print(f"    计算签到奖励: {calculated_reward:.2f} 元（第一条记录）")
                
                if calculated_reward < 0 or calculated_reward > 10:
                    print(f"    ⚠️ 异常: 签到奖励不合理")
                    problem_accounts.append((phone, run_date, calculated_reward))
            
            print()
    
    # 统计问题账号
    if problem_accounts:
        print()
        print("=" * 80)
        print(f"发现 {len(problem_accounts)} 条异常记录:")
        print("=" * 80)
        print()
        
        for phone, run_date, reward in problem_accounts[:20]:
            print(f"  [{phone}] [{run_date}] 签到奖励: {reward:.2f} 元")
    else:
        print()
        print("=" * 80)
        print("✓ 所有记录的签到奖励都正常")
        print("=" * 80)
    
    print()


if __name__ == "__main__":
    try:
        check_fixed_accounts_balance()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
