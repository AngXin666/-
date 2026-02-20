"""
检查签到奖励为0但余额下降的记录

这些记录可能是转账导致余额下降，但签到奖励被错误地设为0

运行方式:
    python dev_tools/check_zero_reward_transfers.py
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


def check_zero_reward_transfers():
    """检查签到奖励为0但余额下降的记录"""
    
    # 从转账配置读取最小转账金额
    MIN_TRANSFER_AMOUNT = 30.0
    try:
        import json
        transfer_config_path = project_root / "transfer_config.json"
        if transfer_config_path.exists():
            with open(transfer_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                MIN_TRANSFER_AMOUNT = config.get('min_transfer_amount', 30.0)
    except Exception as e:
        print(f"⚠️ 读取转账配置失败: {e}, 使用默认值 {MIN_TRANSFER_AMOUNT} 元")
    
    print("=" * 80)
    print("检查签到奖励为0但余额下降的记录")
    print("=" * 80)
    print(f"最小转账金额: {MIN_TRANSFER_AMOUNT} 元")
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
    
    # 找出签到奖励为0但余额下降的记录
    print("=" * 80)
    print("签到奖励为0但余额下降的记录（显示前30条）:")
    print("=" * 80)
    print()
    
    problem_cases = []
    
    for phone, records in records_by_phone.items():
        previous_checkin_balance_after = None
        
        for record in records:
            run_date = record.get('run_date')
            checkin_balance_after = record.get('checkin_balance_after')
            checkin_reward = record.get('checkin_reward', 0.0) or 0.0
            transfer_amount = record.get('transfer_amount', 0.0) or 0.0
            
            if checkin_balance_after is None:
                previous_checkin_balance_after = None
                continue
            
            if previous_checkin_balance_after is not None:
                balance_change = checkin_balance_after - previous_checkin_balance_after
                
                # 余额下降超过20元，签到奖励为0，没有转账记录
                if balance_change < -20 and checkin_reward == 0.0 and transfer_amount < MIN_TRANSFER_AMOUNT:
                    problem_cases.append({
                        'phone': phone,
                        'date': run_date,
                        'previous_balance': previous_checkin_balance_after,
                        'current_balance': checkin_balance_after,
                        'balance_change': balance_change,
                        'checkin_reward': checkin_reward,
                        'transfer_amount': transfer_amount
                    })
            
            previous_checkin_balance_after = checkin_balance_after
    
    print(f"发现 {len(problem_cases)} 条记录")
    print()
    
    if not problem_cases:
        print("✓ 没有发现问题记录")
        return
    
    # 显示前30条
    for idx, case in enumerate(problem_cases[:30], 1):
        phone = case['phone']
        date = case['date']
        previous_balance = case['previous_balance']
        current_balance = case['current_balance']
        balance_change = case['balance_change']
        checkin_reward = case['checkin_reward']
        transfer_amount = case['transfer_amount']
        
        # 推算转账金额
        calculated_transfer = previous_balance - current_balance + checkin_reward
        
        print(f"[{idx}] 账号: {phone}, 日期: {date}")
        print(f"  前一天签到后余额: {previous_balance:.2f}")
        print(f"  当天签到后余额: {current_balance:.2f}")
        print(f"  余额变化: {balance_change:.2f}")
        print(f"  签到奖励: {checkin_reward:.2f}")
        print(f"  转账金额: {transfer_amount:.2f}")
        print(f"  推算转账金额: {calculated_transfer:.2f}")
        print()
    
    print()
    print("=" * 80)
    print("问题分析:")
    print("=" * 80)
    print()
    print(f"这 {len(problem_cases)} 条记录的签到奖励都是0，但余额下降了。")
    print("可能的原因:")
    print("1. 这些记录确实发生了转账，但签到奖励被错误地设为0")
    print("2. 应该根据余额变化推算转账金额")
    print()
    print("建议:")
    print("1. 先重新计算这些记录的签到奖励")
    print("2. 然后再推算转账金额")
    print()


if __name__ == "__main__":
    try:
        check_zero_reward_transfers()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
