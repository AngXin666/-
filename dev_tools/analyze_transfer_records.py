"""
分析转账记录

检查那些余额突然下降的账号，看看前一天是否有转账记录

运行方式:
    python dev_tools/analyze_transfer_records.py
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


def analyze_transfer_records():
    """分析转账记录"""
    
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
    print("分析转账记录")
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
    
    # 找出余额突然大幅下降的记录（可能是转账）
    print("=" * 80)
    print("查找余额突然下降的记录（可能是转账）:")
    print("=" * 80)
    print()
    
    problem_cases = []
    
    for phone, records in records_by_phone.items():
        previous_checkin_balance_after = None
        
        for idx, record in enumerate(records):
            run_date = record.get('run_date')
            checkin_balance_after = record.get('checkin_balance_after')
            balance_after = record.get('balance_after')
            transfer_amount = record.get('transfer_amount', 0.0) or 0.0
            
            # 跳过没有余额数据的记录
            if checkin_balance_after is None:
                continue
            
            # 如果有前一天的签到后余额
            if previous_checkin_balance_after is not None:
                # 计算余额变化
                balance_change = checkin_balance_after - previous_checkin_balance_after
                
                # 如果余额下降超过20元（可能是转账）
                if balance_change < -20:
                    # 检查当天或前一天是否有转账记录
                    current_transfer = transfer_amount
                    
                    # 查找前一天的记录
                    previous_record = None
                    if idx > 0:
                        previous_record = records[idx - 1]
                    
                    previous_transfer = 0.0
                    previous_date = None
                    if previous_record:
                        previous_transfer = previous_record.get('transfer_amount', 0.0) or 0.0
                        previous_date = previous_record.get('run_date')
                    
                    problem_cases.append({
                        'phone': phone,
                        'date': run_date,
                        'previous_date': previous_date,
                        'previous_balance': previous_checkin_balance_after,
                        'current_balance': checkin_balance_after,
                        'balance_change': balance_change,
                        'current_transfer': current_transfer,
                        'previous_transfer': previous_transfer
                    })
            
            # 更新前一天的签到后余额
            previous_checkin_balance_after = checkin_balance_after
    
    print(f"发现 {len(problem_cases)} 个余额突然下降的情况")
    print()
    
    if not problem_cases:
        print("✓ 没有发现余额突然下降的情况")
        return
    
    # 分析这些情况
    print("=" * 80)
    print("详细分析（显示前30个）:")
    print("=" * 80)
    print()
    
    has_transfer_record = 0
    no_transfer_record = 0
    
    for idx, case in enumerate(problem_cases[:30], 1):
        phone = case['phone']
        date = case['date']
        previous_date = case['previous_date']
        previous_balance = case['previous_balance']
        current_balance = case['current_balance']
        balance_change = case['balance_change']
        current_transfer = case['current_transfer']
        previous_transfer = case['previous_transfer']
        
        print(f"[{idx}] 账号: {phone}")
        print(f"  前一天日期: {previous_date}")
        print(f"  当天日期: {date}")
        print(f"  前一天签到后余额: {previous_balance:.2f}")
        print(f"  当天签到后余额: {current_balance:.2f}")
        print(f"  余额变化: {balance_change:.2f} 元")
        print(f"  前一天转账金额: {previous_transfer:.2f} 元")
        print(f"  当天转账金额: {current_transfer:.2f} 元")
        
        # 判断是否有转账记录
        if previous_transfer >= MIN_TRANSFER_AMOUNT:
            print(f"  ✓ 前一天有转账记录（{previous_transfer:.2f}元）")
            has_transfer_record += 1
        elif current_transfer >= MIN_TRANSFER_AMOUNT:
            print(f"  ✓ 当天有转账记录（{current_transfer:.2f}元）")
            has_transfer_record += 1
        else:
            print(f"  ❌ 前一天和当天都没有有效的转账记录")
            no_transfer_record += 1
        
        print()
    
    # 统计
    print()
    print("=" * 80)
    print("统计结果:")
    print("=" * 80)
    print(f"总共: {len(problem_cases)} 个余额突然下降的情况")
    print(f"有转账记录: {has_transfer_record} 个")
    print(f"没有转账记录: {no_transfer_record} 个")
    print()
    
    if no_transfer_record > 0:
        print("=" * 80)
        print("问题分析:")
        print("=" * 80)
        print()
        print(f"有 {no_transfer_record} 个账号的余额突然下降，但没有转账记录。")
        print("可能的原因:")
        print("1. 转账记录丢失或未正确保存")
        print("2. 转账金额字段被错误地设置为0")
        print("3. 数据采集时出现问题")
        print()


if __name__ == "__main__":
    try:
        analyze_transfer_records()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
