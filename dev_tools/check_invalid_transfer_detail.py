"""
详细检查无效转账记录的签到后余额

转账金额 < 最小转账金额(30元) 的记录，检查它们的签到后余额

运行方式:
    python dev_tools/check_invalid_transfer_detail.py
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


def check_invalid_transfer_detail():
    """详细检查无效转账记录的签到后余额"""
    
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
    print("详细检查无效转账记录的签到后余额")
    print("=" * 80)
    print(f"最小转账金额: {MIN_TRANSFER_AMOUNT} 元")
    print()
    
    # 初始化数据库
    db = LocalDatabase()
    all_records = db.get_all_history_records()
    
    print(f"总记录数: {len(all_records)}")
    print()
    
    # 找出转账金额 > 0 但 < 最小转账金额的记录
    invalid_transfer_records = []
    
    for record in all_records:
        transfer_amount = record.get('transfer_amount', 0.0) or 0.0
        if 0 < transfer_amount < MIN_TRANSFER_AMOUNT:
            invalid_transfer_records.append(record)
    
    print(f"转账金额 > 0 但 < {MIN_TRANSFER_AMOUNT} 元的记录: {len(invalid_transfer_records)} 条")
    print()
    
    if not invalid_transfer_records:
        print("✓ 没有无效的转账记录")
        return
    
    # 统计签到后余额情况
    has_checkin_balance = []
    no_checkin_balance = []
    checkin_balance_zero = []
    
    for record in invalid_transfer_records:
        checkin_balance_after = record.get('checkin_balance_after')
        
        if checkin_balance_after is None:
            no_checkin_balance.append(record)
        elif checkin_balance_after == 0:
            checkin_balance_zero.append(record)
        else:
            has_checkin_balance.append(record)
    
    print("=" * 80)
    print("签到后余额统计:")
    print("=" * 80)
    print(f"有签到后余额(>0): {len(has_checkin_balance)} 条")
    print(f"签到后余额=0: {len(checkin_balance_zero)} 条")
    print(f"没有签到后余额(None): {len(no_checkin_balance)} 条")
    print()
    
    # 显示签到后余额=0的记录
    if checkin_balance_zero:
        print("=" * 80)
        print(f"签到后余额=0的记录 (共{len(checkin_balance_zero)}条):")
        print("=" * 80)
        print()
        
        for record in checkin_balance_zero[:20]:
            phone = record.get('phone')
            run_date = record.get('run_date')
            transfer_amount = record.get('transfer_amount', 0.0) or 0.0
            balance_before = record.get('balance_before')
            balance_after = record.get('balance_after')
            
            bal_before = f"{balance_before:.2f}" if balance_before is not None else 'None'
            bal_after = f"{balance_after:.2f}" if balance_after is not None else 'None'
            
            print(f"账号: {phone}, 日期: {run_date}")
            print(f"  转账金额: {transfer_amount:.2f} 元")
            print(f"  余额前: {bal_before}")
            print(f"  签到后余额: 0.00")
            print(f"  最终余额: {bal_after}")
            print()
    
    # 显示没有签到后余额的记录
    if no_checkin_balance:
        print("=" * 80)
        print(f"没有签到后余额的记录 (共{len(no_checkin_balance)}条):")
        print("=" * 80)
        print()
        
        for record in no_checkin_balance[:20]:
            phone = record.get('phone')
            run_date = record.get('run_date')
            transfer_amount = record.get('transfer_amount', 0.0) or 0.0
            balance_before = record.get('balance_before')
            balance_after = record.get('balance_after')
            
            bal_before = f"{balance_before:.2f}" if balance_before is not None else 'None'
            bal_after = f"{balance_after:.2f}" if balance_after is not None else 'None'
            
            print(f"账号: {phone}, 日期: {run_date}")
            print(f"  转账金额: {transfer_amount:.2f} 元")
            print(f"  余额前: {bal_before}")
            print(f"  签到后余额: None")
            print(f"  最终余额: {bal_after}")
            print()
    
    # 显示有签到后余额的记录(前20条)
    if has_checkin_balance:
        print("=" * 80)
        print(f"有签到后余额(>0)的记录 (共{len(has_checkin_balance)}条,显示前20条):")
        print("=" * 80)
        print()
        
        for record in has_checkin_balance[:20]:
            phone = record.get('phone')
            run_date = record.get('run_date')
            transfer_amount = record.get('transfer_amount', 0.0) or 0.0
            balance_before = record.get('balance_before')
            checkin_balance_after = record.get('checkin_balance_after')
            balance_after = record.get('balance_after')
            
            bal_before = f"{balance_before:.2f}" if balance_before is not None else 'None'
            bal_after = f"{balance_after:.2f}" if balance_after is not None else 'None'
            
            print(f"账号: {phone}, 日期: {run_date}")
            print(f"  转账金额: {transfer_amount:.2f} 元")
            print(f"  余额前: {bal_before}")
            print(f"  签到后余额: {checkin_balance_after:.2f}")
            print(f"  最终余额: {bal_after}")
            print()
    
    print("=" * 80)
    print("问题分析:")
    print("=" * 80)
    print()
    
    if checkin_balance_zero:
        print(f"1. {len(checkin_balance_zero)}条记录的签到后余额=0")
        print("   这些记录可能是转账失败或数据采集问题")
        print()
    
    if no_checkin_balance:
        print(f"2. {len(no_checkin_balance)}条记录没有签到后余额")
        print("   这些记录缺少关键数据")
        print()
    
    if has_checkin_balance:
        print(f"3. {len(has_checkin_balance)}条记录有签到后余额(>0)")
        print("   这些记录的转账金额不应该存在（小于最小转账金额）")
        print("   应该将transfer_amount设为0")
        print()


if __name__ == "__main__":
    try:
        check_invalid_transfer_detail()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
