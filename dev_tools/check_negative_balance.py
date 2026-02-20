"""
检查负数余额和转账金额累积错误

检查数据库中balance_after为负数的记录，以及transfer_amount的累积错误

运行方式:
    python dev_tools/check_negative_balance.py
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


def check_negative_balance():
    """检查负数余额和转账金额"""
    
    print("=" * 80)
    print("检查负数余额和转账金额累积错误")
    print("=" * 80)
    print()
    
    # 初始化数据库
    db = LocalDatabase()
    all_records = db.get_all_history_records()
    
    print(f"总记录数: {len(all_records)}")
    print()
    
    # 统计负数余额
    negative_balance_after = []
    negative_balance_before = []
    negative_checkin_balance = []
    
    for record in all_records:
        balance_after = record.get('balance_after')
        balance_before = record.get('balance_before')
        checkin_balance_after = record.get('checkin_balance_after')
        
        if balance_after is not None and balance_after < 0:
            negative_balance_after.append(record)
        
        if balance_before is not None and balance_before < 0:
            negative_balance_before.append(record)
        
        if checkin_balance_after is not None and checkin_balance_after < 0:
            negative_checkin_balance.append(record)
    
    print("=" * 80)
    print("负数余额统计:")
    print("=" * 80)
    print(f"balance_after < 0: {len(negative_balance_after)} 条")
    print(f"balance_before < 0: {len(negative_balance_before)} 条")
    print(f"checkin_balance_after < 0: {len(negative_checkin_balance)} 条")
    print()
    
    # 显示balance_after为负数的记录(前20条)
    if negative_balance_after:
        print("=" * 80)
        print(f"balance_after < 0 的记录 (共{len(negative_balance_after)}条,显示前20条):")
        print("=" * 80)
        print()
        
        for record in negative_balance_after[:20]:
            phone = record.get('phone')
            run_date = record.get('run_date')
            balance_before = record.get('balance_before')
            checkin_balance_after = record.get('checkin_balance_after')
            balance_after = record.get('balance_after')
            transfer_amount = record.get('transfer_amount', 0.0) or 0.0
            
            bal_before = f"{balance_before:.2f}" if balance_before is not None else 'None'
            checkin_bal = f"{checkin_balance_after:.2f}" if checkin_balance_after is not None else 'None'
            
            print(f"账号: {phone}, 日期: {run_date}")
            print(f"  余额前: {bal_before}")
            print(f"  签到后余额: {checkin_bal}")
            print(f"  最终余额: {balance_after:.2f}")
            print(f"  转账金额: {transfer_amount:.2f}")
            
            # 计算应该的最终余额
            if checkin_balance_after is not None and transfer_amount > 0:
                should_balance_after = checkin_balance_after - transfer_amount
                print(f"  应该的最终余额: {should_balance_after:.2f}")
                print(f"  差异: {balance_after - should_balance_after:.2f}")
            
            print()
    
    # 检查转账金额累积错误
    print("=" * 80)
    print("检查转账金额累积错误:")
    print("=" * 80)
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
    
    # 检查转账金额是否合理
    transfer_error_records = []
    
    for phone, records in records_by_phone.items():
        for record in records:
            transfer_amount = record.get('transfer_amount', 0.0) or 0.0
            checkin_balance_after = record.get('checkin_balance_after')
            balance_after = record.get('balance_after')
            
            if transfer_amount > 0 and checkin_balance_after is not None and balance_after is not None:
                # 计算应该的最终余额
                should_balance_after = checkin_balance_after - transfer_amount
                
                # 如果差异超过0.01元，说明有问题
                if abs(balance_after - should_balance_after) > 0.01:
                    transfer_error_records.append({
                        'record': record,
                        'should_balance_after': should_balance_after,
                        'diff': balance_after - should_balance_after
                    })
    
    print(f"转账金额计算有误的记录: {len(transfer_error_records)} 条")
    print()
    
    if transfer_error_records:
        print(f"显示前20条:")
        print()
        
        for item in transfer_error_records[:20]:
            record = item['record']
            should_balance_after = item['should_balance_after']
            diff = item['diff']
            
            phone = record.get('phone')
            run_date = record.get('run_date')
            balance_before = record.get('balance_before')
            checkin_balance_after = record.get('checkin_balance_after')
            balance_after = record.get('balance_after')
            transfer_amount = record.get('transfer_amount', 0.0) or 0.0
            
            bal_before = f"{balance_before:.2f}" if balance_before is not None else 'None'
            
            print(f"账号: {phone}, 日期: {run_date}")
            print(f"  余额前: {bal_before}")
            print(f"  签到后余额: {checkin_balance_after:.2f}")
            print(f"  转账金额: {transfer_amount:.2f}")
            print(f"  实际最终余额: {balance_after:.2f}")
            print(f"  应该的最终余额: {should_balance_after:.2f}")
            print(f"  差异: {diff:.2f}")
            print()


if __name__ == "__main__":
    try:
        check_negative_balance()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
