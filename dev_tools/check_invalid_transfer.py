"""
检查无效的转账记录

转账金额 < 最小转账金额(30元) 的记录都是错误的

运行方式:
    python dev_tools/check_invalid_transfer.py
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


def check_invalid_transfer():
    """检查无效的转账记录"""
    
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
    print("检查无效的转账记录")
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
    
    # 显示这些记录
    print("=" * 80)
    print(f"无效的转账记录 (共{len(invalid_transfer_records)}条):")
    print("=" * 80)
    print()
    
    # 按转账金额排序
    invalid_transfer_records.sort(key=lambda r: r.get('transfer_amount', 0.0) or 0.0)
    
    for record in invalid_transfer_records:
        phone = record.get('phone')
        run_date = record.get('run_date')
        transfer_amount = record.get('transfer_amount', 0.0) or 0.0
        balance_before = record.get('balance_before')
        checkin_balance_after = record.get('checkin_balance_after')
        balance_after = record.get('balance_after')
        checkin_reward = record.get('checkin_reward', 0.0) or 0.0
        
        bal_before = f"{balance_before:.2f}" if balance_before is not None else 'None'
        checkin_bal = f"{checkin_balance_after:.2f}" if checkin_balance_after is not None else 'None'
        bal_after = f"{balance_after:.2f}" if balance_after is not None else 'None'
        
        print(f"账号: {phone}, 日期: {run_date}")
        print(f"  转账金额: {transfer_amount:.2f} 元 (< {MIN_TRANSFER_AMOUNT} 元)")
        print(f"  余额前: {bal_before}")
        print(f"  签到后余额: {checkin_bal}")
        print(f"  最终余额: {bal_after}")
        print(f"  签到奖励: {checkin_reward:.2f}")
        print()
    
    # 统计分析
    print("=" * 80)
    print("统计分析:")
    print("=" * 80)
    print()
    
    # 按转账金额范围统计
    ranges = [
        (0, 1, "0-1元"),
        (1, 5, "1-5元"),
        (5, 10, "5-10元"),
        (10, 20, "10-20元"),
        (20, 30, "20-30元")
    ]
    
    for min_val, max_val, label in ranges:
        count = sum(1 for r in invalid_transfer_records 
                   if min_val < (r.get('transfer_amount', 0.0) or 0.0) <= max_val)
        if count > 0:
            print(f"  {label}: {count} 条")
    
    print()
    
    # 这些记录的问题
    print("=" * 80)
    print("问题分析:")
    print("=" * 80)
    print()
    print("这些记录的转账金额都小于最小转账金额，说明：")
    print("1. 这些记录不应该有转账")
    print("2. balance_after 应该等于 checkin_balance_after")
    print("3. 需要将这些记录的 transfer_amount 设为 0")
    print("4. 需要将这些记录的 balance_after 设为 checkin_balance_after")
    print()


if __name__ == "__main__":
    try:
        check_invalid_transfer()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
