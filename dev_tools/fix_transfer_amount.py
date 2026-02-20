"""
修复转账金额累积错误

将transfer_amount除以2，并重新计算balance_after

运行方式:
    python dev_tools/fix_transfer_amount.py
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


def fix_transfer_amount():
    """修复转账金额累积错误"""
    
    print("=" * 80)
    print("修复转账金额累积错误")
    print("=" * 80)
    print()
    print("操作:")
    print("1. 将所有 transfer_amount > 0 的记录的 transfer_amount 除以 2")
    print("2. 重新计算 balance_after = checkin_balance_after - transfer_amount")
    print()
    
    # 初始化数据库
    db = LocalDatabase()
    all_records = db.get_all_history_records()
    
    print(f"总记录数: {len(all_records)}")
    print()
    
    # 找出需要修复的记录
    records_to_fix = []
    
    for record in all_records:
        transfer_amount = record.get('transfer_amount', 0.0) or 0.0
        if transfer_amount > 0:
            records_to_fix.append(record)
    
    print(f"需要修复的记录: {len(records_to_fix)} 条")
    print()
    
    if not records_to_fix:
        print("没有需要修复的记录")
        return
    
    # 开始修复
    print("=" * 80)
    print("开始修复...")
    print("=" * 80)
    print()
    
    updated_count = 0
    error_count = 0
    
    for record in records_to_fix:
        record_id = record.get('id')
        phone = record.get('phone')
        run_date = record.get('run_date')
        old_transfer_amount = record.get('transfer_amount', 0.0) or 0.0
        checkin_balance_after = record.get('checkin_balance_after')
        old_balance_after = record.get('balance_after')
        
        # 计算新的转账金额
        new_transfer_amount = old_transfer_amount / 2.0
        
        # 计算新的最终余额
        if checkin_balance_after is not None:
            new_balance_after = checkin_balance_after - new_transfer_amount
        else:
            # 如果没有签到后余额，无法计算
            print(f"⚠️ [{phone}] [{run_date}] 没有签到后余额，跳过")
            continue
        
        try:
            # 更新数据库
            conn = db._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE history_records 
                SET transfer_amount = ?, balance_after = ?
                WHERE id = ?
            ''', (new_transfer_amount, new_balance_after, record_id))
            conn.commit()
            conn.close()
            
            updated_count += 1
            
            # 显示前20条更新
            if updated_count <= 20:
                bal_after_str = f"{old_balance_after:.2f}" if old_balance_after is not None else 'None'
                print(f"[{phone}] [{run_date}]")
                print(f"  转账金额: {old_transfer_amount:.2f} → {new_transfer_amount:.2f}")
                print(f"  最终余额: {bal_after_str} → {new_balance_after:.2f}")
                print()
        
        except Exception as e:
            print(f"❌ [{phone}] [{run_date}] 更新失败: {e}")
            error_count += 1
    
    # 输出统计
    print()
    print("=" * 80)
    print("修复完成")
    print("=" * 80)
    print(f"已更新: {updated_count} 条")
    print(f"错误: {error_count} 条")
    print()
    
    # 验证修复结果
    print("=" * 80)
    print("验证修复结果...")
    print("=" * 80)
    print()
    
    all_records = db.get_all_history_records()
    
    # 检查负数余额
    negative_balance_after = []
    for record in all_records:
        balance_after = record.get('balance_after')
        if balance_after is not None and balance_after < 0:
            negative_balance_after.append(record)
    
    print(f"balance_after < 0: {len(negative_balance_after)} 条")
    
    if negative_balance_after:
        print()
        print("仍有负数余额的记录:")
        for record in negative_balance_after[:10]:
            phone = record.get('phone')
            run_date = record.get('run_date')
            balance_after = record.get('balance_after')
            print(f"  [{phone}] [{run_date}] balance_after: {balance_after:.2f}")
    else:
        print("✓ 所有记录的 balance_after 都不是负数")
    
    print()


if __name__ == "__main__":
    try:
        fix_transfer_amount()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
