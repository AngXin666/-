"""
正确修复balance_after

对于有转账的记录，直接使用 checkin_balance_after 作为 balance_after
因为转账金额有累积错误，不可靠

运行方式:
    python dev_tools/fix_balance_after_correct.py
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


def fix_balance_after_correct():
    """正确修复balance_after"""
    
    print("=" * 80)
    print("正确修复balance_after")
    print("=" * 80)
    print()
    print("对于有转账的记录:")
    print("  balance_after = checkin_balance_after (忽略transfer_amount)")
    print()
    
    # 初始化数据库
    db = LocalDatabase()
    all_records = db.get_all_history_records()
    
    print(f"总记录数: {len(all_records)}")
    print()
    
    # 找出有转账的记录
    records_with_transfer = []
    
    for record in all_records:
        transfer_amount = record.get('transfer_amount', 0.0) or 0.0
        if transfer_amount > 0:
            records_with_transfer.append(record)
    
    print(f"有转账的记录: {len(records_with_transfer)} 条")
    print()
    
    if not records_with_transfer:
        print("没有需要修复的记录")
        return
    
    # 开始修复
    print("=" * 80)
    print("开始修复...")
    print("=" * 80)
    print()
    
    updated_count = 0
    error_count = 0
    
    for record in records_with_transfer:
        record_id = record.get('id')
        phone = record.get('phone')
        run_date = record.get('run_date')
        old_balance_after = record.get('balance_after')
        checkin_balance_after = record.get('checkin_balance_after')
        transfer_amount = record.get('transfer_amount', 0.0) or 0.0
        
        if checkin_balance_after is None:
            print(f"⚠️ [{phone}] [{run_date}] 没有签到后余额，跳过")
            continue
        
        # 直接使用签到后余额作为最终余额
        new_balance_after = checkin_balance_after
        
        # 检查是否需要更新
        if old_balance_after is None or abs(new_balance_after - old_balance_after) > 0.001:
            try:
                conn = db._get_connection()
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE history_records 
                    SET balance_after = ?
                    WHERE id = ?
                ''', (new_balance_after, record_id))
                conn.commit()
                conn.close()
                
                updated_count += 1
                
                # 显示前20条更新
                if updated_count <= 20:
                    old_bal_str = f"{old_balance_after:.2f}" if old_balance_after is not None else 'None'
                    print(f"[{phone}] [{run_date}]")
                    print(f"  签到后余额: {checkin_balance_after:.2f}")
                    print(f"  转账金额: {transfer_amount:.2f}")
                    print(f"  最终余额: {old_bal_str} → {new_balance_after:.2f}")
                    print()
            
            except Exception as e:
                print(f"❌ [{phone}] [{run_date}] 更新失败: {e}")
                error_count += 1
    
    # 输出统计信息
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
        fix_balance_after_correct()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
