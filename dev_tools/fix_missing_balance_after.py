"""
修复缺少balance_after的记录

通过 checkin_balance_after - transfer_amount 推算 balance_after

运行方式:
    python dev_tools/fix_missing_balance_after.py
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


def fix_missing_balance_after():
    """修复缺少balance_after的记录"""
    
    print("=" * 80)
    print("修复缺少balance_after的记录")
    print("=" * 80)
    print()
    
    # 初始化数据库
    db = LocalDatabase()
    all_records = db.get_all_history_records()
    
    # 找出缺少balance_after的记录
    missing_records = []
    for record in all_records:
        if record.get('balance_after') is None:
            missing_records.append(record)
    
    if not missing_records:
        print("✓ 没有缺少balance_after的记录")
        return
    
    print(f"找到 {len(missing_records)} 条缺少balance_after的记录")
    print()
    
    # 统计信息
    fixed_count = 0
    skipped_count = 0
    
    conn = db._get_connection()
    cursor = conn.cursor()
    
    for record in missing_records:
        record_id = record.get('id')
        phone = record.get('phone')
        run_date = record.get('run_date')
        checkin_balance_after = record.get('checkin_balance_after')
        transfer_amount = record.get('transfer_amount', 0.0) or 0.0
        
        print(f"记录 {phone} - {run_date} (ID: {record_id}):")
        print(f"  签到后余额: {checkin_balance_after}")
        print(f"  转账金额: {transfer_amount}")
        
        # 检查是否可以推算
        if checkin_balance_after is None:
            print(f"  ✗ 跳过: 缺少签到后余额")
            skipped_count += 1
            print()
            continue
        
        # 推算balance_after
        balance_after = checkin_balance_after - transfer_amount
        
        print(f"  推算最终余额: {balance_after:.2f}")
        
        # 更新数据库
        try:
            cursor.execute(
                "UPDATE history_records SET balance_after = ? WHERE id = ?",
                (balance_after, record_id)
            )
            conn.commit()
            fixed_count += 1
            print(f"  ✓ 已更新")
        except Exception as e:
            print(f"  ✗ 更新失败: {e}")
        
        print()
    
    conn.close()
    
    print("=" * 80)
    print(f"修复完成:")
    print(f"  已修复: {fixed_count} 条")
    print(f"  跳过: {skipped_count} 条")
    print("=" * 80)


if __name__ == "__main__":
    try:
        fix_missing_balance_after()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
