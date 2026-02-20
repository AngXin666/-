"""
修复剩余的负数余额记录

将balance_after < 0的记录直接设为0

运行方式:
    python dev_tools/fix_remaining_negative.py
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


def fix_remaining_negative():
    """修复剩余的负数余额记录"""
    
    print("=" * 80)
    print("修复剩余的负数余额记录")
    print("=" * 80)
    print()
    
    # 初始化数据库
    db = LocalDatabase()
    all_records = db.get_all_history_records()
    
    # 找出balance_after < 0的记录
    negative_records = []
    for record in all_records:
        balance_after = record.get('balance_after')
        if balance_after is not None and balance_after < 0:
            negative_records.append(record)
    
    print(f"balance_after < 0的记录: {len(negative_records)} 条")
    print()
    
    if not negative_records:
        print("没有需要修复的记录")
        return
    
    # 修复这些记录
    print("开始修复...")
    print()
    
    updated_count = 0
    
    for record in negative_records:
        record_id = record.get('id')
        phone = record.get('phone')
        run_date = record.get('run_date')
        old_balance_after = record.get('balance_after')
        
        try:
            conn = db._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE history_records 
                SET balance_after = 0.0
                WHERE id = ?
            ''', (record_id,))
            conn.commit()
            conn.close()
            
            updated_count += 1
            print(f"[{phone}] [{run_date}] balance_after: {old_balance_after:.2f} → 0.00")
        
        except Exception as e:
            print(f"❌ [{phone}] [{run_date}] 更新失败: {e}")
    
    print()
    print("=" * 80)
    print("修复完成")
    print("=" * 80)
    print(f"已更新: {updated_count} 条")
    print()
    
    # 验证
    all_records = db.get_all_history_records()
    negative_count = sum(1 for r in all_records if r.get('balance_after') is not None and r.get('balance_after') < 0)
    print(f"验证: balance_after < 0的记录: {negative_count} 条")
    
    if negative_count == 0:
        print("✓ 所有负数余额已修复")


if __name__ == "__main__":
    try:
        fix_remaining_negative()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
