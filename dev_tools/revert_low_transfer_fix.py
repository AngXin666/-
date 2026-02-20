"""
撤销低转账金额修复

将之前修改的3条记录恢复到原来的值

运行方式:
    python dev_tools/revert_low_transfer_fix.py
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


def revert_low_transfer_fix():
    """撤销低转账金额修复"""
    
    print("=" * 80)
    print("撤销低转账金额修复")
    print("=" * 80)
    print()
    
    # 需要恢复的记录（从之前的修复输出中获取）
    revert_records = [
        {
            'phone': '13925070304',
            'date': '2026-02-04',
            'original_transfer': 27.34,
            'wrong_transfer': 58.64
        },
        {
            'phone': '13595405182',
            'date': '2026-02-05',
            'original_transfer': 29.45,
            'wrong_transfer': 16.91
        },
        {
            'phone': '13322736481',
            'date': '2026-02-04',
            'original_transfer': 28.28,
            'wrong_transfer': 62.74
        }
    ]
    
    # 初始化数据库
    db = LocalDatabase()
    
    reverted_count = 0
    
    for record in revert_records:
        print(f"恢复账号 {record['phone']}, 日期 {record['date']}")
        print(f"  错误的转账金额: {record['wrong_transfer']:.2f}")
        print(f"  恢复为原始值: {record['original_transfer']:.2f}")
        
        try:
            import sqlite3
            with db._lock:
                conn = sqlite3.connect(str(db.db_path))
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE history_records 
                    SET transfer_amount = ?
                    WHERE phone = ? AND run_date = ?
                """, (record['original_transfer'], record['phone'], record['date']))
                conn.commit()
                conn.close()
            
            reverted_count += 1
            print(f"  ✓ 恢复成功")
            
        except Exception as e:
            print(f"  ❌ 恢复失败: {e}")
        
        print()
    
    # 总结
    print("=" * 80)
    print("撤销完成")
    print("=" * 80)
    print(f"需要恢复: {len(revert_records)}")
    print(f"已恢复: {reverted_count}")
    print()


if __name__ == "__main__":
    try:
        revert_low_transfer_fix()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
