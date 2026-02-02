"""
检查所有可能有问题的数据
"""

import sqlite3
from pathlib import Path

def check_all_bad_data():
    """检查所有可能有问题的数据"""
    db_path = Path("runtime_data") / "license.db"
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    print("=" * 80)
    print("检查1: 昵称为'-'的记录")
    print("=" * 80)
    
    cursor.execute("""
        SELECT COUNT(*), MIN(created_at), MAX(created_at)
        FROM history_records
        WHERE nickname = '-'
    """)
    
    count, min_date, max_date = cursor.fetchone()
    print(f"共 {count} 条记录")
    print(f"最早: {min_date}")
    print(f"最晚: {max_date}")
    
    # 查看这些记录的user_id是否像昵称
    cursor.execute("""
        SELECT phone, nickname, user_id, run_date, created_at
        FROM history_records
        WHERE nickname = '-'
        ORDER BY created_at DESC
        LIMIT 20
    """)
    
    print(f"\n最近20条昵称为'-'的记录:")
    for row in cursor.fetchall():
        phone, nickname, user_id, run_date, created_at = row
        print(f"  {phone} | 昵称='{nickname}' | 用户ID='{user_id}' | {run_date} | {created_at}")
    
    print("\n" + "=" * 80)
    print("检查2: 用户ID不是纯数字的记录（可能是昵称）")
    print("=" * 80)
    
    # 查找用户ID包含中文或特殊字符的记录
    cursor.execute("""
        SELECT COUNT(*)
        FROM history_records
        WHERE user_id NOT GLOB '*[0-9]*' OR user_id GLOB '*[一-龥]*'
    """)
    
    count = cursor.fetchone()[0]
    print(f"共 {count} 条记录")
    
    cursor.execute("""
        SELECT phone, nickname, user_id, run_date, created_at
        FROM history_records
        WHERE user_id NOT GLOB '*[0-9]*' OR user_id GLOB '*[一-龥]*'
        ORDER BY created_at DESC
        LIMIT 20
    """)
    
    print(f"\n最近20条用户ID异常的记录:")
    for row in cursor.fetchall():
        phone, nickname, user_id, run_date, created_at = row
        print(f"  {phone} | 昵称='{nickname}' | 用户ID='{user_id}' | {run_date} | {created_at}")
    
    print("\n" + "=" * 80)
    print("检查3: 按日期统计异常数据")
    print("=" * 80)
    
    cursor.execute("""
        SELECT run_date, COUNT(*) as count
        FROM history_records
        WHERE nickname = '-'
        GROUP BY run_date
        ORDER BY run_date DESC
    """)
    
    print("\n每天昵称为'-'的记录数:")
    for row in cursor.fetchall():
        run_date, count = row
        print(f"  {run_date}: {count} 条")
    
    conn.close()


if __name__ == '__main__':
    check_all_bad_data()
