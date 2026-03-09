"""
检查数据库中昵称和用户ID字段的实际值
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3

# 连接数据库
db_path = "runtime_data/license.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 查询最近20条记录的昵称和用户ID
cursor.execute("""
    SELECT phone, nickname, user_id, run_date, created_at
    FROM history_records
    ORDER BY created_at DESC
    LIMIT 20
""")

print("=" * 100)
print("数据库中最近20条记录的昵称和用户ID：")
print("=" * 100)
print(f"{'手机号':<15} {'昵称':<20} {'用户ID':<15} {'日期':<12} {'创建时间':<20}")
print("-" * 100)

none_count = 0
null_count = 0
valid_count = 0

for row in cursor.fetchall():
    phone, nickname, user_id, run_date, created_at = row
    
    # 检查昵称和用户ID的类型
    nickname_type = type(nickname).__name__
    user_id_type = type(user_id).__name__
    
    # 统计
    if nickname is None or user_id is None:
        null_count += 1
    elif str(nickname).lower() == 'none' or str(user_id).lower() == 'none':
        none_count += 1
    else:
        valid_count += 1
    
    # 显示值和类型
    nickname_display = f"{nickname} ({nickname_type})" if nickname else f"NULL ({nickname_type})"
    user_id_display = f"{user_id} ({user_id_type})" if user_id else f"NULL ({user_id_type})"
    
    print(f"{phone:<15} {nickname_display:<20} {user_id_display:<15} {run_date:<12} {created_at:<20}")

print("=" * 100)
print(f"\n统计：")
print(f"  NULL 值（Python None）: {null_count} 条")
print(f"  字符串 'None': {none_count} 条")
print(f"  有效值: {valid_count} 条")
print("=" * 100)

# 查询所有唯一的昵称值（包括NULL和'None'）
cursor.execute("""
    SELECT DISTINCT nickname
    FROM history_records
    WHERE nickname IS NULL OR nickname = 'None' OR nickname = 'none'
    LIMIT 10
""")

print("\n所有 NULL 或 'None' 的昵称值：")
for row in cursor.fetchall():
    nickname = row[0]
    print(f"  值: {repr(nickname)}, 类型: {type(nickname).__name__}")

# 查询所有唯一的用户ID值（包括NULL和'None'）
cursor.execute("""
    SELECT DISTINCT user_id
    FROM history_records
    WHERE user_id IS NULL OR user_id = 'None' OR user_id = 'none'
    LIMIT 10
""")

print("\n所有 NULL 或 'None' 的用户ID值：")
for row in cursor.fetchall():
    user_id = row[0]
    print(f"  值: {repr(user_id)}, 类型: {type(user_id).__name__}")

conn.close()
