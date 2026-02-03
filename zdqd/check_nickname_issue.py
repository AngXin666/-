"""检查昵称识别异常问题"""
import sqlite3
import os

db_path = 'runtime_data/license.db'

if not os.path.exists(db_path):
    print(f"数据库文件不存在: {db_path}")
    exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 查看所有表
print("=" * 80)
print("数据库表:")
print("=" * 80)
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
for table in tables:
    print(f"  - {table[0]}")

print("\n" + "=" * 80)
print("检查账号表结构:")
print("=" * 80)

# 查看license表结构（这是实际的用户表）
cursor.execute("PRAGMA table_info(license)")
columns = cursor.fetchall()
print("\nlicense表字段:")
for col in columns:
    print(f"  {col[1]} ({col[2]})")

# 查询最近的账号数据（包含昵称）
print("\n" + "=" * 80)
print("所有账号的昵称数据:")
print("=" * 80)
print(f"{'手机号':<15} {'昵称':<20} {'用户ID':<10}")
print("-" * 80)

cursor.execute("""
    SELECT phone, nickname, user_id 
    FROM license 
    LIMIT 200
""")
rows = cursor.fetchall()

for row in rows:
    phone = row[0] or 'N/A'
    nickname = row[1] or 'N/A'
    user_id = row[2] or 'N/A'
    print(f"{phone:<15} {nickname:<20} {user_id:<10}")

# 统计昵称异常的账号
print("\n" + "=" * 80)
print("昵称异常统计:")
print("=" * 80)

cursor.execute("SELECT COUNT(*) FROM license WHERE nickname IS NULL OR nickname = ''")
null_count = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM license WHERE nickname IS NOT NULL AND nickname != ''")
valid_count = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM license")
total_count = cursor.fetchone()[0]

print(f"总账号数: {total_count}")
print(f"有效昵称: {valid_count}")
print(f"空昵称: {null_count}")
if total_count > 0:
    print(f"异常比例: {null_count/total_count*100:.1f}%")

# 查看一些异常昵称的例子
print("\n" + "=" * 80)
print("异常昵称示例（长度<=3或包含特殊字符）:")
print("=" * 80)
print(f"{'手机号':<15} {'昵称':<20} {'用户ID':<10}")
print("-" * 80)

cursor.execute("""
    SELECT phone, nickname, user_id 
    FROM license 
    WHERE nickname IS NOT NULL 
    AND nickname != ''
    AND LENGTH(nickname) <= 6
    LIMIT 50
""")
rows = cursor.fetchall()

for row in rows:
    phone = row[0] or 'N/A'
    nickname = row[1] or 'N/A'
    user_id = row[2] or 'N/A'
    print(f"{phone:<15} {nickname:<20} {user_id:<10}")

conn.close()
