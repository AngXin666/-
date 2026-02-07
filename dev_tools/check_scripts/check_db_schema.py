"""检查数据库表结构"""
import sys
sys.path.insert(0, 'src')

from local_db import LocalDatabase
import sqlite3

db = LocalDatabase()

conn = db._get_connection()
cursor = conn.cursor()

# 查看表结构
cursor.execute("PRAGMA table_info(history_records)")
columns = cursor.fetchall()

print("=" * 60)
print("history_records 表结构")
print("=" * 60)
for col in columns:
    cid, name, type_, notnull, default, pk = col
    print(f"{cid:2d}. {name:30s} {type_:15s} {'NOT NULL' if notnull else ''} {'PK' if pk else ''}")

print()
print("=" * 60)
print("查看一条完整记录")
print("=" * 60)

cursor.execute("""
    SELECT * FROM history_records 
    WHERE run_date = '2026-02-01'
    LIMIT 1
""")

row = cursor.fetchone()
if row:
    for i, col in enumerate(columns):
        col_name = col[1]
        value = row[i]
        print(f"{col_name:30s} = {value}")

conn.close()
