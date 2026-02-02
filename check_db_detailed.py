"""详细检查数据库"""
import sys
sys.path.insert(0, 'src')

from local_db import LocalDatabase
from datetime import datetime
import sqlite3

db = LocalDatabase()

# 1. 检查数据库文件
print("=" * 60)
print("1. 数据库文件检查")
print("=" * 60)
print(f"数据库路径: {db.db_path}")
print()

# 2. 直接查询所有记录（不过滤日期）
print("=" * 60)
print("2. 查询所有历史记录")
print("=" * 60)
conn = db._get_connection()
cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM history_records")
total_count = cursor.fetchone()[0]
print(f"总记录数: {total_count}")
print()

# 3. 查询最近10条记录
print("=" * 60)
print("3. 最近10条记录")
print("=" * 60)
cursor.execute("""
    SELECT phone, run_date, status, checkin_reward, balance_before, balance_after
    FROM history_records
    ORDER BY run_date DESC, id DESC
    LIMIT 10
""")

records = cursor.fetchall()
if records:
    for r in records:
        phone, run_date, status, checkin_reward, balance_before, balance_after = r
        print(f"  手机号: {phone}")
        print(f"    日期: {run_date}")
        print(f"    状态: {status}")
        print(f"    签到奖励: {checkin_reward}")
        print(f"    余额前: {balance_before}")
        print(f"    余额后: {balance_after}")
        print()
else:
    print("  没有记录")
print()

# 4. 查询今天的记录（使用不同的方式）
print("=" * 60)
print("4. 查询今天的记录（多种方式）")
print("=" * 60)

today = datetime.now().strftime('%Y-%m-%d')
print(f"今天日期: {today}")
print()

# 方式1: 精确匹配
cursor.execute("SELECT COUNT(*) FROM history_records WHERE run_date = ?", (today,))
count1 = cursor.fetchone()[0]
print(f"方式1（精确匹配）: {count1} 条")

# 方式2: LIKE匹配
cursor.execute("SELECT COUNT(*) FROM history_records WHERE run_date LIKE ?", (f"{today}%",))
count2 = cursor.fetchone()[0]
print(f"方式2（LIKE匹配）: {count2} 条")

# 方式3: 查看所有不同的日期
cursor.execute("SELECT DISTINCT run_date FROM history_records ORDER BY run_date DESC LIMIT 10")
dates = cursor.fetchall()
print(f"\n最近的日期:")
for d in dates:
    print(f"  - {d[0]}")

conn.close()

# 5. 使用LocalDatabase的方法查询
print()
print("=" * 60)
print("5. 使用LocalDatabase.get_history_records()查询")
print("=" * 60)
records = db.get_history_records(start_date=today, end_date=today)
print(f"返回记录数: {len(records)}")

if records:
    for r in records:
        print(f"  手机号: {r.get('phone')}")
        print(f"    状态: {r.get('status')}")
        print(f"    签到奖励: {r.get('checkin_reward')}")
        print()
