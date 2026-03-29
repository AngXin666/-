"""检查最近添加的账号"""
from src.local_db import LocalDatabase
from datetime import datetime, timedelta

db = LocalDatabase()
conn = db._get_connection()
cursor = conn.cursor()

# 获取今天的日期
today = datetime.now().strftime('%Y-%m-%d')
print(f"今天日期: {today}")

# 检查今天添加的账号（按创建时间）
cursor.execute('''
    SELECT phone, created_at, run_date
    FROM history_records 
    WHERE DATE(created_at) = ?
    ORDER BY created_at DESC
''', (today,))
today_records = cursor.fetchall()

print(f"\n今天添加的记录数: {len(today_records)}")
if today_records:
    print("\n最近10条记录:")
    for phone, created_at, run_date in today_records[:10]:
        print(f"  {phone} - 创建时间: {created_at}, 运行日期: {run_date}")

# 检查最近24小时内添加的记录
yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')
cursor.execute('''
    SELECT phone, created_at, run_date
    FROM history_records 
    WHERE created_at >= ?
    ORDER BY created_at DESC
''', (yesterday,))
recent_records = cursor.fetchall()

print(f"\n最近24小时内添加的记录数: {len(recent_records)}")

# 统计不同手机号数量
cursor.execute('SELECT COUNT(DISTINCT phone) FROM history_records')
total_unique = cursor.fetchone()[0]
print(f"\n数据库中不同手机号总数: {total_unique}")

# 检查是否有重复的手机号（同一天多条记录）
cursor.execute('''
    SELECT phone, run_date, COUNT(*) as cnt 
    FROM history_records 
    WHERE run_date = ?
    GROUP BY phone, run_date 
    HAVING cnt > 1
''', (today,))
duplicates_today = cursor.fetchall()

if duplicates_today:
    print(f"\n今天有重复记录的手机号数量: {len(duplicates_today)}")
    print("前5个重复的手机号:")
    for phone, run_date, cnt in duplicates_today[:5]:
        print(f"  {phone} - 日期: {run_date}, 记录数: {cnt}")

conn.close()
