"""检查所有账号的最新记录日期"""
import sys
sys.path.insert(0, 'src')

from local_db import LocalDatabase
from datetime import datetime

# 初始化数据库
db = LocalDatabase()

print("=" * 60)
print("检查所有账号的最新记录")
print("=" * 60)

# 获取所有记录（不限制日期）
conn = db._get_connection()
cursor = conn.cursor()

# 查询每个账号的最新记录
cursor.execute("""
    SELECT phone, MAX(run_date) as latest_date, MAX(created_at) as latest_created
    FROM history_records
    WHERE status LIKE '%成功%' OR status LIKE '%失败%'
    GROUP BY phone
    ORDER BY latest_date DESC, phone
""")

rows = cursor.fetchall()
conn.close()

today = datetime.now().strftime('%Y-%m-%d')
yesterday = '2026-01-30'

today_count = 0
yesterday_count = 0
older_count = 0

print(f"\n今天日期: {today}")
print(f"昨天日期: {yesterday}\n")

print("最新记录日期统计:")
for phone, latest_date, latest_created in rows:
    if latest_date == today:
        today_count += 1
    elif latest_date == yesterday:
        yesterday_count += 1
    else:
        older_count += 1

print(f"  今天({today}): {today_count} 个账号")
print(f"  昨天({yesterday}): {yesterday_count} 个账号")
print(f"  更早: {older_count} 个账号")
print(f"  总计: {len(rows)} 个账号")

print(f"\n昨天有记录但今天没有更新的账号:")
no_update_count = 0
for phone, latest_date, latest_created in rows:
    if latest_date == yesterday:
        no_update_count += 1
        print(f"  {no_update_count}. {phone} - 最新: {latest_date} (创建于: {latest_created})")

print(f"\n总共 {no_update_count} 个账号今天没有更新记录")
print("=" * 60)
