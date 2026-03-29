"""查找缺失的2个账号"""
from src.local_db import LocalDatabase
from datetime import datetime

db = LocalDatabase()
conn = db._get_connection()
cursor = conn.cursor()

# 获取今天的日期
today = datetime.now().strftime('%Y-%m-%d')
print(f"检查日期: {today}")

# 获取今天所有新增的账号（按创建时间排序）
cursor.execute('''
    SELECT phone, created_at, status, nickname
    FROM history_records 
    WHERE DATE(created_at) = ?
    ORDER BY created_at ASC
''', (today,))
today_records = cursor.fetchall()

print(f"\n今天新增的账号数量: {len(today_records)}")
print(f"应该有: 38 个")
print(f"实际有: {len(today_records)} 个")
print(f"缺失: {38 - len(today_records)} 个")

print("\n今天新增的所有账号:")
for i, (phone, created_at, status, nickname) in enumerate(today_records, 1):
    print(f"{i:2d}. {phone} - {created_at} - {status} - {nickname or '未知'}")

# 检查是否有导入失败的记录（状态为失败）
cursor.execute('''
    SELECT phone, created_at, status, nickname
    FROM history_records 
    WHERE DATE(created_at) = ? AND status LIKE '%失败%'
    ORDER BY created_at ASC
''', (today,))
failed_records = cursor.fetchall()

if failed_records:
    print(f"\n今天导入失败的账号:")
    for phone, created_at, status, nickname in failed_records:
        print(f"  {phone} - {status}")

# 检查数据库中所有不同的手机号
cursor.execute('SELECT COUNT(DISTINCT phone) FROM history_records')
total_unique = cursor.fetchone()[0]
print(f"\n数据库中不同手机号总数: {total_unique}")

# 检查是否有重复导入的情况（今天同一个手机号多次导入）
cursor.execute('''
    SELECT phone, COUNT(*) as cnt
    FROM history_records 
    WHERE DATE(created_at) = ?
    GROUP BY phone
    HAVING cnt > 1
''', (today,))
duplicates = cursor.fetchall()

if duplicates:
    print(f"\n今天重复导入的账号:")
    for phone, cnt in duplicates:
        print(f"  {phone} - 导入了 {cnt} 次")

conn.close()
