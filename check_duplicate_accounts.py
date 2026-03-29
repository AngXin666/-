"""检查重复账号"""
from src.local_db import LocalDatabase

db = LocalDatabase()
conn = db._get_connection()
cursor = conn.cursor()

# 检查重复的手机号
cursor.execute('''
    SELECT phone, COUNT(*) as cnt 
    FROM history_records 
    GROUP BY phone 
    HAVING cnt > 1
''')
duplicates = cursor.fetchall()

print(f'重复的手机号数量: {len(duplicates)}')
if duplicates:
    print('\n前10个重复的手机号:')
    for phone, cnt in duplicates[:10]:
        print(f'  {phone}: {cnt}条记录')

# 统计总的不同手机号数量
cursor.execute('SELECT COUNT(DISTINCT phone) FROM history_records')
total_unique = cursor.fetchone()[0]

# 统计总记录数
cursor.execute('SELECT COUNT(*) FROM history_records')
total_records = cursor.fetchone()[0]

print(f'\n总记录数: {total_records}')
print(f'不同手机号数量: {total_unique}')
print(f'平均每个手机号的记录数: {total_records / total_unique:.2f}')

conn.close()
