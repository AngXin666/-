"""检查账号数量统计"""
from src.local_db import LocalDatabase

db = LocalDatabase()
conn = db._get_connection()
cursor = conn.cursor()

# 总账号数
cursor.execute('SELECT COUNT(DISTINCT phone) FROM history_records')
total = cursor.fetchone()[0]

# 待处理账号数
cursor.execute('SELECT COUNT(DISTINCT phone) FROM history_records WHERE status IN ("", "待处理")')
pending = cursor.fetchone()[0]

# 未使用账号数（三个字段都是0）
cursor.execute('''
    SELECT COUNT(DISTINCT h1.phone)
    FROM history_records h1
    INNER JOIN (
        SELECT phone, MAX(created_at) as max_created_at
        FROM history_records
        GROUP BY phone
    ) h2 ON h1.phone = h2.phone AND h1.created_at = h2.max_created_at
    WHERE (h1.balance_before = 0 OR h1.balance_before IS NULL)
      AND (h1.checkin_total_times = 0 OR h1.checkin_total_times IS NULL)
      AND (h1.balance_after = 0 OR h1.balance_after IS NULL)
      AND h1.phone NOT IN (
          SELECT DISTINCT phone 
          FROM history_records 
          WHERE checkin_total_times > 0
      )
''')
unused = cursor.fetchone()[0]

print(f'数据库总账号数: {total}')
print(f'待处理账号数: {pending}')
print(f'未使用账号数: {unused}')

conn.close()
