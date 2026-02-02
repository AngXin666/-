import sys
import sqlite3
sys.path.insert(0, 'src')
from local_db import LocalDatabase

db = LocalDatabase()

# 直接查询数据库
conn = sqlite3.connect(str(db.db_path))
cursor = conn.cursor()

# 查询几个账号的最新记录
cursor.execute("""
    SELECT phone, nickname, user_id, run_date
    FROM history_records
    WHERE phone IN ('13247351660', '15118671290', '15201762883')
    ORDER BY phone, created_at DESC
""")

rows = cursor.fetchall()
conn.close()

print("数据库原始数据：")
print("="*100)
print(f"{'手机号':<15} {'昵称':<20} {'用户ID':<20} {'日期':<15}")
print("="*100)

current_phone = None
for row in rows:
    phone, nickname, user_id, run_date = row
    if phone != current_phone:
        print(f"{phone:<15} {str(nickname):<20} {str(user_id):<20} {run_date:<15}")
        current_phone = phone
