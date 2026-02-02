import sqlite3
from pathlib import Path

db_path = Path("runtime_data") / "license.db"
conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

# 查询那些昵称为'-'但用户ID看起来像昵称的记录
cursor.execute("""
    SELECT phone, nickname, user_id, run_date, created_at
    FROM history_records
    WHERE nickname = '-' AND user_id NOT LIKE '%[0-9]%'
    ORDER BY created_at DESC
    LIMIT 20
""")

rows = cursor.fetchall()
conn.close()

print("昵称为'-'且用户ID看起来像昵称的记录：")
print("="*100)
print(f"{'手机号':<15} {'昵称':<20} {'用户ID':<20} {'日期':<15} {'创建时间':<20}")
print("="*100)

for row in rows:
    phone, nickname, user_id, run_date, created_at = row
    print(f"{phone:<15} {str(nickname):<20} {str(user_id):<20} {run_date:<15} {created_at:<20}")
