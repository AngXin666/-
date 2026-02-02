import sqlite3
from pathlib import Path

db_path = Path("runtime_data") / "license.db"
conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

# 查询最近的10条记录
cursor.execute("""
    SELECT phone, nickname, user_id, run_date, created_at
    FROM history_records
    ORDER BY created_at DESC
    LIMIT 10
""")

rows = cursor.fetchall()
conn.close()

print("最近的10条记录：")
print("="*100)
print(f"{'手机号':<15} {'昵称':<20} {'用户ID':<20} {'日期':<15} {'创建时间':<20}")
print("="*100)

for row in rows:
    phone, nickname, user_id, run_date, created_at = row
    print(f"{phone:<15} {str(nickname):<20} {str(user_id):<20} {run_date:<15} {created_at:<20}")
