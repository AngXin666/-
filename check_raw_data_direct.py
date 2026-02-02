import sqlite3
from pathlib import Path

db_path = Path("runtime_data") / "license.db"
conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

# 直接查询几个账号的最新记录
cursor.execute("""
    SELECT phone, nickname, user_id
    FROM history_records
    WHERE phone IN ('15118671290', '15201762883', '15766121960')
    ORDER BY phone, created_at DESC
""")

rows = cursor.fetchall()
conn.close()

print("数据库原始数据（直接查询）：")
print("="*80)
print(f"{'手机号':<15} {'昵称':<25} {'用户ID':<25}")
print("="*80)

current_phone = None
for row in rows:
    phone, nickname, user_id = row
    if phone != current_phone:
        print(f"{phone:<15} {str(nickname):<25} {str(user_id):<25}")
        current_phone = phone
