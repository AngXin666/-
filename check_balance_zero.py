"""检查余额为0但有签到奖励的记录"""
import sqlite3
from pathlib import Path

db_path = Path("runtime_data/license.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 查询余额为0但签到奖励不为0的记录
cursor.execute("""
    SELECT phone, user_id, balance_before, checkin_reward, balance_after, transfer_amount, run_date
    FROM history_records
    WHERE balance_after = 0 AND checkin_reward > 0
    ORDER BY created_at DESC
    LIMIT 10
""")

rows = cursor.fetchall()

print("="*100)
print("余额为0但有签到奖励的记录")
print("="*100)
print(f"{'手机号':<15} {'用户ID':<10} {'余额前':<10} {'签到奖励':<12} {'余额后':<10} {'转账金额':<12} {'日期':<12}")
print("-"*100)

for row in rows:
    phone, user_id, balance_before, checkin_reward, balance_after, transfer_amount, run_date = row
    print(f"{phone:<15} {user_id:<10} {balance_before:<10.2f} {checkin_reward:<12.2f} {balance_after:<10.2f} {transfer_amount:<12.2f} {run_date:<12}")

conn.close()
