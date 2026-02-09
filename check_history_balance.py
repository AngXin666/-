"""检查账号的历史余额记录"""
import sqlite3
from pathlib import Path

db_path = Path("runtime_data/license.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 查询用户 1875608 的历史记录
cursor.execute("""
    SELECT run_date, balance_before, checkin_reward, balance_after, transfer_amount, created_at
    FROM history_records
    WHERE user_id = '1875608'
    ORDER BY created_at DESC
    LIMIT 10
""")

rows = cursor.fetchall()

print("="*120)
print("用户 1875608 (19050647048) 的历史记录")
print("="*120)
print(f"{'日期':<12} {'余额前':<10} {'签到奖励':<12} {'余额后':<10} {'转账金额':<12} {'创建时间':<20}")
print("-"*120)

for row in rows:
    run_date, balance_before, checkin_reward, balance_after, transfer_amount, created_at = row
    balance_before = balance_before if balance_before is not None else 0.0
    checkin_reward = checkin_reward if checkin_reward is not None else 0.0
    balance_after = balance_after if balance_after is not None else 0.0
    transfer_amount = transfer_amount if transfer_amount is not None else 0.0
    print(f"{run_date:<12} {balance_before:<10.2f} {checkin_reward:<12.2f} {balance_after:<10.2f} {transfer_amount:<12.2f} {created_at:<20}")

conn.close()
