"""
直接查询数据库验证签到奖励是否已修复
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3

# 连接数据库
db_path = "runtime_data/license.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 查询最近10条记录
cursor.execute("""
    SELECT phone, run_date, balance_before, balance_after, checkin_reward
    FROM history_records
    ORDER BY run_date DESC, created_at DESC
    LIMIT 10
""")

print("=" * 80)
print("数据库中最近10条记录：")
print("=" * 80)
print(f"{'手机号':<15} {'日期':<12} {'余额前':<10} {'余额后':<10} {'签到奖励':<10}")
print("-" * 80)

for row in cursor.fetchall():
    phone, run_date, balance_before, balance_after, checkin_reward = row
    balance_before = balance_before if balance_before is not None else 0.0
    balance_after = balance_after if balance_after is not None else 0.0
    checkin_reward = checkin_reward if checkin_reward is not None else 0.0
    print(f"{phone:<15} {run_date:<12} {balance_before:<10.2f} {balance_after:<10.2f} {checkin_reward:<10.2f}")

conn.close()
print("=" * 80)
