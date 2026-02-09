import sqlite3

conn = sqlite3.connect('runtime_data/license.db')
cursor = conn.cursor()

cursor.execute('''
    SELECT phone, balance_before, checkin_balance_after, balance_after, checkin_reward 
    FROM history_records 
    WHERE run_date='2026-02-09' 
    AND balance_before IS NOT NULL 
    AND checkin_balance_after IS NOT NULL
    ORDER BY created_at DESC 
    LIMIT 10
''')

print("手机号 | 余额前 | 签到后 | 余额后 | 记录的奖励 | 应该的奖励")
print("-" * 80)
for row in cursor.fetchall():
    phone, before, checkin_after, after, reward = row
    should_be = round(checkin_after - before, 2) if checkin_after and before else 0
    print(f"{phone} | {before} | {checkin_after} | {after} | {reward} | {should_be}")

conn.close()
