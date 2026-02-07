"""检查今天的签到记录"""
import sys
sys.path.insert(0, 'src')

from local_db import LocalDatabase
from datetime import datetime

db = LocalDatabase()
conn = db._get_connection()
cursor = conn.cursor()

today = datetime.now().strftime('%Y-%m-%d')

cursor.execute("""
    SELECT phone, nickname, balance_before, balance_after, 
           checkin_reward, checkin_total_times, status, duration
    FROM history_records 
    WHERE run_date = ? 
    ORDER BY phone 
    LIMIT 30
""", (today,))

rows = cursor.fetchall()
conn.close()

print(f"\n今天的签到记录 ({today}):")
print("=" * 120)
print(f"{'手机号':<15} {'昵称':<12} {'余额前':<10} {'余额后':<10} {'签到奖励':<10} {'总次数':<8} {'耗时':<8} {'状态':<15}")
print("=" * 120)

for row in rows:
    phone, nickname, balance_before, balance_after, checkin_reward, total_times, status, duration = row
    print(f"{phone:<15} {nickname:<12} {balance_before:<10.2f} {balance_after:<10.2f} {checkin_reward:<10.2f} {total_times if total_times else 'N/A':<8} {duration:<8.1f} {status:<15}")

print("=" * 120)
print(f"总记录数: {len(rows)}")
