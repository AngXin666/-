"""检查今天的转账记录"""
from src.local_db import LocalDatabase
from datetime import datetime

db = LocalDatabase()
records = db.get_all_history_records()

today = datetime.now().strftime('%Y-%m-%d')
today_records = [r for r in records if r['运行日期'] == today]

print(f"\n今天的记录数: {len(today_records)}\n")
print("=" * 80)
print(f"{'手机号':<15} {'转账金额':<10} {'收款人':<15} {'余额':<10}")
print("=" * 80)

for r in today_records[:10]:
    transfer_amount = r.get('转账金额(元)', 0)
    transfer_recipient = r.get('转账收款人', '')
    balance = r.get('余额(元)', '-')
    
    print(f"{r['手机号']:<15} {transfer_amount:<10} {transfer_recipient:<15} {balance:<10}")

print("=" * 80)
