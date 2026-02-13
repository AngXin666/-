"""检查今天和昨天的转账对比"""
from src.local_db import LocalDatabase
from datetime import datetime, timedelta

db = LocalDatabase()
today = datetime.now().strftime('%Y-%m-%d')
yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

print("=" * 60)
print("今天 vs 昨天 转账对比分析")
print("=" * 60)

# 获取记录
today_records = db.get_history_records(start_date=today, end_date=today, limit=1000)
yesterday_records = db.get_history_records(start_date=yesterday, end_date=yesterday, limit=1000)

print(f"\n今天日期: {today}")
print(f"昨天日期: {yesterday}")
print(f"\n今天记录数: {len(today_records)}")
print(f"昨天记录数: {len(yesterday_records)}")

# 统计转账
today_with_transfer = [r for r in today_records if r.get('transfer_amount') and float(r.get('transfer_amount', 0)) > 0]
yesterday_with_transfer = [r for r in yesterday_records if r.get('transfer_amount') and float(r.get('transfer_amount', 0)) > 0]

today_total = sum([float(r.get('transfer_amount', 0)) for r in today_with_transfer])
yesterday_total = sum([float(r.get('transfer_amount', 0)) for r in yesterday_with_transfer])

print(f"\n今天有转账的账号数: {len(today_with_transfer)}")
print(f"昨天有转账的账号数: {len(yesterday_with_transfer)}")

print(f"\n今天总转账金额: {today_total:.2f} 元")
print(f"昨天总转账金额: {yesterday_total:.2f} 元")
print(f"差额: {today_total - yesterday_total:.2f} 元")

if today_with_transfer:
    print(f"\n今天平均转账金额: {today_total / len(today_with_transfer):.2f} 元")
if yesterday_with_transfer:
    print(f"昨天平均转账金额: {yesterday_total / len(yesterday_with_transfer):.2f} 元")

# 显示今天转账金额最大的前10个账号
print("\n今天转账金额最大的前10个账号:")
today_sorted = sorted(today_with_transfer, key=lambda x: float(x.get('transfer_amount', 0)), reverse=True)
for i, r in enumerate(today_sorted[:10], 1):
    print(f"  {i}. {r.get('phone')}: {float(r.get('transfer_amount', 0)):.2f} 元 → {r.get('transfer_recipient', 'N/A')}")

print("\n昨天转账金额最大的前10个账号:")
yesterday_sorted = sorted(yesterday_with_transfer, key=lambda x: float(x.get('transfer_amount', 0)), reverse=True)
for i, r in enumerate(yesterday_sorted[:10], 1):
    print(f"  {i}. {r.get('phone')}: {float(r.get('transfer_amount', 0)):.2f} 元 → {r.get('transfer_recipient', 'N/A')}")

print("\n" + "=" * 60)
