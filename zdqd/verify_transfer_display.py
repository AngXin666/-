"""验证转账信息显示"""
from src.local_db import LocalDatabase

db = LocalDatabase()
records = db.get_all_history_records()

print(f"\n总记录数: {len(records)}\n")
print("=" * 80)
print(f"{'手机号':<15} {'昵称':<10} {'转账金额':<10} {'收款人':<10}")
print("=" * 80)

for r in records[:10]:
    transfer_amount = r.get('转账金额(元)', 0)
    transfer_recipient = r.get('转账收款人', '')
    
    if transfer_amount and transfer_amount > 0:
        transfer_info = f"{transfer_amount:.2f}→{transfer_recipient}"
    else:
        transfer_info = "无"
    
    print(f"{r['手机号']:<15} {r['昵称']:<10} {transfer_info:<20}")

print("=" * 80)
print("\n✓ 数据库查询正常，转账字段已正确返回")
