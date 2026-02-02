import sys
sys.path.insert(0, 'src')
from local_db import LocalDatabase

db = LocalDatabase()
summaries = db.get_all_accounts_summary(limit=10)

print("检查数据库读取顺序：")
print("="*80)
for s in summaries:
    print(f"手机号: {s['phone']:<15} 昵称: {s['nickname']:<20} 用户ID: {s['user_id']}")
