from src.local_db import LocalDatabase
from src.user_manager import UserManager

db = LocalDatabase()
um = UserManager()

# 获取所有账号
all_summaries = db.get_all_accounts_summary(limit=100)

# 获取已分配的账号
assigned = set()
for uid in um.users.keys():
    assigned.update(um.get_user_accounts(uid))

# 筛选未分配的账号
unassigned = [s for s in all_summaries if s['phone'] not in assigned]

print(f'未分配账号: {len(unassigned)} 个')
for s in unassigned[:10]:
    print(f"  {s['phone']} - {s.get('nickname', '-')}")
