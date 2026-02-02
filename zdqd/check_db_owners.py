"""检查数据库中的管理员情况"""
from src.local_db import LocalDatabase
from src.user_manager import UserManager
from pathlib import Path

# 1. 检查数据库中有管理员的账号
db = LocalDatabase()
db_owners = db.get_all_account_owners()
print(f"数据库中有管理员的账号: {len(db_owners)} 个")
for phone, owner in list(db_owners.items())[:10]:
    print(f"  {phone} -> {owner}")
if len(db_owners) > 10:
    print(f"  ... 还有 {len(db_owners) - 10} 个")
print()

# 2. 检查数据库中所有账号
summaries = db.get_all_accounts_summary(limit=10000)
print(f"数据库中总账号数: {len(summaries)} 个")
print()

# 3. 检查账号文件中的账号
from src.config import ConfigLoader
config = ConfigLoader().load()
accounts_file = config.accounts_file

if accounts_file and Path(accounts_file).exists():
    all_phones = []
    with open(accounts_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and '----' in line:
                phone = line.split('----')[0].strip()
                all_phones.append(phone)
    
    print(f"账号文件中的账号: {len(all_phones)} 个")
    print()
    
    # 4. 检查未分配的账号
    um = UserManager()
    unassigned = um.get_unassigned_accounts(all_phones)
    print(f"未分配的账号: {len(unassigned)} 个")
    
    # 显示前10个未分配账号
    for phone in unassigned[:10]:
        print(f"  {phone}")
    if len(unassigned) > 10:
        print(f"  ... 还有 {len(unassigned) - 10} 个")
    print()
    
    # 5. 分析差异
    print("=" * 60)
    print("数据分析:")
    print("=" * 60)
    print(f"账号文件中的账号: {len(all_phones)}")
    print(f"数据库中有记录的账号: {len(summaries)}")
    print(f"数据库中有管理员的账号: {len(db_owners)}")
    print(f"未分配的账号: {len(unassigned)}")
    print()
    print("说明:")
    print("- 账号文件：所有配置的账号")
    print("- 数据库记录：运行过的账号（有历史数据）")
    print("- 有管理员：数据库中 owner 字段不为空的账号")
    print("- 未分配：账号文件中有，但没有分配给任何用户的账号")
else:
    print("未配置账号文件")
