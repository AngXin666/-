"""验证修复结果"""
import sys
sys.path.insert(0, 'src')
from local_db import LocalDatabase

db = LocalDatabase()

print("=" * 80)
print("验证修复结果")
print("=" * 80)

# 1. 检查之前有问题的账号
problem_phones = [
    '13247351660', '13307857120', '13322736481', '15201762883', 
    '15766121960', '15873379556', '18692238332', '19065355068'
]

print("\n1. 检查之前有问题的账号:")
print("-" * 80)
print(f"{'手机号':<15} {'昵称':<20} {'用户ID':<20} {'状态':<10}")
print("-" * 80)

all_correct = True
for phone in problem_phones:
    summary = db.get_account_summary(phone)
    if summary:
        nickname = summary.get('nickname', '-') or '-'
        user_id = summary.get('user_id', '-') or '-'
        
        # 检查是否还有问题（昵称为'-'且用户ID看起来像昵称）
        is_problem = (nickname == '-' and not user_id.isdigit())
        status = "❌ 仍有问题" if is_problem else "✅ 正常"
        
        if is_problem:
            all_correct = False
        
        print(f"{phone:<15} {nickname:<20} {user_id:<20} {status:<10}")

# 2. 检查数据库中是否还有昵称为'-'的记录
import sqlite3
from pathlib import Path

db_path = Path("runtime_data") / "license.db"
conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

cursor.execute("""
    SELECT COUNT(*)
    FROM history_records
    WHERE nickname = '-'
""")

count_dash_nickname = cursor.fetchone()[0]

print(f"\n2. 数据库中昵称为'-'的记录数: {count_dash_nickname}")

# 3. 对比两种查询方法的结果
print("\n3. 对比查询方法的一致性:")
print("-" * 80)

test_phones = ['13247351660', '13307857120', '13322736481']
consistent = True

for phone in test_phones:
    # 方法1: get_account_summary
    summary1 = db.get_account_summary(phone)
    
    # 方法2: get_all_accounts_summary (找到该账号)
    all_summaries = db.get_all_accounts_summary(limit=200)
    summary2 = next((s for s in all_summaries if s['phone'] == phone), None)
    
    if summary1 and summary2:
        match = (summary1['nickname'] == summary2['nickname'] and 
                summary1['user_id'] == summary2['user_id'])
        
        status = "✅ 一致" if match else "❌ 不一致"
        if not match:
            consistent = False
        
        print(f"{phone}: {status}")
        if not match:
            print(f"  get_account_summary: 昵称={summary1['nickname']}, 用户ID={summary1['user_id']}")
            print(f"  get_all_accounts_summary: 昵称={summary2['nickname']}, 用户ID={summary2['user_id']}")

conn.close()

# 总结
print("\n" + "=" * 80)
print("验证结果总结")
print("=" * 80)

if all_correct and consistent:
    print("✅ 所有问题已修复！")
    print("  - 之前有问题的账号现在都正常了")
    print("  - 所有查询方法返回的数据一致")
else:
    print("❌ 仍有问题需要修复：")
    if not all_correct:
        print("  - 部分账号的数据仍然不正确")
    if not consistent:
        print("  - 不同查询方法返回的数据不一致")
