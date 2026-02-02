"""检查特定手机号的记录"""
import sys
sys.path.insert(0, 'src')

from local_db import LocalDatabase

# 初始化数据库
db = LocalDatabase()

# 用户截图中显示的5个账号
phones = [
    '16762954669',
    '17716779110',
    '18122173880',
    '18071306515',
    '18007200308'
]

print("=" * 60)
print("检查用户截图中的5个账号")
print("=" * 60)

for phone in phones:
    print(f"\n手机号: {phone}")
    records = db.get_history_records(phone=phone, limit=10)
    
    if records:
        print(f"  找到 {len(records)} 条记录:")
        for i, record in enumerate(records, 1):
            print(f"  {i}. 运行日期: {record.get('run_date')}")
            print(f"     创建时间: {record.get('created_at')}")
            print(f"     状态: {record.get('status')}")
            print(f"     签到奖励: {record.get('checkin_reward')}")
    else:
        print("  ❌ 没有找到记录！")

print("\n" + "=" * 60)
