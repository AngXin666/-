"""检查缺失账号的状态"""
from src.local_db import LocalDatabase

db = LocalDatabase()
conn = db._get_connection()
cursor = conn.cursor()

missing_phones = ['15265000126', '18878357377']

print("检查缺失账号的数据库记录:")
print("=" * 60)

for phone in missing_phones:
    print(f"\n手机号: {phone}")
    
    # 检查是否有任何历史记录
    cursor.execute('''
        SELECT phone, run_date, status, nickname, created_at
        FROM history_records 
        WHERE phone = ?
        ORDER BY created_at DESC
    ''', (phone,))
    records = cursor.fetchall()
    
    if records:
        print(f"  找到 {len(records)} 条历史记录:")
        for phone, run_date, status, nickname, created_at in records:
            print(f"    - 日期: {run_date}, 状态: {status}, 昵称: {nickname or '未知'}, 创建时间: {created_at}")
    else:
        print(f"  ❌ 数据库中没有任何记录")

print("\n" + "=" * 60)
print("结论:")
print("这2个账号在账号文件中，但从未成功导入到数据库。")
print("可能原因:")
print("1. 添加账号后没有运行过程序")
print("2. 运行时这2个账号处理失败，没有保存到数据库")
print("3. 账号信息有误，导致无法处理")

conn.close()
