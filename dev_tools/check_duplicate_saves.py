"""检查是否有重复保存导致签到奖励累加"""
import sqlite3
import sys
sys.path.insert(0, '.')

# 连接数据库
conn = sqlite3.connect('runtime_data/license.db')
cursor = conn.cursor()

# 查询今天的记录，按创建时间排序
cursor.execute('''
    SELECT phone, nickname, balance_before, checkin_reward, balance_after, 
           run_date, created_at, id
    FROM history_records
    WHERE phone = "19120553831" AND run_date = date('now')
    ORDER BY created_at DESC
''')

records = cursor.fetchall()

print("=" * 100)
print(f"今天的记录（手机号: 19120553831）- 共 {len(records)} 条")
print("=" * 100)

if len(records) > 1:
    print("⚠️ 发现多条记录！可能存在重复保存问题")
    print()

for i, record in enumerate(records, 1):
    phone, nickname, balance_before, checkin_reward, balance_after, run_date, created_at, record_id = record
    print(f"记录 {i} (ID: {record_id}):")
    print(f"  创建时间: {created_at}")
    print(f"  余额前: {balance_before:.2f} 元")
    print(f"  签到奖励: {checkin_reward:.2f} 元")
    print(f"  余额后: {balance_after:.2f} 元")
    print(f"  计算值: {balance_after - balance_before:.2f} 元 (余额后 - 余额前)")
    print()

# 检查是否有重复保存
if len(records) == 1:
    record = records[0]
    phone, nickname, balance_before, checkin_reward, balance_after, run_date, created_at, record_id = record
    expected_reward = balance_after - balance_before
    
    if abs(checkin_reward - expected_reward) > 0.01:
        print("=" * 100)
        print("⚠️ 签到奖励不匹配分析")
        print("=" * 100)
        print(f"数据库中的签到奖励: {checkin_reward:.2f} 元")
        print(f"计算值 (余额后 - 余额前): {expected_reward:.2f} 元")
        print(f"差异: {abs(checkin_reward - expected_reward):.2f} 元")
        print()
        
        # 计算可能的保存次数
        if expected_reward > 0:
            save_count = round(checkin_reward / expected_reward)
            print(f"推测保存次数: {save_count} 次")
            print(f"  (签到奖励 {checkin_reward:.2f} ÷ 计算值 {expected_reward:.2f} ≈ {save_count})")
        print("=" * 100)

conn.close()
