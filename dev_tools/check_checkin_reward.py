"""检查签到奖励数据"""
import sqlite3
import sys
sys.path.insert(0, '.')

# 连接数据库
conn = sqlite3.connect('runtime_data/license.db')
cursor = conn.cursor()

# 查询最近的记录
cursor.execute('''
    SELECT phone, nickname, balance_before, checkin_reward, balance_after, run_date, created_at
    FROM history_records
    WHERE phone = "19120553831"
    ORDER BY created_at DESC
    LIMIT 5
''')

records = cursor.fetchall()

print("=" * 80)
print("最近5条记录（手机号: 19120553831）")
print("=" * 80)
print(f"{'手机号':<15} {'昵称':<10} {'余额前':<10} {'签到奖励':<10} {'余额后':<10} {'日期':<12} {'创建时间':<20}")
print("-" * 80)

for record in records:
    phone, nickname, balance_before, checkin_reward, balance_after, run_date, created_at = record
    print(f"{phone:<15} {nickname:<10} {balance_before:<10.2f} {checkin_reward:<10.2f} {balance_after:<10.2f} {run_date:<12} {created_at:<20}")

print("=" * 80)

# 检查今天的记录
cursor.execute('''
    SELECT phone, nickname, balance_before, checkin_reward, balance_after, run_date
    FROM history_records
    WHERE phone = "19120553831" AND run_date = date('now')
''')

today_record = cursor.fetchone()

if today_record:
    phone, nickname, balance_before, checkin_reward, balance_after, run_date = today_record
    print(f"\n今天的记录:")
    print(f"  手机号: {phone}")
    print(f"  昵称: {nickname}")
    print(f"  余额前: {balance_before:.2f} 元")
    print(f"  签到奖励: {checkin_reward:.2f} 元")
    print(f"  余额后: {balance_after:.2f} 元")
    print(f"  日期: {run_date}")
    
    # 计算预期的签到奖励
    if balance_before is not None and balance_after is not None:
        expected_reward = balance_after - balance_before
        print(f"\n  预期签到奖励: {expected_reward:.2f} 元 (余额后 - 余额前)")
        
        if abs(checkin_reward - expected_reward) > 0.01:
            print(f"  ⚠️ 签到奖励不匹配！")
            print(f"     数据库中: {checkin_reward:.2f} 元")
            print(f"     计算值: {expected_reward:.2f} 元")
            print(f"     差异: {abs(checkin_reward - expected_reward):.2f} 元")
else:
    print("\n今天没有记录")

conn.close()
