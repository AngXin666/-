"""检查签到金额识别准确率"""
import sys
sys.path.insert(0, 'src')

from local_db import LocalDatabase

db = LocalDatabase()
conn = db._get_connection()
cursor = conn.cursor()

# 查询最近7天的签到数据
cursor.execute('''
    SELECT 
        COUNT(*) as total,
        COUNT(CASE WHEN checkin_reward > 0 THEN 1 END) as with_amount,
        AVG(CASE WHEN checkin_reward > 0 THEN checkin_reward END) as avg_amount,
        MIN(CASE WHEN checkin_reward > 0 THEN checkin_reward END) as min_amount,
        MAX(CASE WHEN checkin_reward > 0 THEN checkin_reward END) as max_amount
    FROM history_records 
    WHERE run_date >= date('now', '-30 days')
''')

result = cursor.fetchone()
total, with_amount, avg_amount, min_amount, max_amount = result

print("=" * 60)
print("最近30天签到金额识别统计")
print("=" * 60)
print(f"总签到记录数: {total}")
print(f"识别到金额的记录: {with_amount}")
print(f"识别率: {with_amount/total*100:.1f}%" if total > 0 else "识别率: 0%")
print(f"平均金额: {avg_amount:.2f}元" if avg_amount else "平均金额: 0元")
print(f"最小金额: {min_amount:.2f}元" if min_amount else "最小金额: 0元")
print(f"最大金额: {max_amount:.2f}元" if max_amount else "最大金额: 0元")

# 查询最近20条有金额的记录
print("\n最近20条签到金额记录:")
cursor.execute('''
    SELECT phone, run_date, checkin_reward, checkin_total_times
    FROM history_records
    WHERE checkin_reward > 0
    ORDER BY run_date DESC, created_at DESC
    LIMIT 20
''')

records = cursor.fetchall()
for i, (phone, date, amount, times) in enumerate(records, 1):
    print(f"  {i}. {phone} - {date} - {amount:.2f}元 ({times}次)")

conn.close()
