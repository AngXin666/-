"""检查昵称识别问题"""
import sqlite3

# 连接数据库
conn = sqlite3.connect('runtime_data/license.db')
cursor = conn.cursor()

# 有问题的用户ID列表
user_ids = [
    '1881031', '1875655', '1881006', '1875500', '1875515', 
    '1881013', '1810962', '1880991', '1881015', '1579219',
    '1875494', '1875826', '1802004', '1803381', '1881036',
    '1875503', '1881024', '1880989', '1803229', '1880987', '1881011'
]

print("检查这些用户ID的正确昵称:\n")

for uid in user_ids:
    # 查找该用户的正确昵称（排除错误识别）
    cursor.execute('''
        SELECT DISTINCT nickname 
        FROM history_records
        WHERE user_id=? 
        AND nickname IS NOT NULL 
        AND nickname NOT IN ('西', '1 0', '10', '1', '0')
        ORDER BY created_at DESC 
        LIMIT 1
    ''', (uid,))
    
    result = cursor.fetchone()
    correct_nickname = result[0] if result else "无正确昵称"
    
    # 统计该用户有多少条"1 0"记录
    cursor.execute('''
        SELECT COUNT(*) 
        FROM history_records
        WHERE user_id=? AND nickname='1 0'
    ''', (uid,))
    
    bad_count = cursor.fetchone()[0]
    
    print(f"用户ID: {uid}")
    print(f"  正确昵称: {correct_nickname}")
    print(f"  错误记录数: {bad_count}")
    print()

conn.close()
