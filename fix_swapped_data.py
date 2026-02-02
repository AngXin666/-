"""
修复数据库中昵称和用户ID互换的问题
"""

import sqlite3
from pathlib import Path

def fix_swapped_data():
    """修复昵称和用户ID互换的数据"""
    db_path = Path("runtime_data") / "license.db"
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # 查找昵称为'-'且用户ID不是数字的记录（这些记录可能是互换的）
    cursor.execute("""
        SELECT id, phone, nickname, user_id
        FROM history_records
        WHERE nickname = '-' AND user_id NOT LIKE '%[0-9]%'
    """)
    
    swapped_records = cursor.fetchall()
    
    print(f"找到 {len(swapped_records)} 条可能互换的记录")
    
    if not swapped_records:
        print("没有需要修复的记录")
        conn.close()
        return
    
    # 显示前10条
    print("\n前10条记录:")
    for i, (record_id, phone, nickname, user_id) in enumerate(swapped_records[:10], 1):
        print(f"  {i}. ID={record_id}, 手机号={phone}, 昵称='{nickname}', 用户ID='{user_id}'")
    
    # 询问是否修复
    print(f"\n准备修复这 {len(swapped_records)} 条记录...")
    
    # 自动修复数据：对于每条记录，查找该手机号的前一条记录
    fixed_count = 0
    for record_id, phone, nickname, user_id in swapped_records:
        # 查找该手机号的前一条正常记录
        cursor.execute("""
            SELECT nickname, user_id
            FROM history_records
            WHERE phone = ? AND id < ? AND nickname != '-'
            ORDER BY id DESC
            LIMIT 1
        """, (phone, record_id))
        
        prev_record = cursor.fetchone()
        
        if prev_record:
            prev_nickname, prev_user_id = prev_record
            # 使用前一条记录的数据来修复
            cursor.execute("""
                UPDATE history_records
                SET nickname = ?, user_id = ?
                WHERE id = ?
            """, (prev_nickname, prev_user_id, record_id))
            fixed_count += 1
            print(f"  修复记录 ID={record_id}: 昵称='{prev_nickname}', 用户ID='{prev_user_id}'")
    
    conn.commit()
    conn.close()
    
    print(f"\n✅ 成功修复 {fixed_count} 条记录")


if __name__ == '__main__':
    fix_swapped_data()
