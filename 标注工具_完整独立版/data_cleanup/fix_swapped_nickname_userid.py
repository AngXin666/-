"""
修复昵称和用户ID对调的记录
- 识别昵称为'-'但用户ID看起来像昵称的记录
- 从该账号的其他正常记录中获取正确的昵称和用户ID
- 交换这些记录的昵称和用户ID字段
"""

import sqlite3
from pathlib import Path
import re

def is_likely_nickname(text):
    """判断文本是否更像昵称而不是用户ID"""
    if not text or text == '-':
        return False
    
    # 用户ID通常是纯数字
    if text.isdigit():
        return False
    
    # 包含中文字符的更可能是昵称
    if re.search(r'[\u4e00-\u9fff]', text):
        return True
    
    # 长度较短且包含字母的可能是昵称
    if len(text) <= 10 and re.search(r'[a-zA-Z]', text):
        return True
    
    return False

def fix_swapped_records():
    """修复昵称和用户ID对调的记录"""
    db_path = Path("runtime_data") / "license.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    print("=" * 80)
    print("修复昵称和用户ID对调的记录")
    print("=" * 80)
    
    # 查找可能对调的记录：昵称为'-'且用户ID看起来像昵称
    cursor.execute("""
        SELECT id, phone, nickname, user_id, run_date, created_at
        FROM history_records
        WHERE nickname = '-'
        ORDER BY phone, created_at
    """)
    
    records = cursor.fetchall()
    print(f"\n找到 {len(records)} 条昵称为'-'的记录")
    
    fixed_count = 0
    skipped_count = 0
    
    for record in records:
        record_id, phone, nickname, user_id, run_date, created_at = record
        
        # 检查用户ID是否看起来像昵称
        if not is_likely_nickname(user_id):
            print(f"\n跳过记录 ID={record_id} (用户ID看起来正常): {phone}, user_id={user_id}")
            skipped_count += 1
            continue
        
        print(f"\n处理记录 ID={record_id}:")
        print(f"  手机号: {phone}")
        print(f"  当前昵称: '{nickname}'")
        print(f"  当前用户ID: '{user_id}' (看起来像昵称)")
        print(f"  日期: {run_date}")
        print(f"  创建时间: {created_at}")
        
        # 查找该账号的其他正常记录
        cursor.execute("""
            SELECT nickname, user_id
            FROM history_records
            WHERE phone = ? 
              AND id != ?
              AND nickname != '-'
              AND nickname IS NOT NULL
            ORDER BY created_at DESC
            LIMIT 1
        """, (phone, record_id))
        
        normal_record = cursor.fetchone()
        
        if normal_record:
            normal_nickname, normal_user_id = normal_record
            print(f"  找到正常记录:")
            print(f"    正常昵称: {normal_nickname}")
            print(f"    正常用户ID: {normal_user_id}")
            
            # 判断：如果当前的user_id和正常记录的nickname相似，说明确实对调了
            if user_id == normal_nickname or user_id in normal_nickname or normal_nickname in user_id:
                print(f"  ✓ 确认对调，进行修复:")
                print(f"    新昵称: {user_id}")
                print(f"    新用户ID: {normal_user_id}")
                
                # 交换昵称和用户ID
                cursor.execute("""
                    UPDATE history_records
                    SET nickname = ?, user_id = ?
                    WHERE id = ?
                """, (user_id, normal_user_id, record_id))
                
                fixed_count += 1
            else:
                print(f"  ⚠️ 无法确认是否对调，跳过")
                skipped_count += 1
        else:
            print(f"  ⚠️ 未找到该账号的其他正常记录，跳过")
            skipped_count += 1
    
    conn.commit()
    conn.close()
    
    print("\n" + "=" * 80)
    print("修复完成")
    print("=" * 80)
    print(f"成功修复: {fixed_count} 条")
    print(f"跳过: {skipped_count} 条")


if __name__ == '__main__':
    fix_swapped_records()
