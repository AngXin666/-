"""
修复剩余的异常数据
- 昵称为'-'的记录
- 昵称和用户ID为'N/A'或空字符串的记录
- 自动从最近正常的数据中获取正确值写入
- 从 phone_userid_mapping.txt 中获取用户ID
"""

import sqlite3
from pathlib import Path
from datetime import datetime

def load_phone_userid_mapping():
    """加载手机号-用户ID映射"""
    mapping_file = Path("login_cache") / "phone_userid_mapping.txt"
    
    if not mapping_file.exists():
        print("[警告] 未找到 phone_userid_mapping.txt 文件")
        return {}
    
    mapping = {}
    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '=' in line:
                    phone, user_id = line.split('=', 1)
                    mapping[phone] = user_id
        print(f"[信息] 加载了 {len(mapping)} 个手机号-用户ID映射")
        return mapping
    except Exception as e:
        print(f"[错误] 加载映射文件失败: {e}")
        return {}

def fix_remaining_bad_data():
    """修复剩余的异常数据"""
    db_path = Path("runtime_data") / "license.db"
    
    # 加载手机号-用户ID映射
    phone_userid_map = load_phone_userid_mapping()
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    print("=" * 80)
    print("修复 OCR 识别失败的异常数据")
    print("=" * 80)
    
    # 查找所有异常记录（包括'待处理'状态）
    cursor.execute("""
        SELECT id, phone, nickname, user_id, run_date, created_at
        FROM history_records
        WHERE nickname IN ('-', 'N/A', '', '待处理') 
           OR user_id IN ('-', 'N/A', '', '待处理')
           OR nickname IS NULL
           OR user_id IS NULL
        ORDER BY phone, created_at
    """)
    
    bad_records = cursor.fetchall()
    print(f"\n找到 {len(bad_records)} 条异常记录")
    
    fixed_count = 0
    cannot_fix_count = 0
    
    for record in bad_records:
        record_id, phone, nickname, user_id, run_date, created_at = record
        print(f"\n处理记录 ID={record_id}:")
        print(f"  手机号: {phone}")
        print(f"  昵称: '{nickname}'")
        print(f"  用户ID: '{user_id}'")
        print(f"  日期: {run_date}")
        print(f"  创建时间: {created_at}")
        
        correct_nickname = None
        correct_user_id = None
        
        # 策略1: 查找该账号的前一条正常记录
        cursor.execute("""
            SELECT nickname, user_id
            FROM history_records
            WHERE phone = ? 
              AND created_at < ?
              AND nickname NOT IN ('-', 'N/A', '', '待处理')
              AND user_id NOT IN ('-', 'N/A', '', '待处理')
              AND nickname IS NOT NULL
              AND user_id IS NOT NULL
            ORDER BY created_at DESC
            LIMIT 1
        """, (phone, created_at))
        
        prev_record = cursor.fetchone()
        
        if prev_record:
            correct_nickname, correct_user_id = prev_record
            print(f"  ✓ 找到前一条正常记录:")
            print(f"    正确昵称: {correct_nickname}")
            print(f"    正确用户ID: {correct_user_id}")
        else:
            # 策略2: 如果没有前一条记录，查找该账号的后一条正常记录
            cursor.execute("""
                SELECT nickname, user_id
                FROM history_records
                WHERE phone = ? 
                  AND created_at > ?
                  AND nickname NOT IN ('-', 'N/A', '', '待处理')
                  AND user_id NOT IN ('-', 'N/A', '', '待处理')
                  AND nickname IS NOT NULL
                  AND user_id IS NOT NULL
                ORDER BY created_at ASC
                LIMIT 1
            """, (phone, created_at))
            
            next_record = cursor.fetchone()
            
            if next_record:
                correct_nickname, correct_user_id = next_record
                print(f"  ✓ 找到后一条正常记录:")
                print(f"    正确昵称: {correct_nickname}")
                print(f"    正确用户ID: {correct_user_id}")
            else:
                # 策略3: 如果前后都没有，查找该账号的任意一条正常记录
                cursor.execute("""
                    SELECT nickname, user_id
                    FROM history_records
                    WHERE phone = ? 
                      AND nickname NOT IN ('-', 'N/A', '', '待处理')
                      AND user_id NOT IN ('-', 'N/A', '', '待处理')
                      AND nickname IS NOT NULL
                      AND user_id IS NOT NULL
                    ORDER BY created_at DESC
                    LIMIT 1
                """, (phone,))
                
                any_record = cursor.fetchone()
                
                if any_record:
                    correct_nickname, correct_user_id = any_record
                    print(f"  ✓ 找到任意一条正常记录:")
                    print(f"    正确昵称: {correct_nickname}")
                    print(f"    正确用户ID: {correct_user_id}")
                else:
                    # 策略4: 从 phone_userid_mapping.txt 中获取用户ID
                    if phone in phone_userid_map:
                        correct_user_id = phone_userid_map[phone]
                        print(f"  ✓ 从映射文件中找到用户ID: {correct_user_id}")
                        # 昵称暂时设置为 "待更新"
                        correct_nickname = "待更新"
                        print(f"    昵称设置为: {correct_nickname}")
                    else:
                        print(f"  ✗ 无法找到任何正常记录或映射")
                        print(f"    建议：等待该账号下次成功运行后，数据会自动更新")
                        cannot_fix_count += 1
                        continue
        
        # 更新记录
        if correct_nickname and correct_user_id:
            cursor.execute("""
                UPDATE history_records
                SET nickname = ?, user_id = ?
                WHERE id = ?
            """, (correct_nickname, correct_user_id, record_id))
            
            print(f"  ✓ 已修复")
            fixed_count += 1
    
    conn.commit()
    conn.close()
    
    print("\n" + "=" * 80)
    print("修复完成")
    print("=" * 80)
    print(f"成功修复: {fixed_count} 条")
    print(f"无法修复: {cannot_fix_count} 条")
    
    if cannot_fix_count > 0:
        print(f"\n无法修复的记录将在下次该账号成功运行时自动更新")
    
    # 验证修复结果
    print("\n" + "=" * 80)
    print("验证修复结果")
    print("=" * 80)
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT COUNT(*)
        FROM history_records
        WHERE nickname IN ('-', 'N/A', '', '待处理') 
           OR user_id IN ('-', 'N/A', '', '待处理')
           OR nickname IS NULL
           OR user_id IS NULL
    """)
    
    remaining_bad = cursor.fetchone()[0]
    conn.close()
    
    print(f"剩余异常记录: {remaining_bad} 条")
    
    if remaining_bad == 0:
        print("✅ 所有异常记录已修复！")
    else:
        print(f"⚠️ 还有 {remaining_bad} 条记录无法修复")


if __name__ == '__main__':
    fix_remaining_bad_data()
