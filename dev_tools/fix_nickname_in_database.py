"""
修复数据库中昵称为"西"的记录

问题：之前的OCR识别错误，导致很多昵称被识别为"西"
解决方案：从同一手机号的其他历史记录中获取正确的昵称
"""

import sys
import sqlite3
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def fix_nickname_in_database():
    """修复数据库中昵称为'西'的记录"""
    
    db_path = project_root / "runtime_data" / "license.db"
    
    if not db_path.exists():
        print(f"❌ 数据库文件不存在: {db_path}")
        return
    
    print(f"数据库路径: {db_path}")
    print("="*60)
    
    try:
        # 连接数据库
        conn = sqlite3.connect(str(db_path))
        conn.text_factory = str  # 确保UTF-8编码
        cursor = conn.cursor()
        
        # 1. 统计昵称为"西"的记录数量
        cursor.execute("""
            SELECT COUNT(*) FROM history_records 
            WHERE nickname = '西'
        """)
        count_before = cursor.fetchone()[0]
        print(f"发现 {count_before} 条昵称为'西'的记录")
        
        if count_before == 0:
            print("✓ 没有需要修复的记录")
            conn.close()
            return
        
        # 2. 获取所有昵称为"西"的记录（按手机号分组）
        cursor.execute("""
            SELECT DISTINCT phone FROM history_records 
            WHERE nickname = '西'
        """)
        phones_to_fix = [row[0] for row in cursor.fetchall()]
        print(f"涉及 {len(phones_to_fix)} 个手机号")
        
        # 3. 显示一些示例记录
        cursor.execute("""
            SELECT phone, nickname, user_id, run_date 
            FROM history_records 
            WHERE nickname = '西'
            LIMIT 10
        """)
        print("\n示例记录（前10条）：")
        print("-"*60)
        for row in cursor.fetchall():
            print(f"  手机号: {row[0]}, 昵称: {row[1]}, 用户ID: {row[2]}, 日期: {row[3]}")
        
        # 4. 询问用户确认
        print("\n" + "="*60)
        confirm = input(f"确认要修复这 {count_before} 条记录吗？(y/n): ")
        
        if confirm.lower() != 'y':
            print("❌ 操作已取消")
            conn.close()
            return
        
        # 5. 执行修复
        print("\n开始修复...")
        fixed_count = 0
        not_found_count = 0
        
        for phone in phones_to_fix:
            # 查找该手机号的正确昵称（从非"西"的记录中获取最新的）
            cursor.execute("""
                SELECT nickname FROM history_records 
                WHERE phone = ? 
                  AND nickname IS NOT NULL 
                  AND nickname != '' 
                  AND nickname != '西'
                ORDER BY created_at DESC
                LIMIT 1
            """, (phone,))
            
            result = cursor.fetchone()
            
            if result:
                correct_nickname = result[0]
                
                # 更新该手机号所有昵称为"西"的记录
                cursor.execute("""
                    UPDATE history_records 
                    SET nickname = ? 
                    WHERE phone = ? AND nickname = '西'
                """, (correct_nickname, phone))
                
                affected = cursor.rowcount
                fixed_count += affected
                print(f"  ✓ {phone}: 修复 {affected} 条记录 → {correct_nickname}")
            else:
                # 没有找到正确的昵称，设置为NULL
                cursor.execute("""
                    UPDATE history_records 
                    SET nickname = NULL 
                    WHERE phone = ? AND nickname = '西'
                """, (phone,))
                
                affected = cursor.rowcount
                not_found_count += affected
                print(f"  ⚠️ {phone}: 未找到正确昵称，设置为NULL ({affected} 条)")
        
        conn.commit()
        
        # 6. 验证修复结果
        cursor.execute("""
            SELECT COUNT(*) FROM history_records 
            WHERE nickname = '西'
        """)
        count_after = cursor.fetchone()[0]
        
        print("\n" + "="*60)
        print(f"✓ 修复完成")
        print(f"  - 成功修复: {fixed_count} 条")
        print(f"  - 未找到昵称: {not_found_count} 条")
        print(f"  - 修复前: {count_before} 条")
        print(f"  - 修复后: {count_after} 条")
        
        if count_after == 0:
            print("\n✓ 所有昵称为'西'的记录已修复")
        else:
            print(f"\n⚠️ 仍有 {count_after} 条记录未修复")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ 修复失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("="*60)
    print("修复数据库中昵称为'西'的记录")
    print("="*60)
    print()
    
    fix_nickname_in_database()
    
    print("\n" + "="*60)
    print("修复完成")
    print("="*60)
