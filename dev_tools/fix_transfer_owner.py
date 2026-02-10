"""修复转账记录中的管理员信息"""
import sqlite3
import json
from pathlib import Path

def fix_transfer_owner():
    """修复转账记录中的管理员信息"""
    
    # 1. 加载用户管理数据
    mapping_file = Path("runtime_data/account_user_mapping.json")
    users_file = Path("runtime_data/users.json")
    
    if not mapping_file.exists():
        print("❌ 账号映射文件不存在")
        return
    
    if not users_file.exists():
        print("❌ 用户文件不存在")
        return
    
    # 加载账号映射
    with open(mapping_file, 'r', encoding='utf-8') as f:
        account_mapping = json.load(f)
    
    # 加载用户信息
    with open(users_file, 'r', encoding='utf-8') as f:
        users = json.load(f)
    
    print("=" * 80)
    print("修复转账记录中的管理员信息")
    print("=" * 80)
    print(f"\n已加载 {len(account_mapping)} 个账号映射")
    print(f"已加载 {len(users)} 个用户")
    
    # 2. 连接数据库
    conn = sqlite3.connect('runtime_data/license.db')
    cursor = conn.cursor()
    
    # 3. 查询管理员为空的转账记录
    cursor.execute("""
        SELECT id, sender_phone 
        FROM transfer_history 
        WHERE owner IS NULL OR owner = '' OR owner = '-'
    """)
    
    empty_records = cursor.fetchall()
    print(f"\n找到 {len(empty_records)} 条管理员为空的记录")
    
    if not empty_records:
        print("✓ 所有记录都有管理员信息")
        conn.close()
        return
    
    # 4. 修复记录
    fixed_count = 0
    not_found_count = 0
    
    for record_id, phone in empty_records:
        # 查找账号对应的用户ID
        user_id = account_mapping.get(phone)
        
        if user_id:
            # 查找用户名
            user_info = users.get(user_id)
            if user_info:
                user_name = user_info.get('user_name', '未知')
                
                # 更新记录
                cursor.execute("""
                    UPDATE transfer_history 
                    SET owner = ? 
                    WHERE id = ?
                """, (user_name, record_id))
                
                fixed_count += 1
                print(f"  ✓ 修复记录 {record_id}: {phone} -> {user_name}")
            else:
                print(f"  ⚠️ 记录 {record_id}: {phone} 的用户ID {user_id} 不存在")
                not_found_count += 1
        else:
            # 账号未分配管理员，设置为"未分配"
            cursor.execute("""
                UPDATE transfer_history 
                SET owner = ? 
                WHERE id = ?
            """, ("未分配", record_id))
            
            fixed_count += 1
            print(f"  ✓ 修复记录 {record_id}: {phone} -> 未分配")
    
    # 5. 提交更改
    conn.commit()
    conn.close()
    
    print("\n" + "=" * 80)
    print(f"修复完成:")
    print(f"  成功修复: {fixed_count} 条")
    print(f"  未找到用户: {not_found_count} 条")
    print("=" * 80)


if __name__ == "__main__":
    fix_transfer_owner()
