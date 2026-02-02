"""同步账号映射到数据库
将 UserManager 中的账号映射同步到数据库的 owner 字段
"""
from src.user_manager import UserManager
from src.local_db import LocalDatabase

def sync_owners_to_database():
    """将账号映射同步到数据库"""
    print("=" * 60)
    print("同步账号管理员到数据库")
    print("=" * 60)
    print()
    
    # 加载 UserManager
    um = UserManager()
    db = LocalDatabase()
    
    # 获取所有账号映射
    print(f"账号映射中有 {len(um.account_mapping)} 个账号")
    print()
    
    if not um.account_mapping:
        print("没有账号映射，无需同步")
        return
    
    # 按用户分组
    user_accounts = {}
    for phone, user_id in um.account_mapping.items():
        if user_id not in user_accounts:
            user_accounts[user_id] = []
        user_accounts[user_id].append(phone)
    
    # 显示分组情况
    print("账号分组:")
    for user_id, phones in user_accounts.items():
        user = um.get_user(user_id)
        user_name = user.user_name if user else "未知用户"
        print(f"  {user_name} ({user_id}): {len(phones)} 个账号")
    print()
    
    # 确认是否继续
    response = input("是否将这些映射同步到数据库？(y/n): ")
    if response.lower() != 'y':
        print("已取消")
        return
    
    print()
    print("开始同步...")
    print()
    
    # 同步到数据库
    total_updated = 0
    for user_id, phones in user_accounts.items():
        user = um.get_user(user_id)
        if not user:
            print(f"⚠️ 用户 {user_id} 不存在，跳过")
            continue
        
        user_name = user.user_name
        count = db.batch_update_account_owner(phones, user_name)
        total_updated += count
        print(f"✓ 已更新 {count} 个账号的管理员为: {user_name}")
    
    print()
    print("=" * 60)
    print(f"✅ 同步完成！共更新 {total_updated} 个账号")
    print("=" * 60)

if __name__ == '__main__':
    sync_owners_to_database()
