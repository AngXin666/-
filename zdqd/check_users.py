"""检查当前用户数据"""
from src.user_manager import UserManager

um = UserManager()
users = um.get_all_users()

print(f'当前用户数: {len(users)}')
print()

if users:
    print('用户列表:')
    for user in users:
        account_count = len(um.get_user_accounts(user.user_id))
        recipients = ', '.join(user.transfer_recipients[:3])
        if len(user.transfer_recipients) > 3:
            recipients += f' 等{len(user.transfer_recipients)}个'
        
        print(f'  - {user.user_name} (ID: {user.user_id})')
        print(f'    收款人: {recipients}')
        print(f'    账号数: {account_count}')
        print(f'    状态: {"启用" if user.enabled else "禁用"}')
        print()
else:
    print('没有用户数据')
    print()
    print('提示：')
    print('1. 打开用户管理界面（主界面 → 用户管理按钮）')
    print('2. 点击"添加用户"创建新用户')
    print('3. 使用"批量添加账号"或"分配未分配账号"功能分配账号')
