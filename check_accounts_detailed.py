"""详细检查账号文件和数据库的差异"""
from src.account_manager import AccountManager
from src.local_db import LocalDatabase
from datetime import datetime
from pathlib import Path

# 检查账号文件
accounts_file = "data/accounts.txt"
print(f"检查账号文件: {accounts_file}")

# 检查文件是否存在
plain_exists = Path(accounts_file).exists()
enc_exists = Path(f"{accounts_file}.enc").exists()

print(f"明文文件存在: {plain_exists}")
print(f"加密文件存在: {enc_exists}")

if enc_exists:
    enc_size = Path(f"{accounts_file}.enc").stat().st_size
    print(f"加密文件大小: {enc_size} 字节")

# 尝试加载账号文件
try:
    account_manager = AccountManager(accounts_file)
    accounts = account_manager.load_accounts()
    print(f"\n从账号文件加载了 {len(accounts)} 个账号")
    
    if accounts:
        print("\n账号文件中的所有账号:")
        for i, acc in enumerate(accounts, 1):
            print(f"{i:3d}. {acc.phone}")
        
        # 获取账号文件中的手机号集合
        file_phones = {acc.phone for acc in accounts}
        
        # 从数据库获取今天新增的账号
        db = LocalDatabase()
        conn = db._get_connection()
        cursor = conn.cursor()
        
        today = datetime.now().strftime('%Y-%m-%d')
        cursor.execute('''
            SELECT phone
            FROM history_records 
            WHERE DATE(created_at) = ?
            ORDER BY created_at ASC
        ''', (today,))
        today_phones = [row[0] for row in cursor.fetchall()]
        today_phones_set = set(today_phones)
        
        print(f"\n数据库中今天新增的账号数: {len(today_phones)}")
        
        # 找出在账号文件中但不在数据库今天记录中的账号
        in_file_not_in_db = file_phones - today_phones_set
        
        # 找出在数据库今天记录中但不在账号文件中的账号
        in_db_not_in_file = today_phones_set - file_phones
        
        if in_file_not_in_db:
            print(f"\n在账号文件中但今天没有导入数据库的账号 ({len(in_file_not_in_db)} 个):")
            for phone in sorted(in_file_not_in_db):
                print(f"  {phone}")
        
        if in_db_not_in_file:
            print(f"\n在数据库今天记录中但不在账号文件中的账号 ({len(in_db_not_in_file)} 个):")
            for phone in sorted(in_db_not_in_file):
                print(f"  {phone}")
        
        # 检查账号文件中最近添加的账号（假设是最后38个）
        if len(accounts) >= 38:
            print(f"\n账号文件中最后38个账号（应该是今天添加的）:")
            recent_accounts = accounts[-38:]
            recent_phones = {acc.phone for acc in recent_accounts}
            
            for i, acc in enumerate(recent_accounts, 1):
                in_db = "✓" if acc.phone in today_phones_set else "✗"
                print(f"{i:2d}. {acc.phone} {in_db}")
            
            # 找出这38个账号中哪些没有导入到数据库
            not_imported = recent_phones - today_phones_set
            if not_imported:
                print(f"\n这38个账号中没有导入到数据库的 ({len(not_imported)} 个):")
                for phone in sorted(not_imported):
                    print(f"  ❌ {phone}")
        
        conn.close()
    else:
        print("账号文件为空")
        
except Exception as e:
    print(f"\n加载账号文件失败: {e}")
    import traceback
    traceback.print_exc()
