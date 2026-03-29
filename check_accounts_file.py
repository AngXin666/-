"""检查账号文件"""
from src.encrypted_accounts_file import EncryptedAccountsFile
import traceback

try:
    f = EncryptedAccountsFile('data/accounts.txt.enc')
    accounts = f.read_accounts()
    print(f'成功读取 {len(accounts)} 个账号')
    
    if len(accounts) > 0:
        print(f'\n前5个账号:')
        for i, acc in enumerate(accounts[:5]):
            print(f'  {i+1}. {acc.phone} - {acc.password}')
    else:
        print('\n账号列表为空，检查文件内容...')
        import os
        size = os.path.getsize('data/accounts.txt.enc')
        print(f'文件大小: {size} 字节')
        
except Exception as e:
    print(f'读取失败: {e}')
    traceback.print_exc()
