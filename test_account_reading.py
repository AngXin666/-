#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试账号文件读取
"""

import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_account_reading():
    """测试账号文件读取"""
    print("=" * 60)
    print("测试账号文件读取")
    print("=" * 60)
    
    # 1. 测试机器ID获取
    print("\n[1] 测试机器ID获取...")
    try:
        from src.crypto_utils import CryptoUtils
        machine_id = CryptoUtils.get_machine_id()
        print(f"  ✓ 机器ID: {machine_id[:16]}...")
    except Exception as e:
        print(f"  ✗ 获取机器ID失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 2. 测试账号文件路径
    print("\n[2] 检查账号文件...")
    accounts_file = "data/accounts.txt"
    encrypted_file = accounts_file + ".enc"
    
    print(f"  明文文件: {accounts_file}")
    print(f"    存在: {os.path.exists(accounts_file)}")
    
    print(f"  加密文件: {encrypted_file}")
    print(f"    存在: {os.path.exists(encrypted_file)}")
    
    if os.path.exists(encrypted_file):
        file_size = os.path.getsize(encrypted_file)
        print(f"    大小: {file_size} 字节")
    
    # 3. 测试读取账号
    print("\n[3] 测试读取账号...")
    try:
        from src.encrypted_accounts_file import EncryptedAccountsFile
        
        encrypted_file_obj = EncryptedAccountsFile(accounts_file)
        accounts = encrypted_file_obj.read_accounts()
        
        print(f"  ✓ 成功读取 {len(accounts)} 个账号")
        
        # 显示前3个账号（隐藏密码）
        for i, (phone, password) in enumerate(accounts[:3]):
            masked_password = password[:2] + "*" * (len(password) - 4) + password[-2:] if len(password) > 4 else "****"
            print(f"    {i+1}. {phone} / {masked_password}")
        
        if len(accounts) > 3:
            print(f"    ... 还有 {len(accounts) - 3} 个账号")
        
    except ValueError as e:
        print(f"  ✗ 解密失败: {e}")
        print(f"\n  可能原因:")
        print(f"    1. 账号文件是在其他机器上创建的")
        print(f"    2. 机器硬件信息发生了变化")
        print(f"    3. 账号文件已损坏")
        print(f"\n  解决方案:")
        print(f"    1. 在打包目录中重新导入账号")
        print(f"    2. 或者将明文账号文件复制到 data/ 目录")
    except Exception as e:
        print(f"  ✗ 读取失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)

if __name__ == '__main__':
    test_account_reading()
