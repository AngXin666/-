#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试账号文件读取（打包环境）
"""

import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

def test_account_reading():
    """测试账号文件读取"""
    print("=" * 60)
    print("测试账号文件读取（打包环境）")
    print("=" * 60)
    
    # 1. 测试机器ID获取
    print("\n[1] 测试机器ID获取...")
    try:
        # 尝试多种导入方式（适配打包环境）
        try:
            from src.crypto_utils import CryptoUtils
        except ImportError:
            try:
                from crypto_utils import CryptoUtils
            except ImportError:
                import crypto_utils
                CryptoUtils = crypto_utils.CryptoUtils
        
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
        # 尝试多种导入方式（适配打包环境）
        try:
            from src.encrypted_accounts_file import EncryptedAccountsFile
        except ImportError:
            try:
                from encrypted_accounts_file import EncryptedAccountsFile
            except ImportError:
                import encrypted_accounts_file
                EncryptedAccountsFile = encrypted_accounts_file.EncryptedAccountsFile
        
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
    except Exception as e:
        print(f"  ✗ 读取失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. 检查登录缓存
    print("\n[4] 检查登录缓存...")
    login_cache_dir = "login_cache"
    if os.path.exists(login_cache_dir):
        cache_count = len([d for d in os.listdir(login_cache_dir) if os.path.isdir(os.path.join(login_cache_dir, d))])
        print(f"  ✓ 登录缓存目录存在")
        print(f"    缓存账号数: {cache_count}")
    else:
        print(f"  ✗ 登录缓存目录不存在")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

if __name__ == '__main__':
    test_account_reading()
    input("\n按回车键退出...")
