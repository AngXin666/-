#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试打包后的加密功能 - 详细诊断
"""

import sys
import os
from pathlib import Path

print("=" * 60)
print("打包后加密功能详细诊断")
print("=" * 60)

# 1. 测试导入
print("\n[1] 测试导入模块...")
try:
    from src.crypto_utils import CryptoUtils
    print("  ✓ CryptoUtils 导入成功")
except Exception as e:
    print(f"  ✗ CryptoUtils 导入失败: {e}")
    sys.exit(1)

# 2. 测试获取机器ID
print("\n[2] 测试获取机器ID...")
try:
    machine_id = CryptoUtils.get_machine_id()
    print(f"  ✓ 机器ID: {machine_id[:16]}...{machine_id[-16:]}")
    print(f"  ✓ 机器ID长度: {len(machine_id)}")
except Exception as e:
    print(f"  ✗ 获取机器ID失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 3. 测试加密解密
print("\n[3] 测试加密解密...")
try:
    test_data = "测试数据123".encode('utf-8')
    print(f"  原始数据: {test_data}")
    
    # 加密
    encrypted = CryptoUtils.encrypt_with_machine_binding(test_data)
    print(f"  ✓ 加密成功，长度: {len(encrypted)}")
    
    # 解密
    decrypted = CryptoUtils.decrypt_with_machine_binding(encrypted)
    print(f"  ✓ 解密成功: {decrypted}")
    
    if decrypted == test_data:
        print("  ✓ 加密解密测试通过")
    else:
        print("  ✗ 加密解密结果不匹配")
except Exception as e:
    print(f"  ✗ 加密解密测试失败: {e}")
    import traceback
    traceback.print_exc()

# 4. 测试读取账号文件
print("\n[4] 测试读取账号文件...")
accounts_file = "data/accounts.txt"
encrypted_file = accounts_file + ".enc"

print(f"  账号文件: {accounts_file}")
print(f"  加密文件: {encrypted_file}")

if os.path.exists(encrypted_file):
    print(f"  ✓ 加密文件存在，大小: {os.path.getsize(encrypted_file)} 字节")
    
    try:
        from src.encrypted_accounts_file import EncryptedAccountsFile
        enc_file = EncryptedAccountsFile(accounts_file)
        accounts = enc_file.read_accounts()
        print(f"  ✓ 读取成功，账号数量: {len(accounts)}")
        if accounts:
            print(f"  ✓ 第一个账号: {accounts[0][0]}")
    except Exception as e:
        print(f"  ✗ 读取账号文件失败: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"  ✗ 加密文件不存在")

# 5. 测试读取账号缓存
print("\n[5] 测试读取账号缓存...")
cache_file = ".account_cache.json.enc"

if os.path.exists(cache_file):
    print(f"  ✓ 缓存文件存在，大小: {os.path.getsize(cache_file)} 字节")
    
    try:
        from src.account_cache import AccountCache
        cache = AccountCache()
        stats = cache.get_stats()
        print(f"  ✓ 读取成功")
        print(f"    - 总缓存数: {stats['total']}")
        print(f"    - 完整信息: {stats['complete']}")
        print(f"    - 部分信息: {stats['partial']}")
    except Exception as e:
        print(f"  ✗ 读取缓存失败: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"  ✗ 缓存文件不存在")

print("\n" + "=" * 60)
print("诊断完成")
print("=" * 60)
