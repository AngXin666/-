#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单测试账号加载（用于打包环境诊断）
"""

import sys
import os

print("=" * 60)
print("测试账号加载")
print("=" * 60)

# 1. 测试文件是否存在
print("\n[1] 检查文件...")
accounts_file = "data/accounts.txt"
encrypted_file = accounts_file + ".enc"

print(f"  加密文件: {encrypted_file}")
if os.path.exists(encrypted_file):
    file_size = os.path.getsize(encrypted_file)
    print(f"    ✓ 存在 ({file_size} 字节)")
else:
    print(f"    ✗ 不存在")
    sys.exit(1)

# 2. 测试AccountManager导入
print("\n[2] 测试AccountManager导入...")
try:
    # 尝试多种导入方式
    try:
        from src.account_manager import AccountManager
        print("  ✓ 导入成功 (src.account_manager)")
    except ImportError:
        try:
            from account_manager import AccountManager
            print("  ✓ 导入成功 (account_manager)")
        except ImportError:
            import account_manager
            AccountManager = account_manager.AccountManager
            print("  ✓ 导入成功 (import account_manager)")
except Exception as e:
    print(f"  ✗ 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 3. 测试加载账号
print("\n[3] 测试加载账号...")
try:
    manager = AccountManager(accounts_file)
    accounts = manager.load_accounts()
    
    print(f"  ✓ 成功加载 {len(accounts)} 个账号")
    
    if len(accounts) > 0:
        print(f"\n  前3个账号:")
        for i, acc in enumerate(accounts[:3]):
            print(f"    {i+1}. {acc.phone}")
    
except Exception as e:
    print(f"  ✗ 加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ 测试完成！账号加载正常")
print("=" * 60)
