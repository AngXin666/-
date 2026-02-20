#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试打包后的程序是否能正确读取账号文件
"""

import sys
import os

# 设置工作目录（模拟打包后的行为）
if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
else:
    application_path = os.path.dirname(os.path.abspath(__file__))

os.chdir(application_path)
sys.path.insert(0, application_path)

print(f"当前工作目录: {os.getcwd()}")
print(f"Python路径: {sys.path[:3]}")

# 测试读取配置文件
try:
    from src.config import ConfigLoader
    config = ConfigLoader().load()
    print(f"\n配置文件加载成功:")
    print(f"  accounts_file: {config.accounts_file}")
    
    # 检查文件是否存在
    from pathlib import Path
    accounts_file = Path(config.accounts_file)
    encrypted_file = Path(str(config.accounts_file) + '.enc')
    
    print(f"\n文件检查:")
    print(f"  {accounts_file} 存在: {accounts_file.exists()}")
    print(f"  {encrypted_file} 存在: {encrypted_file.exists()}")
    
    if encrypted_file.exists():
        print(f"  {encrypted_file} 大小: {encrypted_file.stat().st_size} 字节")
    
except Exception as e:
    print(f"\n配置文件加载失败: {e}")
    import traceback
    traceback.print_exc()

# 测试读取账号
try:
    from src.encrypted_accounts_file import EncryptedAccountsFile
    
    encrypted_file = EncryptedAccountsFile(config.accounts_file)
    accounts = encrypted_file.read_accounts()
    
    print(f"\n账号读取成功:")
    print(f"  账号数量: {len(accounts)}")
    if accounts:
        print(f"  第一个账号: {accounts[0][0]}****")
    
except Exception as e:
    print(f"\n账号读取失败: {e}")
    import traceback
    traceback.print_exc()

# 测试读取账号缓存
try:
    from src.account_cache import AccountCache
    
    cache = AccountCache()
    cached_accounts = cache.load_cache()
    
    print(f"\n账号缓存读取成功:")
    print(f"  缓存账号数量: {len(cached_accounts)}")
    if cached_accounts:
        first_phone = list(cached_accounts.keys())[0]
        print(f"  第一个缓存账号: {first_phone}****")
    
except Exception as e:
    print(f"\n账号缓存读取失败: {e}")
    import traceback
    traceback.print_exc()

print("\n测试完成!")
input("按回车键退出...")
