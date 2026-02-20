#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重新加密账号文件工具
用于在新机器上重新加密账号文件
"""

import sys
import os
from pathlib import Path

print("=" * 60)
print("重新加密账号文件工具")
print("=" * 60)

# 1. 检查是否有明文账号文件
print("\n[1] 检查账号文件...")
accounts_file = "data/账号详情.xlsx"
accounts_txt = "data/accounts.txt"

if not os.path.exists(accounts_file) and not os.path.exists(accounts_txt):
    print(f"  ✗ 找不到账号文件")
    print(f"  请确保以下文件之一存在：")
    print(f"    - {accounts_file}")
    print(f"    - {accounts_txt}")
    sys.exit(1)

# 2. 读取账号
print("\n[2] 读取账号...")
accounts = []

# 优先读取 Excel 文件
if os.path.exists(accounts_file):
    print(f"  从 Excel 文件读取: {accounts_file}")
    try:
        import openpyxl
        wb = openpyxl.load_workbook(accounts_file)
        ws = wb.active
        
        for row in ws.iter_rows(min_row=2, values_only=True):
            if row[0] and row[1]:  # 假设第一列是手机号，第二列是密码
                phone = str(row[0]).strip()
                password = str(row[1]).strip()
                if phone and password:
                    accounts.append((phone, password))
        
        print(f"  ✓ 读取到 {len(accounts)} 个账号")
    except Exception as e:
        print(f"  ✗ 读取 Excel 失败: {e}")
        print(f"  提示：请安装 openpyxl: pip install openpyxl")
        sys.exit(1)

# 如果没有 Excel，读取文本文件
elif os.path.exists(accounts_txt):
    print(f"  从文本文件读取: {accounts_txt}")
    try:
        with open(accounts_txt, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                if '----' in line:
                    parts = line.split('----', 1)
                    phone = parts[0].strip()
                    password = parts[1].strip() if len(parts) > 1 else ""
                    
                    if phone and password:
                        accounts.append((phone, password))
        
        print(f"  ✓ 读取到 {len(accounts)} 个账号")
    except Exception as e:
        print(f"  ✗ 读取文本文件失败: {e}")
        sys.exit(1)

if not accounts:
    print(f"  ✗ 没有读取到任何账号")
    sys.exit(1)

# 3. 重新加密
print("\n[3] 重新加密账号文件...")
try:
    from src.encrypted_accounts_file import EncryptedAccountsFile
    
    enc_file = EncryptedAccountsFile("data/accounts.txt")
    
    # 删除旧的加密文件
    if os.path.exists("data/accounts.txt.enc"):
        os.remove("data/accounts.txt.enc")
        print(f"  删除旧的加密文件")
    
    # 写入新的加密文件
    if enc_file.write_accounts(accounts):
        print(f"  ✓ 重新加密成功")
        print(f"  ✓ 已保存 {len(accounts)} 个账号到 data/accounts.txt.enc")
    else:
        print(f"  ✗ 重新加密失败")
        sys.exit(1)
        
except Exception as e:
    print(f"  ✗ 重新加密失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. 验证
print("\n[4] 验证加密文件...")
try:
    enc_file = EncryptedAccountsFile("data/accounts.txt")
    test_accounts = enc_file.read_accounts()
    
    if len(test_accounts) == len(accounts):
        print(f"  ✓ 验证成功，可以正确读取 {len(test_accounts)} 个账号")
    else:
        print(f"  ✗ 验证失败，账号数量不匹配")
        print(f"    原始: {len(accounts)}, 读取: {len(test_accounts)}")
        
except Exception as e:
    print(f"  ✗ 验证失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("完成")
print("=" * 60)
print("\n提示：")
print("  1. 账号文件已重新加密，使用当前机器的机器ID")
print("  2. 现在可以正常使用程序了")
print("  3. 如果需要在其他机器使用，需要在那台机器上重新运行此工具")
