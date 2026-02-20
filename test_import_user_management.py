#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试导入 user_management_gui 模块
"""

import sys
import os

print("="*60)
print("测试导入 src.user_management_gui 模块")
print("="*60)
print()

# 测试1: 检查 src 是否在 sys.path 中
print("1. 检查 sys.path:")
src_in_path = any('src' in p for p in sys.path)
print(f"   src 在 sys.path 中: {src_in_path}")
print()

# 测试2: 尝试导入 src 包
print("2. 尝试导入 src 包:")
try:
    import src
    print(f"   ✓ 成功导入 src")
    print(f"   src.__file__ = {src.__file__}")
    print(f"   src.__path__ = {src.__path__}")
except Exception as e:
    print(f"   ❌ 导入失败: {e}")
print()

# 测试3: 尝试导入 src.user_management_gui
print("3. 尝试导入 src.user_management_gui:")
try:
    import src.user_management_gui
    print(f"   ✓ 成功导入 src.user_management_gui")
    print(f"   模块路径: {src.user_management_gui.__file__}")
    
    # 检查类是否存在
    if hasattr(src.user_management_gui, 'UserManagementDialog'):
        print(f"   ✓ UserManagementDialog 类存在")
    else:
        print(f"   ❌ UserManagementDialog 类不存在")
except Exception as e:
    print(f"   ❌ 导入失败: {e}")
    import traceback
    traceback.print_exc()
print()

# 测试4: 尝试导入其他依赖模块
print("4. 测试导入依赖模块:")
dependencies = [
    'src.user_manager',
    'src.local_db',
    'src.transfer_config',
    'src.config',
    'src.encrypted_accounts_file',
    'src.login_cache_manager',
    'src.account_cache',
]

for dep in dependencies:
    try:
        __import__(dep)
        print(f"   ✓ {dep}")
    except Exception as e:
        print(f"   ❌ {dep}: {e}")

print()
print("="*60)
print("测试完成")
print("="*60)
