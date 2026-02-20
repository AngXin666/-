#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复特定的缩进问题
"""

file_path = 'src/user_management_gui.py'

# 读取文件
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 修复模式1: 删除登录缓存部分的缩进问题
# 查找并替换错误的缩进模式
old_pattern1 = """            # 删除登录缓存
                from .adb_bridge import ADBBridge
                adb = ADBBridge()
            from .login_cache_manager import LoginCacheManager"""

new_pattern1 = """            # 删除登录缓存
            from .adb_bridge import ADBBridge
            adb = ADBBridge()
            from .login_cache_manager import LoginCacheManager"""

content = content.replace(old_pattern1, new_pattern1)

# 写回文件
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"✓ 文件已保存: {file_path}")

# 验证语法
import py_compile
try:
    py_compile.compile(file_path, doraise=True)
    print("✓ 语法检查通过")
except SyntaxError as e:
    print(f"❌ 语法错误 (第 {e.lineno} 行): {e.msg}")
    if e.text:
        print(f"   {e.text.strip()}")
