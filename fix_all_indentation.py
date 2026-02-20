#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复 user_management_gui.py 中的所有缩进问题
"""

file_path = 'src/user_management_gui.py'

# 读取文件
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 修复规则
fixed_lines = []
for i, line in enumerate(lines):
    # 规则1: 修复过度缩进的 from .xxx import 语句
    # 如果一行有20个或更多空格开头，且包含 "from ." 或 "db = "
    if line.startswith(' ' * 20):
        # 计算当前缩进
        stripped = line.lstrip()
        if stripped.startswith('from .') or stripped.startswith('db = ') or stripped.startswith('adb = '):
            # 替换为16个空格（正确的缩进）
            fixed_line = ' ' * 16 + stripped
            fixed_lines.append(fixed_line)
            print(f"修复第 {i+1} 行: {line.strip()[:50]}...")
            continue
    
    # 保持原样
    fixed_lines.append(line)

# 写回文件
with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(fixed_lines)

print(f"\n✓ 文件已保存: {file_path}")

# 验证语法
import py_compile
try:
    py_compile.compile(file_path, doraise=True)
    print("✓ 语法检查通过")
except SyntaxError as e:
    print(f"❌ 语法错误 (第 {e.lineno} 行): {e.msg}")
    print(f"   {e.text}")
