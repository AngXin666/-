#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复 user_management_gui.py 中的缩进问题
"""

import re

file_path = 'src/user_management_gui.py'

# 读取文件
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 修复模式：将过度缩进的 import 语句修复为正确的缩进
# 查找：20个或更多空格 + from .xxx import
# 替换为：16个空格 + from .xxx import
pattern = r'^(\s{20,})(from \.[a-zA-Z_]+ import)'
replacement = r'                \2'  # 16个空格

fixed_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

# 统计修复的行数
original_lines = content.split('\n')
fixed_lines = fixed_content.split('\n')
fixed_count = sum(1 for o, f in zip(original_lines, fixed_lines) if o != f)

print(f"修复了 {fixed_count} 行缩进问题")

# 写回文件
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(fixed_content)

print(f"✓ 文件已保存: {file_path}")

# 验证语法
import py_compile
try:
    py_compile.compile(file_path, doraise=True)
    print("✓ 语法检查通过")
except SyntaxError as e:
    print(f"❌ 语法错误: {e}")
