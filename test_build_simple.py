#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import shutil

# 设置控制台UTF-8编码
if sys.platform == 'win32':
    import io
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except:
        pass

print("="*60)
print("测试打包脚本")
print("="*60)

# 测试清理目录
print("\n[1/1] 测试清理...")
dirs_to_clean = ['build', 'dist']
for dir_name in dirs_to_clean:
    if os.path.exists(dir_name):
        print(f"  删除: {dir_name}")
        shutil.rmtree(dir_name, ignore_errors=True)

print("  清理完成")
print("\n测试成功！")
