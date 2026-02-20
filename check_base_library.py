#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查 base_library.zip 中的模块
"""

import zipfile
import os

base_library_path = r"D:\溪盟商城自动化助手_打包\溪盟商城自动化助手\_internal\base_library.zip"

if not os.path.exists(base_library_path):
    print(f"❌ 找不到 base_library.zip: {base_library_path}")
    exit(1)

print(f"✓ 找到 base_library.zip")
print()

try:
    with zipfile.ZipFile(base_library_path, 'r') as zf:
        all_files = zf.namelist()
        
        # 查找 src 相关的文件
        src_files = [f for f in all_files if 'src' in f.lower()]
        user_files = [f for f in all_files if 'user' in f.lower()]
        
        print(f"总共 {len(all_files)} 个文件")
        print()
        
        print(f"找到 {len(src_files)} 个 src 相关文件:")
        for f in sorted(src_files)[:30]:  # 只显示前30个
            print(f"  ✓ {f}")
        if len(src_files) > 30:
            print(f"  ... 还有 {len(src_files) - 30} 个文件")
        print()
        
        print(f"找到 {len(user_files)} 个 user 相关文件:")
        for f in sorted(user_files):
            print(f"  ✓ {f}")
        print()
        
        # 检查关键模块
        key_modules = [
            'src/user_management_gui.pyc',
            'src/user_manager.pyc',
            'src/gui.pyc',
            'src/local_db.pyc',
            'src/transfer_config.pyc',
        ]
        
        print("检查关键模块:")
        for mod in key_modules:
            if mod in all_files:
                print(f"  ✓ {mod} - 已包含")
            else:
                print(f"  ❌ {mod} - 未找到")
        
except Exception as e:
    print(f"❌ 检查 base_library.zip 时出错: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("检查完成")
print("="*60)
