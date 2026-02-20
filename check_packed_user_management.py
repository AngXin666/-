#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查打包后的程序是否包含 user_management_gui 模块
"""

import sys
import os

# 设置打包后程序的路径
packed_dir = r"D:\溪盟商城自动化助手_打包\溪盟商城自动化助手"
packed_exe = os.path.join(packed_dir, "溪盟商城自动化助手.exe")

if not os.path.exists(packed_exe):
    print(f"❌ 找不到打包后的程序: {packed_exe}")
    sys.exit(1)

print(f"✓ 找到打包后的程序: {packed_exe}")
print()

# 检查 _internal 目录
internal_dir = os.path.join(packed_dir, "_internal")
if not os.path.exists(internal_dir):
    print(f"❌ 找不到 _internal 目录: {internal_dir}")
    sys.exit(1)

print(f"✓ 找到 _internal 目录")
print()

# 检查 PYZ 文件（Python 模块归档）
pyz_files = [f for f in os.listdir(internal_dir) if f.endswith('.pyz')]
print(f"找到 {len(pyz_files)} 个 PYZ 文件:")
for pyz in pyz_files:
    print(f"  - {pyz}")
print()

# 尝试解压并检查 PYZ 文件内容
try:
    from PyInstaller.archive.readers import CArchiveReader, ZlibArchiveReader
    
    # 读取主 PYZ 文件
    main_pyz = os.path.join(internal_dir, "PYZ-00.pyz")
    if os.path.exists(main_pyz):
        print(f"检查 PYZ-00.pyz 中的模块...")
        
        with open(main_pyz, 'rb') as f:
            reader = ZlibArchiveReader(f)
            toc = reader.toc
            
            # 查找 src 相关的模块
            src_modules = [name for name in toc.keys() if 'src' in name.lower()]
            user_modules = [name for name in toc.keys() if 'user' in name.lower()]
            
            print(f"\n找到 {len(src_modules)} 个 src 相关模块:")
            for mod in sorted(src_modules)[:20]:  # 只显示前20个
                print(f"  ✓ {mod}")
            if len(src_modules) > 20:
                print(f"  ... 还有 {len(src_modules) - 20} 个模块")
            
            print(f"\n找到 {len(user_modules)} 个 user 相关模块:")
            for mod in sorted(user_modules):
                print(f"  ✓ {mod}")
            
            # 检查关键模块
            key_modules = [
                'src.user_management_gui',
                'src.user_manager',
                'src.gui',
                'src.local_db',
                'src.transfer_config',
            ]
            
            print(f"\n检查关键模块:")
            for mod in key_modules:
                if mod in toc:
                    print(f"  ✓ {mod} - 已包含")
                else:
                    print(f"  ❌ {mod} - 未找到")
    else:
        print(f"❌ 找不到 PYZ-00.pyz 文件")
        
except ImportError:
    print("⚠ 无法导入 PyInstaller，跳过 PYZ 文件检查")
    print("提示：可以运行 'pip install pyinstaller' 来安装")
except Exception as e:
    print(f"❌ 检查 PYZ 文件时出错: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("检查完成")
print("="*60)
