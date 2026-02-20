#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复打包后的文件结构
Fix Package Structure After PyInstaller Build
"""

import os
import shutil

def fix_structure():
    """修复打包后的文件结构"""
    print("正在修复打包后的文件结构...")
    
    # 打包输出目录
    dist_dir = "dist/溪盟商城自动化助手"
    
    if not os.path.exists(dist_dir):
        print(f"错误：找不到打包目录 {dist_dir}")
        return False
    
    # 需要从_internal复制到根目录的文件夹
    folders_to_copy = ['config', 'models']
    
    for folder in folders_to_copy:
        src = os.path.join(dist_dir, '_internal', folder)
        dst = os.path.join(dist_dir, folder)
        
        if os.path.exists(src):
            print(f"  复制 {folder} 文件夹...")
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"  ✓ {folder} 复制完成")
        else:
            print(f"  ⚠ 警告：找不到 {src}")
    
    print("✓ 文件结构修复完成！")
    return True

if __name__ == '__main__':
    fix_structure()
