#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""修复 models 目录 - 将根目录的 models 复制回 _internal"""

import os
import shutil
from pathlib import Path

def fix_models_directory(base_dir):
    """修复 models 目录"""
    if not os.path.exists(base_dir):
        print(f"目录不存在: {base_dir}")
        return False
    
    print(f"\n修复 models 目录: {base_dir}")
    print("="*80)
    
    root_models = os.path.join(base_dir, 'models')
    internal_models = os.path.join(base_dir, '_internal', 'models')
    
    # 检查根目录的 models 是否存在
    if not os.path.exists(root_models):
        print(f"错误：根目录的 models 不存在: {root_models}")
        return False
    
    # 如果 _internal/models 已存在，先删除
    if os.path.exists(internal_models):
        print(f"删除旧的 _internal/models...")
        shutil.rmtree(internal_models)
    
    # 复制 models 到 _internal
    print(f"复制 models 到 _internal...")
    shutil.copytree(root_models, internal_models)
    
    # 统计文件数
    file_count = sum(1 for _ in Path(internal_models).rglob('*') if _.is_file())
    dir_size = sum(f.stat().st_size for f in Path(internal_models).rglob('*') if f.is_file())
    
    print(f"复制完成: {file_count} 个文件, {dir_size / 1024 / 1024:.2f} MB")
    print("="*80)
    
    return True

if __name__ == "__main__":
    package_dir = "D:/溪盟商城自动化助手_打包"
    
    if os.path.exists(package_dir):
        fix_models_directory(package_dir)
        
        # 显示最终大小
        final_size = sum(f.stat().st_size for f in Path(package_dir).rglob('*') if f.is_file())
        print(f"\n最终大小: {final_size / 1024 / 1024:.2f} MB ({final_size / 1024 / 1024 / 1024:.2f} GB)")
    else:
        print(f"打包目录不存在: {package_dir}")
