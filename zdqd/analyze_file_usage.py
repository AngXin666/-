#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""分析 src 目录中文件的使用情况"""

import os
import re
from pathlib import Path

def analyze_file_usage():
    src_dir = Path("src")
    
    # 获取所有 Python 文件
    all_files = set()
    for file in src_dir.glob("**/*.py"):
        if file.name != "__init__.py":
            all_files.add(file.stem)
    
    # 主入口文件
    entry_points = [
        "src/gui.py",
        "src/orchestrator.py", 
        "src/main.py",
        "run.py"
    ]
    
    # 收集所有导入
    imported_modules = set()
    
    for entry in entry_points:
        if not os.path.exists(entry):
            continue
            
        with open(entry, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 查找 from .xxx import 或 from src.xxx import
        imports = re.findall(r'from \.(\w+)|from src\.(\w+)', content)
        for imp in imports:
            module = imp[0] or imp[1]
            if module:
                imported_modules.add(module)
    
    # 递归查找被导入模块的依赖
    def find_dependencies(module_name, visited=None):
        if visited is None:
            visited = set()
        
        if module_name in visited:
            return visited
        
        visited.add(module_name)
        
        module_file = src_dir / f"{module_name}.py"
        if not module_file.exists():
            return visited
        
        with open(module_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        imports = re.findall(r'from \.(\w+)|from src\.(\w+)', content)
        for imp in imports:
            dep = imp[0] or imp[1]
            if dep and dep not in visited:
                find_dependencies(dep, visited)
        
        return visited
    
    # 查找所有依赖
    all_used = set()
    for module in imported_modules:
        all_used.update(find_dependencies(module))
    
    # 分类文件
    empty_files = []
    backup_files = []
    unused_files = []
    used_files = []
    
    for file in src_dir.glob("*.py"):
        if file.name == "__init__.py":
            continue
            
        file_size = file.stat().st_size
        file_stem = file.stem
        
        # 空文件
        if file_size == 0:
            empty_files.append(file.name)
        # 备份文件
        elif "backup" in file.name.lower() or "optimized" in file.name.lower():
            backup_files.append(file.name)
        # 未使用的文件
        elif file_stem not in all_used:
            unused_files.append(file.name)
        # 使用中的文件
        else:
            used_files.append(file.name)
    
    # 输出结果
    print("=" * 80)
    print("文件使用情况分析")
    print("=" * 80)
    
    print(f"\n📊 统计:")
    print(f"  总文件数: {len(all_files)}")
    print(f"  使用中: {len(used_files)}")
    print(f"  未使用: {len(unused_files)}")
    print(f"  空文件: {len(empty_files)}")
    print(f"  备份文件: {len(backup_files)}")
    
    if empty_files:
        print(f"\n🗑️  空文件 ({len(empty_files)}):")
        for f in sorted(empty_files):
            print(f"  - {f}")
    
    if backup_files:
        print(f"\n📦 备份/优化版本文件 ({len(backup_files)}):")
        for f in sorted(backup_files):
            size = (src_dir / f).stat().st_size
            print(f"  - {f} ({size:,} bytes)")
    
    if unused_files:
        print(f"\n⚠️  未使用的文件 ({len(unused_files)}):")
        for f in sorted(unused_files):
            size = (src_dir / f).stat().st_size
            print(f"  - {f} ({size:,} bytes)")
    
    print(f"\n✅ 使用中的文件 ({len(used_files)}):")
    for f in sorted(used_files):
        size = (src_dir / f).stat().st_size
        print(f"  - {f} ({size:,} bytes)")
    
    # 计算可清理的空间
    total_cleanup = sum((src_dir / f).stat().st_size for f in empty_files + backup_files + unused_files)
    print(f"\n💾 可清理空间: {total_cleanup:,} bytes ({total_cleanup / 1024 / 1024:.2f} MB)")

if __name__ == "__main__":
    analyze_file_usage()
