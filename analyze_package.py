#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""分析打包后的文件，找出大文件和可能不需要的文件"""

import os
from pathlib import Path
from collections import defaultdict

def analyze_directory(base_dir):
    """分析目录结构和文件大小"""
    if not os.path.exists(base_dir):
        print(f"目录不存在: {base_dir}")
        return
    
    print(f"\n分析目录: {base_dir}")
    print("="*80)
    
    # 统计各个目录的大小
    dir_sizes = defaultdict(int)
    file_list = []
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            filepath = os.path.join(root, file)
            try:
                size = os.path.getsize(filepath)
                rel_path = os.path.relpath(filepath, base_dir)
                file_list.append((rel_path, size))
                
                # 统计目录大小
                dir_name = os.path.dirname(rel_path).split(os.sep)[0] if os.sep in rel_path else "根目录"
                dir_sizes[dir_name] += size
            except:
                pass
    
    # 1. 显示各目录大小
    print("\n【目录大小统计】")
    print("-"*80)
    sorted_dirs = sorted(dir_sizes.items(), key=lambda x: x[1], reverse=True)
    for dir_name, size in sorted_dirs:
        size_mb = size / 1024 / 1024
        size_gb = size / 1024 / 1024 / 1024
        if size_gb >= 0.1:
            print(f"{dir_name:30s} {size_gb:8.2f} GB")
        else:
            print(f"{dir_name:30s} {size_mb:8.2f} MB")
    
    # 2. 显示大文件（>10MB）
    print("\n【大文件列表（>10MB）】")
    print("-"*80)
    large_files = [(path, size) for path, size in file_list if size > 10 * 1024 * 1024]
    large_files.sort(key=lambda x: x[1], reverse=True)
    
    for filepath, size in large_files[:50]:  # 只显示前50个
        size_mb = size / 1024 / 1024
        print(f"{size_mb:8.2f} MB  {filepath}")
    
    # 3. 按文件类型统计
    print("\n【文件类型统计】")
    print("-"*80)
    ext_sizes = defaultdict(int)
    ext_counts = defaultdict(int)
    
    for filepath, size in file_list:
        ext = os.path.splitext(filepath)[1].lower()
        if not ext:
            ext = "<无扩展名>"
        ext_sizes[ext] += size
        ext_counts[ext] += 1
    
    sorted_exts = sorted(ext_sizes.items(), key=lambda x: x[1], reverse=True)
    for ext, size in sorted_exts[:20]:  # 只显示前20个
        count = ext_counts[ext]
        size_mb = size / 1024 / 1024
        print(f"{ext:15s} {count:6d} 个文件  {size_mb:10.2f} MB")
    
    # 4. 可能不需要的文件
    print("\n【可能不需要的文件/目录】")
    print("-"*80)
    
    unnecessary_patterns = {
        "torch": "PyTorch库（运行时可能不需要）",
        "tensorflow": "TensorFlow库（运行时可能不需要）",
        "ultralytics": "YOLO训练库（运行时不需要）",
        "torchvision": "TorchVision库（运行时可能不需要）",
        "torchaudio": "TorchAudio库（运行时不需要）",
        "matplotlib": "Matplotlib绘图库（运行时不需要）",
        "scipy": "SciPy科学计算库（运行时可能不需要）",
        "pandas": "Pandas数据分析库（运行时可能不需要）",
        "sklearn": "Scikit-learn机器学习库（运行时可能不需要）",
        "shapely": "Shapely几何库（运行时可能不需要）",
        "h5py": "HDF5文件库（运行时可能不需要）",
        "pywt": "小波变换库（运行时可能不需要）",
        ".pyc": "Python字节码文件（可以删除）",
        "__pycache__": "Python缓存目录（可以删除）",
        "test": "测试文件（可以删除）",
        "tests": "测试目录（可以删除）",
        "examples": "示例文件（可以删除）",
        "docs": "文档目录（可以删除）",
        "LICENSE": "许可证文件（可以删除）",
        "README": "说明文件（可以删除）",
    }
    
    found_unnecessary = defaultdict(list)
    
    for filepath, size in file_list:
        filepath_lower = filepath.lower()
        for pattern, reason in unnecessary_patterns.items():
            if pattern in filepath_lower:
                found_unnecessary[reason].append((filepath, size))
                break
    
    for reason, files in sorted(found_unnecessary.items()):
        total_size = sum(size for _, size in files)
        size_mb = total_size / 1024 / 1024
        print(f"\n{reason}")
        print(f"  文件数: {len(files)}, 总大小: {size_mb:.2f} MB")
        # 显示前5个最大的文件
        files.sort(key=lambda x: x[1], reverse=True)
        for filepath, size in files[:5]:
            print(f"    {size / 1024 / 1024:8.2f} MB  {filepath}")
    
    # 5. 总结
    print("\n【总结】")
    print("="*80)
    total_size = sum(size for _, size in file_list)
    print(f"总文件数: {len(file_list)}")
    print(f"总大小: {total_size / 1024 / 1024:.2f} MB ({total_size / 1024 / 1024 / 1024:.2f} GB)")

if __name__ == "__main__":
    # 检查两个可能的目录
    dirs_to_check = [
        "D:/溪盟商城自动化助手_打包",
        "dist/溪盟商城自动化助手"
    ]
    
    for dir_path in dirs_to_check:
        if os.path.exists(dir_path):
            analyze_directory(dir_path)
        else:
            print(f"目录不存在: {dir_path}")
