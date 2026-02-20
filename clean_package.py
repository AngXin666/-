#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
清理打包后的大文件
"""

import os
import sys
from pathlib import Path

# 打包目录
PACKAGE_DIR = "D:/溪盟商城自动化助手_打包"

def clean_large_files():
    """清理不需要的大文件"""
    if not os.path.exists(PACKAGE_DIR):
        print(f"目录不存在: {PACKAGE_DIR}")
        return
    
    print(f"开始清理: {PACKAGE_DIR}")
    
    # 需要删除的文件模式
    patterns = [
        # CUDA库
        '_internal/torch_cuda*.dll',
        '_internal/cublas*.dll',
        '_internal/cublasLt*.dll',
        '_internal/cufft*.dll',
        '_internal/cusparse*.dll',
        '_internal/cusolver*.dll',
        '_internal/cusolverMg*.dll',
        '_internal/curand*.dll',
        '_internal/cudnn*.dll',
        '_internal/nvJitLink*.dll',
        '_internal/nvrtc*.dll',
        # TensorFlow
        '_internal/_pywrap_tensorflow*.dll',
        '_internal/_pywrap_tensorflow*.pyd',
        # 训练图片
        'models/**/*.jpg',
        'models/**/*.png',
    ]
    
    total_deleted = 0
    total_size = 0
    
    for pattern in patterns:
        full_pattern = os.path.join(PACKAGE_DIR, pattern)
        for file_path in Path(PACKAGE_DIR).glob(pattern):
            if file_path.is_file():
                try:
                    size = file_path.stat().st_size
                    file_path.unlink()
                    total_deleted += 1
                    total_size += size
                    if size > 10 * 1024 * 1024:  # 大于10MB
                        print(f"  删除: {file_path.name} ({size / 1024 / 1024:.2f} MB)")
                except Exception as e:
                    print(f"  失败: {file_path.name} - {e}")
    
    print(f"\n清理完成:")
    print(f"  删除文件数: {total_deleted}")
    print(f"  释放空间: {total_size / 1024 / 1024:.2f} MB")
    
    # 计算清理后的总大小
    if os.path.exists(PACKAGE_DIR):
        total_size_after = 0
        for dirpath, dirnames, filenames in os.walk(PACKAGE_DIR):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size_after += os.path.getsize(filepath)
        print(f"  清理后总大小: {total_size_after / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    clean_large_files()
