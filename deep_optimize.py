#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""深度优化 - 删除 TensorFlow 和 PyTorch CUDA"""

import os
import shutil
from pathlib import Path

OUTPUT_DIR = "D:/溪盟商城自动化助手_打包"

def deep_optimize():
    """深度优化"""
    print("\n深度优化打包...")
    print("="*80)
    
    internal_dir = os.path.join(OUTPUT_DIR, '_internal')
    if not os.path.exists(internal_dir):
        print("错误：_internal 目录不存在")
        return False
    
    total_size = 0
    
    # 1. 删除 TensorFlow
    print("\n[1/3] 删除 TensorFlow...")
    tf_dir = os.path.join(internal_dir, 'tensorflow')
    if os.path.exists(tf_dir):
        size = sum(f.stat().st_size for f in Path(tf_dir).rglob('*') if f.is_file())
        shutil.rmtree(tf_dir)
        total_size += size
        print(f"  删除: tensorflow ({size / 1024 / 1024:.2f} MB)")
    
    # 2. 删除 PyTorch CUDA 库（保留 nvrtc 和 nvJitLink）
    print("\n[2/3] 删除 PyTorch CUDA 库...")
    cuda_patterns = [
        'torch_cuda.dll',
        'cublas*.dll', 'cublasLt*.dll', 'cufft*.dll',
        'cusparse*.dll', 'cusolver*.dll', 'cusolverMg*.dll', 'curand*.dll',
        'cudnn*.dll',
    ]
    
    for pattern in cuda_patterns:
        for file_path in Path(internal_dir).rglob(pattern):
            if file_path.is_file():
                size = file_path.stat().st_size
                file_path.unlink()
                total_size += size
                if size > 10 * 1024 * 1024:
                    print(f"  删除: {file_path.name} ({size / 1024 / 1024:.2f} MB)")
    
    # 3. 删除不需要的库
    print("\n[3/3] 删除不需要的库...")
    unnecessary_libs = [
        'pandas', 'sklearn', 'shapely', 'h5py', 'pywt',
        'matplotlib', 'scipy', 'ultralytics', 'torchvision', 'torchaudio',
    ]
    
    for lib_name in unnecessary_libs:
        for lib_dir in Path(internal_dir).glob(lib_name):
            if lib_dir.is_dir():
                size = sum(f.stat().st_size for f in lib_dir.rglob('*') if f.is_file())
                shutil.rmtree(lib_dir)
                total_size += size
                if size > 1024 * 1024:
                    print(f"  删除: {lib_name} ({size / 1024 / 1024:.2f} MB)")
        
        for lib_dir in Path(internal_dir).glob(f'{lib_name}.libs'):
            if lib_dir.is_dir():
                size = sum(f.stat().st_size for f in lib_dir.rglob('*') if f.is_file())
                shutil.rmtree(lib_dir)
                total_size += size
    
    # 清理 torch 目录
    torch_dir = os.path.join(internal_dir, 'torch')
    if os.path.exists(torch_dir):
        torch_unnecessary = ['include', 'share', 'test', 'testing', '_inductor']
        for subdir in torch_unnecessary:
            subdir_path = os.path.join(torch_dir, subdir)
            if os.path.exists(subdir_path):
                size = sum(f.stat().st_size for f in Path(subdir_path).rglob('*') if f.is_file())
                shutil.rmtree(subdir_path)
                total_size += size
    
    print("\n" + "="*80)
    print(f"深度优化完成，释放: {total_size / 1024 / 1024:.2f} MB ({total_size / 1024 / 1024 / 1024:.2f} GB)")
    
    # 显示最终大小
    final_size = sum(f.stat().st_size for f in Path(OUTPUT_DIR).rglob('*') if f.is_file())
    print(f"最终大小: {final_size / 1024 / 1024:.2f} MB ({final_size / 1024 / 1024 / 1024:.2f} GB)")
    
    return True

if __name__ == "__main__":
    # 显示优化前大小
    before_size = sum(f.stat().st_size for f in Path(OUTPUT_DIR).rglob('*') if f.is_file())
    print(f"优化前大小: {before_size / 1024 / 1024:.2f} MB ({before_size / 1024 / 1024 / 1024:.2f} GB)")
    
    deep_optimize()
