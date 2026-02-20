#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""增强版打包清理脚本 - 只保留运行时必需的文件"""

import os
import shutil
from pathlib import Path

def clean_package(base_dir):
    """清理打包后不需要的文件"""
    if not os.path.exists(base_dir):
        print(f"目录不存在: {base_dir}")
        return False
    
    print(f"\n开始清理打包目录: {base_dir}")
    print("="*80)
    
    total_deleted = 0
    total_size = 0
    
    # 1. 删除TensorFlow库（完全不需要）
    print("\n[1/8] 删除TensorFlow库...")
    tensorflow_patterns = [
        '_internal/tensorflow',
        '_internal\\tensorflow',
    ]
    for pattern in tensorflow_patterns:
        tf_dir = os.path.join(base_dir, pattern.replace('/', os.sep))
        if os.path.exists(tf_dir):
            size = sum(f.stat().st_size for f in Path(tf_dir).rglob('*') if f.is_file())
            shutil.rmtree(tf_dir)
            total_deleted += 1
            total_size += size
            print(f"  删除目录: {pattern} ({size / 1024 / 1024:.2f} MB)")
    
    # 2. 删除PyTorch CUDA库（保留CPU推理需要的nvrtc和nvJitLink）
    print("\n[2/8] 删除PyTorch CUDA库（保留nvrtc和nvJitLink）...")
    cuda_patterns = [
        'torch_cuda*.dll',
        'cublas*.dll',
        'cublasLt*.dll',
        'cufft*.dll',
        'cusparse*.dll',
        'cusolver*.dll',
        'cusolverMg*.dll',
        'curand*.dll',
        'cudnn*.dll',
        # 注意：不删除 nvJitLink*.dll 和 nvrtc*.dll，CPU推理必需
    ]
    
    internal_dir = os.path.join(base_dir, '_internal')
    if os.path.exists(internal_dir):
        for pattern in cuda_patterns:
            for file_path in Path(internal_dir).rglob(pattern):
                if file_path.is_file():
                    size = file_path.stat().st_size
                    file_path.unlink()
                    total_deleted += 1
                    total_size += size
                    if size > 10 * 1024 * 1024:
                        print(f"  删除: {file_path.name} ({size / 1024 / 1024:.2f} MB)")
    
    # 3. 删除训练图片（.jpg, .png）
    print("\n[3/8] 删除训练图片...")
    img_count = 0
    img_size = 0
    for pattern in ['**/*.jpg', '**/*.png', '**/*.jpeg']:
        for file_path in Path(base_dir).rglob(pattern):
            if file_path.is_file() and 'models' in str(file_path):
                try:
                    size = file_path.stat().st_size
                    file_path.unlink()
                    img_count += 1
                    img_size += size
                except:
                    pass
    if img_count > 0:
        print(f"  删除训练图片: {img_count} 个文件, {img_size / 1024 / 1024:.2f} MB")
        total_deleted += img_count
        total_size += img_size
    
    # 4. 删除YOLO训练权重（last.pt），只保留best.pt
    print("\n[4/8] 删除YOLO训练权重（last.pt）...")
    last_pt_count = 0
    last_pt_size = 0
    for file_path in Path(base_dir).rglob('last.pt'):
        if file_path.is_file():
            size = file_path.stat().st_size
            file_path.unlink()
            last_pt_count += 1
            last_pt_size += size
    if last_pt_count > 0:
        print(f"  删除last.pt: {last_pt_count} 个文件, {last_pt_size / 1024 / 1024:.2f} MB")
        total_deleted += last_pt_count
        total_size += last_pt_size
    
    # 5. 删除备份文件（.backup_epoch4等）
    print("\n[5/8] 删除备份文件...")
    backup_count = 0
    backup_size = 0
    for pattern in ['**/*.backup*', '**/*_backup*']:
        for file_path in Path(base_dir).rglob(pattern.split('/')[-1]):
            if file_path.is_file():
                size = file_path.stat().st_size
                file_path.unlink()
                backup_count += 1
                backup_size += size
    if backup_count > 0:
        print(f"  删除备份文件: {backup_count} 个文件, {backup_size / 1024 / 1024:.2f} MB")
        total_deleted += backup_count
        total_size += backup_size
    
    # 6. 删除不需要的库
    print("\n[6/8] 删除不需要的库...")
    unnecessary_libs = [
        'pandas',
        'sklearn',
        'shapely',
        'h5py',
        'pywt',
        'matplotlib',
        'scipy',
        'ultralytics',  # YOLO训练库
        'torchvision',
        'torchaudio',
    ]
    
    if os.path.exists(internal_dir):
        for lib_name in unnecessary_libs:
            for lib_dir in Path(internal_dir).glob(lib_name):
                if lib_dir.is_dir():
                    size = sum(f.stat().st_size for f in lib_dir.rglob('*') if f.is_file())
                    shutil.rmtree(lib_dir)
                    total_deleted += 1
                    total_size += size
                    print(f"  删除库: {lib_name} ({size / 1024 / 1024:.2f} MB)")
            
            # 也删除.libs目录
            for lib_dir in Path(internal_dir).glob(f'{lib_name}.libs'):
                if lib_dir.is_dir():
                    size = sum(f.stat().st_size for f in lib_dir.rglob('*') if f.is_file())
                    shutil.rmtree(lib_dir)
                    total_deleted += 1
                    total_size += size
                    print(f"  删除库: {lib_name}.libs ({size / 1024 / 1024:.2f} MB)")
    
    # 7. 删除torch目录（保留必要的推理文件）
    print("\n[7/8] 清理torch目录...")
    torch_dir = os.path.join(internal_dir, 'torch')
    if os.path.exists(torch_dir):
        # 删除不需要的子目录
        torch_unnecessary = ['include', 'share', 'test', 'testing', '_inductor']
        for subdir in torch_unnecessary:
            subdir_path = os.path.join(torch_dir, subdir)
            if os.path.exists(subdir_path):
                size = sum(f.stat().st_size for f in Path(subdir_path).rglob('*') if f.is_file())
                shutil.rmtree(subdir_path)
                total_deleted += 1
                total_size += size
                print(f"  删除torch子目录: {subdir} ({size / 1024 / 1024:.2f} MB)")
    
    # 8. 删除重复的models目录（_internal中的）
    print("\n[8/8] 删除重复的models目录...")
    internal_models = os.path.join(internal_dir, 'models')
    root_models = os.path.join(base_dir, 'models')
    if os.path.exists(internal_models) and os.path.exists(root_models):
        size = sum(f.stat().st_size for f in Path(internal_models).rglob('*') if f.is_file())
        shutil.rmtree(internal_models)
        total_deleted += 1
        total_size += size
        print(f"  删除_internal/models（已有根目录models）: {size / 1024 / 1024:.2f} MB")
    
    # 总结
    print("\n" + "="*80)
    print(f"清理完成！")
    print(f"删除项目数: {total_deleted}")
    print(f"释放空间: {total_size / 1024 / 1024:.2f} MB ({total_size / 1024 / 1024 / 1024:.2f} GB)")
    
    # 显示清理后的大小
    if os.path.exists(base_dir):
        final_size = sum(f.stat().st_size for f in Path(base_dir).rglob('*') if f.is_file())
        print(f"清理后大小: {final_size / 1024 / 1024:.2f} MB ({final_size / 1024 / 1024 / 1024:.2f} GB)")
    
    return True

if __name__ == "__main__":
    # 检查可能的打包目录
    dirs_to_clean = [
        "打包测试",
        "D:/溪盟商城自动化助手_打包",
        "dist/溪盟商城自动化助手"
    ]
    
    cleaned = False
    for dir_path in dirs_to_clean:
        if os.path.exists(dir_path):
            print(f"\n找到打包目录: {dir_path}")
            clean_package(dir_path)
            cleaned = True
    
    if not cleaned:
        print("未找到打包目录！")
        print("请确保以下目录之一存在：")
        for dir_path in dirs_to_clean:
            print(f"  - {dir_path}")
