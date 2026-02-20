#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""进一步优化打包大小 - 删除冗余的 YOLO 模型和不需要的文件"""

import os
import shutil
from pathlib import Path

def optimize_package(base_dir):
    """进一步优化打包后的文件"""
    if not os.path.exists(base_dir):
        print(f"目录不存在: {base_dir}")
        return False
    
    print(f"\n开始进一步优化打包目录: {base_dir}")
    print("="*80)
    
    total_deleted = 0
    total_size = 0
    
    # 1. 删除重复的 YOLO 模型文件（保留每个检测器的最新版本）
    print("\n[1/3] 清理重复的 YOLO 模型文件...")
    models_dir = os.path.join(base_dir, 'models')
    
    if os.path.exists(models_dir):
        # 需要保留的 best.pt 文件（每个检测器只保留一个）
        keep_patterns = [
            'yolo_runs/button_detector/weights/best.pt',
            'yolo_runs/checkin_detector/weights/best.pt',
            'yolo_runs/homepage_detector/weights/best.pt',
            'yolo_runs/login_detector/weights/best.pt',
            'yolo_runs/transfer_detector/weights/best.pt',
        ]
        
        # 转换为绝对路径
        keep_files = set()
        for pattern in keep_patterns:
            full_path = os.path.join(models_dir, pattern.replace('/', os.sep))
            if os.path.exists(full_path):
                keep_files.add(os.path.normpath(full_path))
                print(f"  保留: {pattern}")
        
        # 删除所有其他的 best.pt 文件
        deleted_count = 0
        deleted_size = 0
        for file_path in Path(models_dir).rglob('best.pt'):
            if os.path.normpath(file_path) not in keep_files:
                size = file_path.stat().st_size
                file_path.unlink()
                deleted_count += 1
                deleted_size += size
        
        if deleted_count > 0:
            print(f"  删除冗余模型: {deleted_count} 个文件, {deleted_size / 1024 / 1024:.2f} MB")
            total_deleted += deleted_count
            total_size += deleted_size
        else:
            print(f"  没有找到需要删除的冗余模型")
        
        # 删除 runs 目录（训练过程文件）
        runs_dir = os.path.join(models_dir, 'runs')
        if os.path.exists(runs_dir):
            size = sum(f.stat().st_size for f in Path(runs_dir).rglob('*') if f.is_file())
            shutil.rmtree(runs_dir)
            total_deleted += 1
            total_size += size
            print(f"  删除训练目录: runs ({size / 1024 / 1024:.2f} MB)")
    
    # 2. 删除 OpenCV FFmpeg DLL（视频处理，项目不需要）
    print("\n[2/3] 删除 OpenCV FFmpeg DLL（视频处理）...")
    internal_dir = os.path.join(base_dir, '_internal')
    
    if os.path.exists(internal_dir):
        ffmpeg_patterns = ['*ffmpeg*.dll']
        for pattern in ffmpeg_patterns:
            for file_path in Path(internal_dir).rglob(pattern):
                if file_path.is_file():
                    size = file_path.stat().st_size
                    file_path.unlink()
                    total_deleted += 1
                    total_size += size
                    print(f"  删除: {file_path.name} ({size / 1024 / 1024:.2f} MB)")
    
    # 3. 删除 _internal 中重复的 models 目录（如果存在）
    print("\n[3/3] 检查并删除重复的 models 目录...")
    internal_models = os.path.join(internal_dir, 'models')
    root_models = os.path.join(base_dir, 'models')
    
    if os.path.exists(internal_models) and os.path.exists(root_models):
        size = sum(f.stat().st_size for f in Path(internal_models).rglob('*') if f.is_file())
        shutil.rmtree(internal_models)
        total_deleted += 1
        total_size += size
        print(f"  删除 _internal/models: {size / 1024 / 1024:.2f} MB")
    else:
        print(f"  没有找到重复的 models 目录")
    
    # 总结
    print("\n" + "="*80)
    print(f"优化完成！")
    print(f"删除项目数: {total_deleted}")
    print(f"释放空间: {total_size / 1024 / 1024:.2f} MB ({total_size / 1024 / 1024 / 1024:.2f} GB)")
    
    # 显示优化后的大小
    if os.path.exists(base_dir):
        final_size = sum(f.stat().st_size for f in Path(base_dir).rglob('*') if f.is_file())
        print(f"优化后大小: {final_size / 1024 / 1024:.2f} MB ({final_size / 1024 / 1024 / 1024:.2f} GB)")
    
    return True

if __name__ == "__main__":
    package_dir = "D:/溪盟商城自动化助手_打包"
    
    if os.path.exists(package_dir):
        print(f"\n找到打包目录: {package_dir}")
        
        # 显示优化前的大小
        before_size = sum(f.stat().st_size for f in Path(package_dir).rglob('*') if f.is_file())
        print(f"优化前大小: {before_size / 1024 / 1024:.2f} MB ({before_size / 1024 / 1024 / 1024:.2f} GB)")
        
        # 执行优化
        optimize_package(package_dir)
    else:
        print(f"打包目录不存在: {package_dir}")
