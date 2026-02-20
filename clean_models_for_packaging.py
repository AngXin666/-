#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
清理models文件夹中的训练中间文件，只保留必需的模型文件
用于打包前减小体积
"""

import os
import shutil
from pathlib import Path

def clean_models_folder():
    """清理models文件夹，只保留必需的文件"""
    
    models_dir = Path("models")
    
    if not models_dir.exists():
        print("models文件夹不存在")
        return
    
    # 需要保留的文件
    keep_files = {
        "page_classifier_pytorch_best.pth",  # 页面分类器模型
        "yolo26n.pt",  # YOLO模型
        "yolov8n.pt",  # YOLO模型
        "model_version.json",  # 模型版本信息
        "page_classes.json",  # 页面类别
        "page_yolo_mapping.json",  # YOLO映射
        "yolo_model_registry.json",  # YOLO注册表
    }
    
    # 需要删除的文件夹（训练过程文件夹）
    delete_folders = []
    
    # 统计信息
    total_size_before = 0
    total_size_after = 0
    deleted_files = 0
    deleted_folders = 0
    
    print("开始清理models文件夹...")
    print("="*60)
    
    # 遍历models文件夹
    for item in models_dir.iterdir():
        if item.is_file():
            file_size = item.stat().st_size
            total_size_before += file_size
            
            if item.name in keep_files:
                # 保留的文件
                total_size_after += file_size
                print(f"[保留] {item.name} ({file_size / 1024 / 1024:.2f} MB)")
            else:
                # 删除的文件
                print(f"[删除] {item.name} ({file_size / 1024 / 1024:.2f} MB)")
                item.unlink()
                deleted_files += 1
        
        elif item.is_dir():
            # 检查是否是训练文件夹（runs, yolo_runs等）
            if item.name in ["runs", "yolo_runs"]:
                folder_size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                total_size_before += folder_size
                print(f"[删除文件夹] {item.name} ({folder_size / 1024 / 1024:.2f} MB)")
                shutil.rmtree(item)
                deleted_folders += 1
            else:
                # 其他文件夹，递归检查
                folder_size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                total_size_before += folder_size
                total_size_after += folder_size
                print(f"[保留文件夹] {item.name} ({folder_size / 1024 / 1024:.2f} MB)")
    
    print("="*60)
    print(f"清理完成！")
    print(f"删除文件数: {deleted_files}")
    print(f"删除文件夹数: {deleted_folders}")
    print(f"清理前大小: {total_size_before / 1024 / 1024:.2f} MB")
    print(f"清理后大小: {total_size_after / 1024 / 1024:.2f} MB")
    print(f"节省空间: {(total_size_before - total_size_after) / 1024 / 1024:.2f} MB")
    print(f"压缩比例: {(1 - total_size_after / total_size_before) * 100:.1f}%")

if __name__ == "__main__":
    clean_models_folder()
