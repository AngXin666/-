#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验证打包结构是否正确"""

import os
from pathlib import Path

def verify_package(base_dir):
    """验证打包结构"""
    print(f"\n验证打包目录: {base_dir}")
    print("="*80)
    
    # 必需的文件和目录
    required_items = {
        '主程序': '溪盟商城自动化助手.exe',
        '_internal目录': '_internal',
        'models目录（根）': 'models',
        'models目录（_internal）': '_internal/models',
        'config目录': 'config',
        'config.yaml': 'config.yaml',
    }
    
    # 必需的模型文件
    required_models = [
        'models/yolo_runs/button_detector/weights/best.pt',
        'models/yolo_runs/checkin_detector/weights/best.pt',
        'models/yolo_runs/homepage_detector/weights/best.pt',
        'models/yolo_runs/login_detector/weights/best.pt',
        'models/yolo_runs/transfer_detector/weights/best.pt',
        'models/page_classifier_pytorch_best.pth',
        'models/yolov8n.pt',
    ]
    
    all_ok = True
    
    print("\n检查必需文件和目录:")
    for name, path in required_items.items():
        full_path = os.path.join(base_dir, path)
        if os.path.exists(full_path):
            if os.path.isfile(full_path):
                size = os.path.getsize(full_path)
                print(f"  [OK] {name}: {size / 1024 / 1024:.2f} MB")
            else:
                file_count = sum(1 for _ in Path(full_path).rglob('*') if _.is_file())
                print(f"  [OK] {name}: {file_count} 个文件")
        else:
            print(f"  [ERROR] {name}: 缺失")
            all_ok = False
    
    print("\n检查必需的模型文件:")
    for model_path in required_models:
        full_path = os.path.join(base_dir, model_path)
        if os.path.exists(full_path):
            size = os.path.getsize(full_path)
            print(f"  [OK] {model_path.split('/')[-1]}: {size / 1024 / 1024:.2f} MB")
        else:
            print(f"  [ERROR] {model_path}: 缺失")
            all_ok = False
    
    # 检查 _internal/models 是否也有这些文件
    print("\n检查 _internal/models 中的模型文件:")
    for model_path in required_models:
        internal_path = os.path.join(base_dir, '_internal', model_path)
        if os.path.exists(internal_path):
            size = os.path.getsize(internal_path)
            print(f"  [OK] _internal/{model_path.split('/')[-1]}: {size / 1024 / 1024:.2f} MB")
        else:
            print(f"  [WARN] _internal/{model_path}: 缺失（可能不影响运行）")
    
    print("\n" + "="*80)
    if all_ok:
        print("验证通过！所有必需文件都存在。")
    else:
        print("验证失败！有文件缺失。")
    print("="*80)
    
    return all_ok

if __name__ == "__main__":
    package_dir = "D:/溪盟商城自动化助手_打包"
    
    if os.path.exists(package_dir):
        verify_package(package_dir)
    else:
        print(f"打包目录不存在: {package_dir}")
