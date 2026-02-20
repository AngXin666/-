#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查打包后的程序模型文件
用于诊断打包后程序找不到模型文件的问题
"""

import os
import sys
from pathlib import Path

def check_models():
    """检查模型文件是否存在"""
    print("=" * 60)
    print("检查打包后的模型文件")
    print("=" * 60)
    print()
    
    # 确定基础目录
    if getattr(sys, 'frozen', False):
        # 打包后的EXE环境
        base_dir = Path(sys.executable).parent
        print(f"运行环境: 打包后的EXE")
    else:
        # 开发环境
        base_dir = Path(__file__).parent
        print(f"运行环境: 开发环境")
    
    print(f"基础目录: {base_dir}")
    print(f"当前工作目录: {os.getcwd()}")
    print()
    
    # 检查models目录
    models_dir = base_dir / "models"
    print(f"模型目录: {models_dir}")
    print(f"目录存在: {models_dir.exists()}")
    print()
    
    if not models_dir.exists():
        print("❌ 模型目录不存在!")
        print()
        print("尝试查找models目录...")
        
        # 尝试在不同位置查找
        search_paths = [
            base_dir / "models",
            base_dir / "_internal" / "models",
            Path(os.getcwd()) / "models",
            Path(sys.executable).parent / "models",
        ]
        
        for path in search_paths:
            print(f"  检查: {path}")
            if path.exists():
                print(f"    ✓ 找到!")
                models_dir = path
                break
            else:
                print(f"    ✗ 不存在")
        print()
    
    # 检查必需的模型文件
    required_files = [
        'page_classifier_pytorch_best.pth',
        'yolo26n.pt',
        'yolov8n.pt',
        'model_version.json',
        'page_classes.json',
        'page_yolo_mapping.json',
        'yolo_model_registry.json'
    ]
    
    print("检查必需的模型文件:")
    print("-" * 60)
    
    all_exist = True
    for file_name in required_files:
        file_path = models_dir / file_name
        exists = file_path.exists()
        
        if exists:
            size = file_path.stat().st_size / 1024 / 1024
            print(f"  ✓ {file_name:40s} ({size:6.1f}MB)")
        else:
            print(f"  ✗ {file_name:40s} (缺失)")
            all_exist = False
    
    print()
    print("=" * 60)
    
    if all_exist:
        print("✅ 所有模型文件都存在")
    else:
        print("❌ 部分模型文件缺失")
    
    print("=" * 60)
    print()
    
    # 检查rapidocr的ONNX模型文件
    print("检查RapidOCR模型文件:")
    print("-" * 60)
    
    # 尝试找到rapidocr目录
    rapidocr_dirs = [
        base_dir / "_internal" / "rapidocr",
        base_dir / "rapidocr",
    ]
    
    rapidocr_dir = None
    for path in rapidocr_dirs:
        if path.exists():
            rapidocr_dir = path
            break
    
    if rapidocr_dir:
        print(f"RapidOCR目录: {rapidocr_dir}")
        
        # 检查必需的ONNX模型文件
        rapidocr_files = [
            'config.yaml',
            'models/ch_PP-OCRv4_det_infer.onnx',
            'models/ch_PP-OCRv4_rec_infer.onnx',
            'models/ch_ppocr_mobile_v2.0_cls_infer.onnx',
        ]
        
        rapidocr_ok = True
        for file_rel in rapidocr_files:
            file_path = rapidocr_dir / file_rel
            exists = file_path.exists()
            
            if exists:
                size = file_path.stat().st_size / 1024 / 1024
                print(f"  ✓ {file_rel:50s} ({size:6.1f}MB)")
            else:
                print(f"  ✗ {file_rel:50s} (缺失)")
                rapidocr_ok = False
        
        if not rapidocr_ok:
            print()
            print("❌ RapidOCR模型文件缺失，OCR功能将无法使用")
            all_exist = False
    else:
        print("❌ 找不到RapidOCR目录")
        all_exist = False
    
    print()
    print("=" * 60)
    
    # 列出models目录中的所有文件
    if models_dir.exists():
        print("models目录中的所有文件:")
        print("-" * 60)
        for item in sorted(models_dir.iterdir()):
            if item.is_file():
                size = item.stat().st_size / 1024 / 1024
                print(f"  {item.name:40s} ({size:6.1f}MB)")
            elif item.is_dir():
                print(f"  {item.name}/ (目录)")
        print()
    
    return all_exist

if __name__ == '__main__':
    try:
        success = check_models()
        
        print("\n按任意键退出...")
        input()
        
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 检查过程出错: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n按任意键退出...")
        input()
        
        sys.exit(1)
