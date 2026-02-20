#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
打包后模型加载测试脚本
将此脚本复制到打包后的目录中运行
"""

import os
import sys
import json
from pathlib import Path

def test_model_loading():
    print("测试模型加载...")
    print(f"Python版本: {sys.version}")
    print(f"当前目录: {os.getcwd()}")
    print(f"是否打包: {getattr(sys, 'frozen', False)}")
    
    if getattr(sys, 'frozen', False):
        base_dir = Path(sys.executable).parent
    else:
        base_dir = Path(__file__).parent
    
    print(f"基础目录: {base_dir}")
    
    # 检查目录
    print("\n检查目录:")
    for dir_name in ['config', 'models', '_internal']:
        dir_path = base_dir / dir_name
        exists = dir_path.exists()
        status = "✓" if exists else "❌"
        print(f"  {status} {dir_name}/: {exists}")
    
    # 检查关键文件
    print("\n检查关键文件:")
    files_to_check = [
        'config/yolo_model_registry.json',
        'models/page_yolo_mapping.json',
        'config/page_state_mapping.json',
        'models/page_classifier_pytorch_best.pth',
    ]
    
    for filepath in files_to_check:
        full_path = base_dir / filepath
        exists = full_path.exists()
        status = "✓" if exists else "❌"
        
        if exists:
            size_mb = full_path.stat().st_size / 1024 / 1024
            print(f"  {status} {filepath} ({size_mb:.2f} MB)")
        else:
            print(f"  {status} {filepath} (缺失)")
    
    # 尝试加载模型管理器
    print("\n尝试加载模型管理器:")
    try:
        sys.path.insert(0, str(base_dir))
        from src.model_manager import ModelManager
        
        manager = ModelManager.get_instance()
        print("  ✓ ModelManager初始化成功")
        print(f"  - 基础目录: {manager.base_dir}")
        print(f"  - 模型目录: {manager.models_dir}")
        print(f"  - 模型目录存在: {manager.models_dir.exists()}")
        
    except Exception as e:
        print(f"  ❌ ModelManager初始化失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n测试完成")
    input("按回车键退出...")

if __name__ == '__main__':
    test_model_loading()
