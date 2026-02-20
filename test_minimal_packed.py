#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最小化测试打包后程序
"""

import sys
import os

# 切换到打包目录
os.chdir(r"D:\溪盟商城自动化助手_打包")

# 添加到路径
sys.path.insert(0, r"D:\溪盟商城自动化助手_打包")
sys.path.insert(0, r"D:\溪盟商城自动化助手_打包\_internal")

print("="*60)
print("最小化测试")
print("="*60)

try:
    print("\n[1] 测试基础导入...")
    import tkinter
    print("  ✓ tkinter")
    
    import yaml
    print("  ✓ yaml")
    
    import cv2
    print("  ✓ cv2")
    
    from PIL import Image
    print("  ✓ PIL")
    
    import numpy
    print("  ✓ numpy")
    
    print("\n[2] 测试torch导入...")
    import torch
    print(f"  ✓ torch {torch.__version__}")
    
    print("\n[3] 测试ultralytics导入...")
    from ultralytics import YOLO
    print("  ✓ ultralytics.YOLO")
    
    print("\n[4] 测试加载YOLO模型...")
    model_path = r"D:\溪盟商城自动化助手_打包\models\homepage_detector\weights\best.pt"
    if os.path.exists(model_path):
        model = YOLO(model_path)
        print(f"  ✓ 模型加载成功: {model_path}")
    else:
        print(f"  ✗ 模型文件不存在: {model_path}")
    
    print("\n✓ 所有测试通过！")
    
except Exception as e:
    print(f"\n✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
