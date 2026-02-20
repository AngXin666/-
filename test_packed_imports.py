#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试打包后的程序导入问题
Test Packed Program Import Issues
"""

import sys
import os

print("=" * 60)
print("测试打包后的程序导入")
print("=" * 60)

# 设置工作目录为打包后的程序目录
packed_dir = r"D:\溪盟商城自动化助手_打包_新\溪盟商城自动化助手"

if not os.path.exists(packed_dir):
    print(f"✗ 打包目录不存在: {packed_dir}")
    sys.exit(1)

os.chdir(packed_dir)
sys.path.insert(0, packed_dir)

print(f"\n当前目录: {os.getcwd()}")
print(f"sys.path: {sys.path[:3]}")

# 测试基础导入
print("\n[测试1] 基础模块导入...")
try:
    import yaml
    print("  ✓ yaml")
except Exception as e:
    print(f"  ✗ yaml: {e}")

try:
    import PIL
    print("  ✓ PIL")
except Exception as e:
    print(f"  ✗ PIL: {e}")

try:
    import cv2
    print("  ✓ cv2")
except Exception as e:
    print(f"  ✗ cv2: {e}")

try:
    import torch
    print("  ✓ torch")
except Exception as e:
    print(f"  ✗ torch: {e}")

try:
    import rapidocr_onnxruntime
    print("  ✓ rapidocr_onnxruntime")
except Exception as e:
    print(f"  ✗ rapidocr_onnxruntime: {e}")

# 测试src模块导入
print("\n[测试2] src模块导入...")

# 检查src目录是否存在
if os.path.exists('src'):
    print("  ✓ src目录存在")
else:
    print("  ✗ src目录不存在")
    # 检查_internal目录
    if os.path.exists('_internal/src'):
        print("  ✓ _internal/src目录存在")
        sys.path.insert(0, '_internal')
    else:
        print("  ✗ _internal/src目录也不存在")

try:
    import src
    print(f"  ✓ src (路径: {src.__file__ if hasattr(src, '__file__') else 'N/A'})")
except Exception as e:
    print(f"  ✗ src: {e}")
    import traceback
    traceback.print_exc()

try:
    from src import adb_bridge
    print("  ✓ src.adb_bridge")
except Exception as e:
    print(f"  ✗ src.adb_bridge: {e}")
    import traceback
    traceback.print_exc()

try:
    from src import page_state_dynamic
    print("  ✓ src.page_state_dynamic")
except Exception as e:
    print(f"  ✗ src.page_state_dynamic: {e}")
    import traceback
    traceback.print_exc()

try:
    from src import page_detector_integrated
    print("  ✓ src.page_detector_integrated")
except Exception as e:
    print(f"  ✗ src.page_detector_integrated: {e}")
    import traceback
    traceback.print_exc()

try:
    from src import auto_login
    print("  ✓ src.auto_login")
except Exception as e:
    print(f"  ✗ src.auto_login: {e}")
    import traceback
    traceback.print_exc()

try:
    from src import daily_checkin
    print("  ✓ src.daily_checkin")
except Exception as e:
    print(f"  ✗ src.daily_checkin: {e}")
    import traceback
    traceback.print_exc()

# 测试配置文件加载
print("\n[测试3] 配置文件加载...")

config_files = [
    'config/page_state_mapping.json',
    'config/page_classes.json',
    'config/yolo_model_registry.json',
    'config.yaml',
]

for config_file in config_files:
    if os.path.exists(config_file):
        print(f"  ✓ {config_file}")
    else:
        print(f"  ✗ {config_file} (不存在)")

# 测试模型文件
print("\n[测试4] 模型文件...")

model_files = [
    'models/page_classifier_pytorch_best.pth',
    'models/page_classes.json',
    'models/yolo26n.pt',
]

for model_file in model_files:
    if os.path.exists(model_file):
        size = os.path.getsize(model_file) / 1024 / 1024
        print(f"  ✓ {model_file} ({size:.1f} MB)")
    else:
        print(f"  ✗ {model_file} (不存在)")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
