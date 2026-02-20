#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试打包后的OCR初始化
"""

import sys
import os
from pathlib import Path

print("="*60)
print("测试OCR初始化")
print("="*60)
print()

# 检测运行环境
if getattr(sys, 'frozen', False):
    print("运行环境: 打包后的EXE")
    base_dir = Path(sys.executable).parent
else:
    print("运行环境: 开发环境")
    base_dir = Path(__file__).parent

print(f"基础目录: {base_dir}")
print(f"当前工作目录: {os.getcwd()}")
print()

# 检查rapidocr目录
rapidocr_dir = base_dir / '_internal' / 'rapidocr'
print(f"RapidOCR目录: {rapidocr_dir}")
print(f"目录存在: {rapidocr_dir.exists()}")
print()

if rapidocr_dir.exists():
    print("RapidOCR目录内容:")
    for item in rapidocr_dir.iterdir():
        if item.is_file():
            size = item.stat().st_size / 1024
            print(f"  {item.name} ({size:.1f}KB)")
        elif item.is_dir():
            print(f"  {item.name}/ (目录)")
    print()

# 尝试导入RapidOCR
print("尝试导入RapidOCR...")
try:
    from rapidocr import RapidOCR
    print("✓ RapidOCR导入成功")
except Exception as e:
    print(f"✗ RapidOCR导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 尝试初始化OCR（方法1：直接初始化）
print()
print("方法1: 直接初始化RapidOCR()...")
try:
    ocr1 = RapidOCR()
    print("✓ 直接初始化成功")
except Exception as e:
    print(f"✗ 直接初始化失败: {e}")
    import traceback
    traceback.print_exc()

# 尝试初始化OCR（方法2：切换工作目录）
print()
print("方法2: 切换到rapidocr目录后初始化...")
if rapidocr_dir.exists():
    original_cwd = os.getcwd()
    try:
        os.chdir(str(rapidocr_dir))
        print(f"  切换到: {os.getcwd()}")
        ocr2 = RapidOCR()
        print("✓ 切换目录后初始化成功")
    except Exception as e:
        print(f"✗ 切换目录后初始化失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        os.chdir(original_cwd)
        print(f"  恢复到: {os.getcwd()}")
else:
    print("  跳过（rapidocr目录不存在）")

print()
print("="*60)
print("测试完成")
print("="*60)

input("\n按任意键退出...")
