#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
打包前的最终检查和准备
"""

import json
import os
from pathlib import Path

def check_config_paths():
    """检查配置文件中的路径"""
    print("\n[1] 检查配置文件路径...")
    
    config_path = Path("config/yolo_model_registry.json")
    if not config_path.exists():
        print("   ✗ 配置文件不存在")
        return False
    
    with open(config_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 检查嵌套路径
    nested = []
    backslash = []
    
    for model_name, model_info in data['models'].items():
        model_path = model_info.get('model_path', '')
        
        if 'runs/detect/runs/detect' in model_path:
            nested.append(model_name)
        
        if '\\' in model_path:
            backslash.append(model_name)
    
    if nested:
        print(f"   ✗ 发现 {len(nested)} 个嵌套路径")
        for name in nested[:5]:
            print(f"      - {name}")
        return False
    else:
        print("   ✓ 无嵌套路径")
    
    if backslash:
        print(f"   ✗ 发现 {len(backslash)} 个反斜杠路径")
        for name in backslash[:5]:
            print(f"      - {name}")
        return False
    else:
        print("   ✓ 无反斜杠路径")
    
    return True

def check_directory_structure():
    """检查目录结构"""
    print("\n[2] 检查目录结构...")
    
    # 检查多级目录是否存在
    nested_dir = Path("models/runs/detect/runs/detect")
    if nested_dir.exists():
        print(f"   ✗ 多级目录仍然存在: {nested_dir}")
        return False
    else:
        print("   ✓ 无多级目录")
    
    # 检查目标目录
    target_dir = Path("models/runs/detect/yolo_runs")
    if not target_dir.exists():
        print(f"   ✗ 目标目录不存在: {target_dir}")
        return False
    
    detector_count = sum(1 for d in target_dir.iterdir() if d.is_dir())
    print(f"   ✓ 目标目录包含 {detector_count} 个detector")
    
    # 检查必需的目录
    required_dirs = [
        "models",
        "config",
        "src",
    ]
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            print(f"   ✗ 必需目录不存在: {dir_name}")
            return False
    
    print(f"   ✓ 所有必需目录存在")
    
    return True

def check_required_files():
    """检查必需的文件"""
    print("\n[3] 检查必需文件...")
    
    required_files = [
        "run.py",
        "config/config.yaml",
        "config/yolo_model_registry.json",
        "config/page_classes.json",
        "config/page_state_mapping.json",
        "models/page_classifier_pytorch_best.pth",
    ]
    
    missing = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing.append(file_path)
            print(f"   ✗ 缺失: {file_path}")
    
    if missing:
        print(f"   ✗ 缺失 {len(missing)} 个必需文件")
        return False
    else:
        print(f"   ✓ 所有必需文件存在")
    
    return True

def check_packaging_script():
    """检查打包脚本"""
    print("\n[4] 检查打包脚本...")
    
    script_path = Path("build_fixed_rapidocr.py")
    if not script_path.exists():
        print("   ✗ 打包脚本不存在")
        return False
    
    # 读取脚本内容
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查关键配置
    checks = [
        ("--add-data", "config;config", "config目录打包"),
        ("--add-data", "models;models", "models目录打包"),
        ("shutil.move", "models", "models目录移动"),
        ("shutil.move", "config", "config目录移动"),
    ]
    
    all_ok = True
    for check_type, check_str, desc in checks:
        if check_str in content:
            print(f"   ✓ {desc}")
        else:
            print(f"   ✗ {desc} - 未找到: {check_str}")
            all_ok = False
    
    return all_ok

def main():
    print("=" * 60)
    print("打包前最终检查")
    print("=" * 60)
    
    checks = [
        ("配置文件路径", check_config_paths),
        ("目录结构", check_directory_structure),
        ("必需文件", check_required_files),
        ("打包脚本", check_packaging_script),
    ]
    
    all_passed = True
    for name, check_func in checks:
        if not check_func():
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ 所有检查通过，可以开始打包！")
        print("\n执行打包命令:")
        print("  python build_fixed_rapidocr.py")
    else:
        print("❌ 检查未通过，请先修复问题")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
