#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
彻底修复所有路径问题
1. 统一使用正斜杠
2. 去除所有嵌套的 runs/detect
"""

import json
import re
from pathlib import Path

def normalize_path(path):
    """规范化路径：统一使用正斜杠，去除嵌套"""
    if not path:
        return path
    
    # 统一使用正斜杠
    path = path.replace('\\', '/')
    
    # 去除嵌套的 runs/detect/runs/detect
    # 使用正则表达式处理所有可能的嵌套情况
    while 'runs/detect/runs/detect' in path:
        path = path.replace('runs/detect/runs/detect', 'runs/detect')
    
    return path

def main():
    print("=" * 60)
    print("彻底修复所有路径问题")
    print("=" * 60)
    
    config_path = Path("config/yolo_model_registry.json")
    
    if not config_path.exists():
        print(f"✗ 配置文件不存在: {config_path}")
        return False
    
    # 读取配置
    print("\n[1] 读取配置文件...")
    with open(config_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"   ✓ 找到 {len(data['models'])} 个模型")
    
    # 修复所有路径
    print("\n[2] 修复路径...")
    fixed_count = 0
    
    for model_name, model_info in data['models'].items():
        model_path = model_info.get('model_path', '')
        if not model_path:
            continue
        
        # 规范化路径
        new_path = normalize_path(model_path)
        
        # 如果路径改变了，记录并更新
        if new_path != model_path:
            print(f"\n   修复: {model_name}")
            print(f"      旧: {model_path}")
            print(f"      新: {new_path}")
            
            model_info['model_path'] = new_path
            fixed_count += 1
    
    print(f"\n   ✓ 共修复 {fixed_count} 个路径")
    
    # 保存配置
    print("\n[3] 保存配置...")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"   ✓ 配置已保存")
    
    # 验证结果
    print("\n[4] 验证结果...")
    
    # 检查是否还有嵌套路径
    nested_count = 0
    backslash_count = 0
    
    for model_name, model_info in data['models'].items():
        model_path = model_info.get('model_path', '')
        
        # 检查嵌套
        if 'runs/detect/runs/detect' in model_path:
            nested_count += 1
            print(f"   ✗ 仍有嵌套: {model_name} -> {model_path}")
        
        # 检查反斜杠
        if '\\' in model_path:
            backslash_count += 1
            print(f"   ✗ 仍有反斜杠: {model_name} -> {model_path}")
    
    if nested_count == 0:
        print("   ✓ 无嵌套路径")
    else:
        print(f"   ✗ 仍有 {nested_count} 个嵌套路径")
    
    if backslash_count == 0:
        print("   ✓ 无反斜杠路径")
    else:
        print(f"   ✗ 仍有 {backslash_count} 个反斜杠路径")
    
    # 显示一些示例路径
    print("\n[5] 示例路径:")
    import random
    samples = random.sample(list(data['models'].items()), min(5, len(data['models'])))
    for name, info in samples:
        print(f"   {name}:")
        print(f"      {info.get('model_path', '')}")
    
    print("\n" + "=" * 60)
    if nested_count == 0 and backslash_count == 0:
        print("✅ 所有路径已修复！")
    else:
        print("⚠️ 仍有问题需要修复")
    print("=" * 60)
    
    return nested_count == 0 and backslash_count == 0

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ 修复未完成")
        exit(1)
