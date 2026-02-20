#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复多级嵌套目录问题
1. 移动 models/runs/detect/runs/detect/yolo_runs/ 中的所有detector到 models/runs/detect/yolo_runs/
2. 更新 config/yolo_model_registry.json 中的路径
"""

import os
import json
import shutil
from pathlib import Path

def main():
    print("=" * 60)
    print("开始修复多级嵌套目录问题")
    print("=" * 60)
    
    # 1. 检查并移动目录
    print("\n[1] 检查多级目录...")
    nested_dir = Path("models/runs/detect/runs/detect/yolo_runs")
    target_dir = Path("models/runs/detect/yolo_runs")
    
    if not nested_dir.exists():
        print(f"   ✓ 多级目录不存在: {nested_dir}")
    else:
        print(f"   ✗ 发现多级目录: {nested_dir}")
        
        # 确保目标目录存在
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # 移动所有detector子目录
        moved_count = 0
        for detector_dir in nested_dir.iterdir():
            if detector_dir.is_dir():
                target_path = target_dir / detector_dir.name
                
                # 如果目标已存在，先删除
                if target_path.exists():
                    print(f"   删除旧目录: {target_path}")
                    shutil.rmtree(target_path)
                
                print(f"   移动: {detector_dir.name}")
                shutil.move(str(detector_dir), str(target_path))
                moved_count += 1
        
        print(f"   ✓ 已移动 {moved_count} 个detector目录")
        
        # 删除空的多级目录
        try:
            # 删除 models/runs/detect/runs/detect/yolo_runs
            if nested_dir.exists() and not list(nested_dir.iterdir()):
                nested_dir.rmdir()
                print(f"   删除空目录: {nested_dir}")
            
            # 删除 models/runs/detect/runs/detect
            parent_dir = nested_dir.parent
            if parent_dir.exists() and not list(parent_dir.iterdir()):
                parent_dir.rmdir()
                print(f"   删除空目录: {parent_dir}")
            
            # 删除 models/runs/detect/runs
            grandparent_dir = parent_dir.parent
            if grandparent_dir.exists() and not list(grandparent_dir.iterdir()):
                grandparent_dir.rmdir()
                print(f"   删除空目录: {grandparent_dir}")
        except Exception as e:
            print(f"   ⚠️ 清理空目录时出错: {e}")
    
    # 2. 更新配置文件
    print("\n[2] 更新配置文件...")
    config_path = Path("config/yolo_model_registry.json")
    
    if not config_path.exists():
        print(f"   ✗ 配置文件不存在: {config_path}")
        return False
    
    # 读取配置
    with open(config_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 修复路径
    fixed_count = 0
    for model_name, model_info in data['models'].items():
        model_path = model_info.get('model_path', '')
        original_path = model_path
        
        # 检查是否包含多级目录
        if 'runs/detect/runs/detect' in model_path or 'runs\\detect\\runs\\detect' in model_path:
            # 替换路径（处理正斜杠和反斜杠）
            new_path = model_path.replace('runs/detect/runs/detect/', 'runs/detect/')
            new_path = new_path.replace('runs\\detect\\runs\\detect\\', 'runs\\detect\\')
            new_path = new_path.replace('runs/detect/runs/detect\\', 'runs/detect/')
            new_path = new_path.replace('runs\\detect\\runs\\detect/', 'runs\\detect\\')
            
            # 统一使用正斜杠
            new_path = new_path.replace('\\', '/')
            
            if new_path != original_path:
                print(f"   修复: {model_name}")
                print(f"      旧路径: {model_path}")
                print(f"      新路径: {new_path}")
                
                model_info['model_path'] = new_path
                fixed_count += 1
    
    print(f"   ✓ 已修复 {fixed_count} 个模型路径")
    
    # 保存配置
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"   ✓ 配置文件已保存: {config_path}")
    
    # 3. 验证修复结果
    print("\n[3] 验证修复结果...")
    
    # 检查是否还有多级目录
    if nested_dir.exists():
        print(f"   ✗ 多级目录仍然存在: {nested_dir}")
        return False
    else:
        print(f"   ✓ 多级目录已清理")
    
    # 检查配置文件中是否还有多级路径
    remaining_nested = 0
    for model_name, model_info in data['models'].items():
        model_path = model_info.get('model_path', '')
        if 'runs/detect/runs/detect' in model_path or 'runs\\detect\\runs\\detect' in model_path:
            remaining_nested += 1
            print(f"   ✗ 仍有嵌套路径: {model_name} -> {model_path}")
    
    if remaining_nested > 0:
        print(f"   ✗ 配置文件中仍有 {remaining_nested} 个嵌套路径")
        return False
    else:
        print(f"   ✓ 配置文件中无嵌套路径")
    
    # 检查目标目录中的detector数量
    if target_dir.exists():
        detector_count = sum(1 for d in target_dir.iterdir() if d.is_dir())
        print(f"   ✓ 目标目录包含 {detector_count} 个detector")
    
    print("\n" + "=" * 60)
    print("✅ 多级嵌套目录修复完成！")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ 修复过程中出现错误")
        exit(1)
