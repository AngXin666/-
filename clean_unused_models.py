#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理未被引用的YOLO模型
只保留在 page_yolo_mapping.json 中实际引用的模型
"""

import json
import os
from pathlib import Path

def load_json(file_path):
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(file_path, data):
    """保存JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        original_content = f.read()
    
    # 备份原文件
    backup_path = f"{file_path}.backup"
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(original_content)
    print(f"✓ 已备份原文件: {backup_path}")
    
    # 保存新文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_referenced_models():
    """从 page_yolo_mapping.json 获取所有被引用的模型"""
    mapping_file = "models/page_yolo_mapping.json"
    mapping_data = load_json(mapping_file)
    
    referenced_models = set()
    for page_type, page_config in mapping_data.get("mapping", {}).items():
        yolo_models = page_config.get("yolo_models", [])
        for model_info in yolo_models:
            model_key = model_info.get("model_key")
            if model_key:
                referenced_models.add(model_key)
    
    return referenced_models

def clean_unused_models():
    """清理未被引用的模型"""
    print("=" * 70)
    print("清理未被引用的YOLO模型")
    print("=" * 70)
    
    # 1. 获取被引用的模型
    print("\n【1】分析模型引用...")
    referenced_models = get_referenced_models()
    print(f"✓ 在 page_yolo_mapping.json 中找到 {len(referenced_models)} 个被引用的模型")
    
    # 2. 加载模型注册表
    registry_file = "config/yolo_model_registry.json"
    registry_data = load_json(registry_file)
    all_models = registry_data.get("models", {})
    print(f"✓ 在 yolo_model_registry.json 中找到 {len(all_models)} 个注册模型")
    
    # 3. 找出未被引用的模型
    print("\n【2】识别未引用的模型...")
    unused_models = []
    for model_key in all_models.keys():
        if model_key not in referenced_models:
            unused_models.append(model_key)
    
    if not unused_models:
        print("✓ 所有模型都被引用，无需清理")
        return
    
    print(f"✗ 找到 {len(unused_models)} 个未被引用的模型:")
    for i, model_key in enumerate(unused_models, 1):
        model_info = all_models[model_key]
        model_name = model_info.get("name", model_key)
        model_path = model_info.get("model_path", "")
        file_size = model_info.get("file_size_mb", 0)
        print(f"  {i}. {model_key}")
        print(f"     名称: {model_name}")
        print(f"     路径: {model_path}")
        print(f"     大小: {file_size} MB")
    
    # 4. 计算可节省的空间
    total_size = sum(all_models[key].get("file_size_mb", 0) for key in unused_models)
    print(f"\n✓ 删除这些模型可节省约 {total_size:.2f} MB 空间")
    
    # 5. 删除未引用的模型
    print("\n【3】从注册表中删除未引用的模型...")
    for model_key in unused_models:
        del registry_data["models"][model_key]
    
    # 6. 保存更新后的注册表
    save_json(registry_file, registry_data)
    print(f"✓ 已更新 {registry_file}")
    print(f"✓ 保留 {len(registry_data['models'])} 个被引用的模型")
    
    # 7. 显示保留的模型列表
    print("\n【4】保留的模型列表:")
    kept_models = sorted(registry_data["models"].keys())
    for i, model_key in enumerate(kept_models, 1):
        model_info = registry_data["models"][model_key]
        model_name = model_info.get("name", model_key)
        print(f"  {i}. {model_key} - {model_name}")
    
    print("\n" + "=" * 70)
    print("✓ 清理完成")
    print("=" * 70)
    print(f"\n删除前: {len(all_models)} 个模型")
    print(f"删除后: {len(registry_data['models'])} 个模型")
    print(f"已删除: {len(unused_models)} 个未引用模型")
    print(f"节省空间: {total_size:.2f} MB")
    print(f"\n备份文件: {registry_file}.backup")
    print("如需恢复，运行: copy config\\yolo_model_registry.json.backup config\\yolo_model_registry.json")

if __name__ == "__main__":
    try:
        clean_unused_models()
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
