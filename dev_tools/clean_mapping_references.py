#!/usr/bin/env python3
"""清理page_yolo_mapping.json中对已删除模型的引用"""
import json
from pathlib import Path

def clean_mapping_file(mapping_path, registry_path):
    """清理映射文件中的无效模型引用
    
    Args:
        mapping_path: 映射文件路径
        registry_path: 注册表文件路径
    """
    # 读取注册表，获取有效的模型key列表
    with open(registry_path, 'r', encoding='utf-8') as f:
        registry = json.load(f)
        valid_keys = set(registry['models'].keys())
    
    print(f"注册表中有效的模型key: {len(valid_keys)} 个")
    
    # 读取映射文件
    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    
    # 统计
    total_removed = 0
    pages_cleaned = []
    
    # 遍历所有页面映射
    for page_name, page_config in mapping['mapping'].items():
        yolo_models = page_config.get('yolo_models', [])
        
        if not yolo_models:
            continue
        
        # 过滤出有效的模型引用
        valid_models = []
        removed_models = []
        
        for model in yolo_models:
            model_key = model.get('model_key')
            if model_key in valid_keys:
                valid_models.append(model)
            else:
                removed_models.append(model_key)
                total_removed += 1
        
        # 如果有模型被删除，更新映射
        if removed_models:
            pages_cleaned.append(page_name)
            print(f"\n页面: {page_name}")
            print(f"  删除的模型引用: {removed_models}")
            print(f"  保留的模型: {len(valid_models)} 个")
            
            # 更新优先级
            for i, model in enumerate(valid_models, 1):
                model['priority'] = i
            
            page_config['yolo_models'] = valid_models
    
    # 保存清理后的映射文件
    mapping['last_updated'] = '2026-02-22'
    
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print("清理完成")
    print("=" * 60)
    print(f"清理的页面数: {len(pages_cleaned)}")
    print(f"删除的模型引用总数: {total_removed}")
    
    if pages_cleaned:
        print(f"\n清理的页面:")
        for page in pages_cleaned:
            print(f"  - {page}")

if __name__ == '__main__':
    print("=" * 60)
    print("清理page_yolo_mapping.json中的无效模型引用")
    print("=" * 60)
    
    # 清理models目录下的映射文件
    models_mapping = Path("models/page_yolo_mapping.json")
    models_registry = Path("models/yolo_model_registry.json")
    
    if models_mapping.exists() and models_registry.exists():
        print("\n清理 models/page_yolo_mapping.json...")
        print("-" * 60)
        clean_mapping_file(models_mapping, models_registry)
    
    # 清理config目录下的映射文件
    config_mapping = Path("config/page_yolo_mapping.json")
    config_registry = Path("config/yolo_model_registry.json")
    
    if config_mapping.exists() and config_registry.exists():
        print("\n\n清理 config/page_yolo_mapping.json...")
        print("-" * 60)
        clean_mapping_file(config_mapping, config_registry)
    
    print("\n✅ 所有映射文件已清理完成")
