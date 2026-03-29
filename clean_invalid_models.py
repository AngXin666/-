#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理YOLO模型注册表中的无效模型
- 删除所有auto_registered=true且性能为0的模型
- 保留有效的训练模型
"""

import json
import os
from pathlib import Path

def clean_invalid_models():
    """清理无效的自动注册模型"""
    
    registry_path = Path("models/yolo_model_registry.json")
    
    if not registry_path.exists():
        print("❌ 找不到模型注册表文件")
        return
    
    # 读取注册表
    with open(registry_path, 'r', encoding='utf-8') as f:
        registry = json.load(f)
    
    original_count = len(registry['models'])
    print(f"📊 原始模型数量: {original_count}")
    
    # 统计要删除的模型
    invalid_models = []
    valid_models = {}
    
    for model_key, model_data in registry['models'].items():
        # 检查是否为无效模型
        is_auto_registered = model_data.get('auto_registered', False)
        performance = model_data.get('performance', {})
        mAP50 = performance.get('mAP50', 0)
        
        if is_auto_registered and mAP50 == 0.0:
            invalid_models.append(model_key)
            print(f"🗑️  删除无效模型: {model_key}")
        else:
            valid_models[model_key] = model_data
            print(f"✅ 保留有效模型: {model_key} (mAP50: {mAP50})")
    
    # 更新注册表
    registry['models'] = valid_models
    
    # 保存清理后的注册表
    with open(registry_path, 'w', encoding='utf-8') as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
    
    print(f"\n📈 清理结果:")
    print(f"   删除无效模型: {len(invalid_models)} 个")
    print(f"   保留有效模型: {len(valid_models)} 个")
    print(f"   清理完成！")
    
    return len(invalid_models), len(valid_models)

if __name__ == "__main__":
    clean_invalid_models()