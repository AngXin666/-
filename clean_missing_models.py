#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
清理注册表中缺失的模型记录
"""

import json
from pathlib import Path

def clean_registry():
    """清理注册表中缺失的模型"""
    registry_path = 'config/yolo_model_registry.json'
    
    print("=" * 70)
    print("清理注册表中缺失的模型")
    print("=" * 70)
    
    print("\n读取注册表...")
    with open(registry_path, 'r', encoding='utf-8') as f:
        registry = json.load(f)
    
    models = registry.get('models', {})
    print(f"注册表中有 {len(models)} 个模型")
    
    # 检查每个模型文件是否存在
    missing_models = []
    for model_name, model_info in models.items():
        model_path = model_info.get('model_path', '')
        if not model_path:
            continue
        
        full_path = Path('models') / model_path
        if not full_path.exists():
            missing_models.append((model_name, model_info))
            print(f"❌ {model_name}: {model_path} (缺失)")
    
    if not missing_models:
        print("\n✓ 所有模型文件都存在，无需清理")
        return
    
    print(f"\n发现 {len(missing_models)} 个缺失的模型:")
    for name, info in missing_models:
        print(f"\n  模型名: {name}")
        print(f"  描述: {info.get('name', 'N/A')}")
        print(f"  路径: {info.get('model_path', 'N/A')}")
        print(f"  用途: {info.get('notes', 'N/A')}")
    
    # 询问是否删除
    print("\n" + "=" * 70)
    print("⚠ 重要说明：")
    print("  - profile_detailed 是一个优化的综合模型")
    print("  - 它可以一次检测8个元素，性能比分散模型快23倍")
    print("  - 但目前代码中使用的是分散模型（profile_logged + balance等）")
    print("  - 移除此记录不会影响当前功能")
    print("  - 以后可以重新训练并添加回来")
    print("=" * 70)
    
    print("\n是否从注册表中删除这些模型记录？")
    confirm = input("输入 yes 确认删除: ")
    
    if confirm.lower() != 'yes':
        print("\n已取消")
        return
    
    # 删除缺失的模型记录
    for model_name, _ in missing_models:
        del models[model_name]
        print(f"✓ 已删除: {model_name}")
    
    # 保存注册表
    with open(registry_path, 'w', encoding='utf-8') as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 注册表已更新")
    print(f"  剩余模型数量: {len(models)}")
    print("\n建议：以后可以使用'新个人页已登陆'数据重新训练profile_detailed模型")

if __name__ == '__main__':
    clean_registry()
