"""
清理无效的自动注册模型
Clean Invalid Auto-registered Models

识别并删除：
1. 重复的自动注册模型（与手动注册模型指向同一文件）
2. 训练过程中的中间版本模型（train, train2, exp, exp2等）
3. 性能指标全为0且没有实际用途的模型

安全措施：
- 先备份原文件（.backup后缀）
- 只删除确认无效的模型
- 测试通过后手动删除备份
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set


def load_registry(registry_path: Path) -> Dict:
    """加载注册表"""
    with open(registry_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_registry(registry_path: Path, registry: Dict):
    """保存注册表"""
    with open(registry_path, 'w', encoding='utf-8') as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)


def identify_invalid_models(registry: Dict) -> Dict[str, List[Dict]]:
    """识别无效模型
    
    Returns:
        {
            'duplicate': [模型信息列表],  # 重复的自动注册模型
            'intermediate': [模型信息列表],  # 中间版本模型
            'definitely_invalid': [模型信息列表]  # 确定无效的模型
        }
    """
    models = registry.get('models', {})
    
    # 收集手动注册模型的路径
    manual_model_paths = {}  # {path: [keys]}
    for key, info in models.items():
        if not info.get('auto_registered', False):
            path = info.get('model_path', '')
            if path not in manual_model_paths:
                manual_model_paths[path] = []
            manual_model_paths[path].append(key)
    
    invalid_models = {
        'duplicate': [],
        'intermediate': [],
        'definitely_invalid': []
    }
    
    # 识别无效模型
    for key, info in models.items():
        # 只检查自动注册的模型
        if not info.get('auto_registered', False):
            continue
        
        model_path = info.get('model_path', '')
        page_type = info.get('page_type', '')
        
        model_info = {
            'key': key,
            'name': info.get('name', 'N/A'),
            'path': model_path,
            'page_type': page_type,
            'reason': ''
        }
        
        # 1. 重复模型：自动注册模型的路径与手动注册模型相同
        if model_path in manual_model_paths:
            model_info['reason'] = f"与手动注册模型重复: {', '.join(manual_model_paths[model_path])}"
            invalid_models['duplicate'].append(model_info)
            continue
        
        # 2. 中间版本模型：train, train2, exp, exp2 等（这些肯定是无效的）
        intermediate_patterns = ['train', 'train2', 'train3', 'train4', 'exp', 'exp2', 'exp3']
        if page_type in intermediate_patterns or key in intermediate_patterns:
            model_info['reason'] = f"训练中间版本（{page_type}）"
            invalid_models['intermediate'].append(model_info)
            continue
        
        # 3. 确定无效：性能为0 + 有手动注册版本 + 时间戳后缀
        performance = info.get('performance', {})
        has_timestamp = '_20260131_' in key  # 带时间戳的自动注册模型
        
        if (performance.get('mAP50', 0) == 0 and 
            performance.get('precision', 0) == 0 and 
            performance.get('recall', 0) == 0 and
            has_timestamp):
            # 检查是否有对应的手动注册版本
            has_manual_version = False
            for manual_key, manual_info in models.items():
                if (not manual_info.get('auto_registered', False) and 
                    manual_info.get('page_type', '') == page_type):
                    has_manual_version = True
                    model_info['reason'] = f"性能为0且有手动版本: {manual_key}"
                    break
            
            if has_manual_version:
                invalid_models['definitely_invalid'].append(model_info)
    
    return invalid_models


def backup_registry(registry_path: Path) -> Path:
    """备份注册表文件
    
    Returns:
        备份文件路径
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = registry_path.with_suffix(f'.backup_{timestamp}.json')
    shutil.copy2(registry_path, backup_path)
    return backup_path


def clean_models(registry_path: Path, dry_run: bool = True) -> Dict:
    """清理无效模型
    
    Args:
        registry_path: 注册表路径
        dry_run: 是否只是预览，不实际删除
        
    Returns:
        清理结果统计
    """
    print(f"正在分析注册表: {registry_path}")
    print("=" * 80)
    
    # 加载注册表
    registry = load_registry(registry_path)
    models = registry.get('models', {})
    
    print(f"总模型数: {len(models)}")
    
    # 统计手动注册和自动注册模型
    manual_count = sum(1 for info in models.values() if not info.get('auto_registered', False))
    auto_count = sum(1 for info in models.values() if info.get('auto_registered', False))
    
    print(f"  - 手动注册: {manual_count}")
    print(f"  - 自动注册: {auto_count}")
    print()
    
    # 识别无效模型
    invalid_models = identify_invalid_models(registry)
    
    # 显示无效模型
    total_invalid = sum(len(v) for v in invalid_models.values())
    
    print(f"发现 {total_invalid} 个无效模型:")
    print()
    
    # 1. 重复模型（确定无效）
    if invalid_models['duplicate']:
        print(f"1. 重复模型 ({len(invalid_models['duplicate'])}个) - 确定无效")
        print("   这些自动注册模型与手动注册模型指向同一文件")
        for model_info in invalid_models['duplicate']:
            print(f"   - {model_info['key']}: {model_info['name']}")
            print(f"     路径: {model_info['path']}")
            print(f"     原因: {model_info['reason']}")
        print()
    
    # 2. 中间版本模型（确定无效）
    if invalid_models['intermediate']:
        print(f"2. 中间版本模型 ({len(invalid_models['intermediate'])}个) - 确定无效")
        print("   训练过程中的中间版本（train, exp等）")
        for model_info in invalid_models['intermediate']:
            print(f"   - {model_info['key']}: {model_info['name']}")
            print(f"     路径: {model_info['path']}")
            print(f"     原因: {model_info['reason']}")
        print()
    
    # 3. 确定无效的模型
    if invalid_models['definitely_invalid']:
        print(f"3. 确定无效的模型 ({len(invalid_models['definitely_invalid'])}个)")
        print("   性能为0 + 有手动注册版本 + 带时间戳")
        for model_info in invalid_models['definitely_invalid']:
            print(f"   - {model_info['key']}: {model_info['name']}")
            print(f"     路径: {model_info['path']}")
            print(f"     原因: {model_info['reason']}")
        print()
    
    # 删除无效模型
    if not dry_run:
        print("=" * 80)
        
        # 先备份
        backup_path = backup_registry(registry_path)
        print(f"✓ 已备份到: {backup_path}")
        print()
        
        print("开始删除无效模型...")
        
        all_invalid_keys = []
        for category_models in invalid_models.values():
            all_invalid_keys.extend([m['key'] for m in category_models])
        
        for key in all_invalid_keys:
            del models[key]
            print(f"  ✓ 已删除: {key}")
        
        # 保存注册表
        save_registry(registry_path, registry)
        print()
        print(f"✅ 已删除 {len(all_invalid_keys)} 个无效模型")
        print(f"剩余模型数: {len(models)}")
        print()
        print(f"⚠️  备份文件: {backup_path}")
        print("   测试通过后请手动删除备份文件")
    else:
        print("=" * 80)
        print("⚠️  这是预览模式，未实际删除任何模型")
        print("如需实际删除，请使用参数: dry_run=False")
    
    return {
        'total_before': len(models) + total_invalid if not dry_run else len(models),
        'total_after': len(models),
        'deleted': total_invalid,
        'invalid_models': invalid_models
    }


def main():
    """主函数"""
    # 检查两个注册表文件
    config_registry = Path("config/yolo_model_registry.json")
    models_registry = Path("models/yolo_model_registry.json")
    
    print("YOLO模型注册表清理工具")
    print("=" * 80)
    print()
    
    # 先预览
    print("【预览模式】")
    print()
    
    if config_registry.exists():
        print("1. config/yolo_model_registry.json")
        print("-" * 80)
        clean_models(config_registry, dry_run=True)
        print()
    
    if models_registry.exists():
        print("2. models/yolo_model_registry.json")
        print("-" * 80)
        clean_models(models_registry, dry_run=True)
        print()
    
    # 询问是否执行
    print("=" * 80)
    response = input("是否执行删除？(yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        print()
        print("【执行删除】")
        print()
        
        if config_registry.exists():
            print("1. config/yolo_model_registry.json")
            print("-" * 80)
            clean_models(config_registry, dry_run=False)
            print()
        
        if models_registry.exists():
            print("2. models/yolo_model_registry.json")
            print("-" * 80)
            clean_models(models_registry, dry_run=False)
            print()
        
        print("✅ 清理完成！")
    else:
        print("❌ 已取消删除")


if __name__ == "__main__":
    main()
