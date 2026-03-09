"""
分析模型加载差异
"""
import json

def analyze_model_loading():
    """分析为什么注册表有78个模型但只加载46个"""
    
    registry_path = 'models/yolo_model_registry.json'
    
    with open(registry_path, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)
    
    models = data.get('models', {})
    
    print(f"=" * 80)
    print(f"模型加载分析")
    print(f"=" * 80)
    print(f"\n注册表中的模型总数: {len(models)}")
    
    # 分类统计
    categories = {
        '自动注册的模型': [],
        '手动注册的模型': [],
        '重复的模型': []
    }
    
    # 用于检测重复的字典
    seen_paths = {}
    
    for key, model in models.items():
        path = model.get('model_path', '')
        auto_registered = model.get('auto_registered', False)
        
        if auto_registered:
            categories['自动注册的模型'].append((key, path))
        else:
            categories['手动注册的模型'].append((key, path))
        
        # 检查路径重复
        if path in seen_paths:
            categories['重复的模型'].append((key, path, seen_paths[path]))
        else:
            seen_paths[path] = key
    
    # 输出结果
    print(f"\n" + "=" * 80)
    print("分类统计:")
    print("=" * 80)
    
    for category, items in categories.items():
        print(f"\n【{category}】 ({len(items)}个)")
        if category == '重复的模型':
            for key, path, original_key in items[:10]:
                print(f"  - {key} (重复于 {original_key})")
                print(f"    路径: {path}")
            if len(items) > 10:
                print(f"  ... 还有 {len(items) - 10} 个")
        else:
            for key, path in items[:10]:
                print(f"  - {key}")
            if len(items) > 10:
                print(f"  ... 还有 {len(items) - 10} 个")
    
    # 分析
    print(f"\n" + "=" * 80)
    print("分析:")
    print("=" * 80)
    
    auto_count = len(categories['自动注册的模型'])
    manual_count = len(categories['手动注册的模型'])
    duplicate_count = len(categories['重复的模型'])
    
    print(f"手动注册的模型: {manual_count}个")
    print(f"自动注册的模型: {auto_count}个")
    print(f"重复的模型: {duplicate_count}个")
    print(f"\n实际有效模型数: {len(seen_paths)}个")
    
    # 推测
    print(f"\n" + "=" * 80)
    print("推测:")
    print("=" * 80)
    print(f"程序可能只加载手动注册的模型（非auto_registered=true的模型）")
    print(f"预期加载数量: {manual_count}个")
    print(f"实际加载数量: 46个")
    
    if manual_count != 46:
        print(f"\n⚠️ 数量不匹配！可能还有其他过滤条件")
        print(f"差异: {manual_count - 46}个")

if __name__ == "__main__":
    analyze_model_loading()
