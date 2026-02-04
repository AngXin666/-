"""
测试页面-YOLO映射配置加载
"""
import json
import os

def test_mapping_load():
    """测试映射配置加载"""
    
    # 加载映射配置
    mapping_path = 'models/page_yolo_mapping.json'
    
    if not os.path.exists(mapping_path):
        print(f"✗ 映射文件不存在: {mapping_path}")
        return
    
    print(f"✓ 找到映射文件: {mapping_path}")
    
    with open(mapping_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        mapping = data.get('mapping', {})
    
    print(f"\n✓ 加载了 {len(mapping)} 个页面类型的映射")
    
    # 检查"个人页_已登录"的映射
    target_page = "个人页_已登录"
    
    if target_page in mapping:
        print(f"\n✓ 找到 '{target_page}' 的映射:")
        page_mapping = mapping[target_page]
        print(f"  - page_state: {page_mapping.get('page_state')}")
        print(f"  - yolo_models数量: {len(page_mapping.get('yolo_models', []))}")
        
        for model in page_mapping.get('yolo_models', []):
            print(f"    • {model.get('model_key')} (优先级: {model.get('priority')})")
            print(f"      用途: {model.get('purpose')}")
    else:
        print(f"\n✗ 未找到 '{target_page}' 的映射")
        print(f"\n可用的页面类型:")
        for i, key in enumerate(list(mapping.keys())[:20], 1):
            print(f"  {i}. {key}")
    
    # 加载YOLO注册表
    registry_path = 'yolo_model_registry.json'
    
    if not os.path.exists(registry_path):
        print(f"\n✗ 注册表文件不存在: {registry_path}")
        return
    
    print(f"\n✓ 找到注册表文件: {registry_path}")
    
    with open(registry_path, 'r', encoding='utf-8') as f:
        registry_data = json.load(f)
        models = registry_data.get('models', {})
    
    print(f"✓ 注册表中有 {len(models)} 个模型")
    
    # 检查profile_logged模型
    if 'profile_logged' in models:
        print(f"\n✓ 找到 'profile_logged' 模型:")
        model_info = models['profile_logged']
        print(f"  - 名称: {model_info.get('name')}")
        print(f"  - 页面类型: {model_info.get('page_type')}")
        print(f"  - 模型路径: {model_info.get('model_path')}")
        print(f"  - 类别: {model_info.get('classes')}")
        
        # 检查模型文件是否存在
        model_path = model_info.get('model_path')
        if model_path:
            full_path = os.path.join('models', model_path)
            if os.path.exists(full_path):
                print(f"  ✓ 模型文件存在: {full_path}")
            else:
                print(f"  ✗ 模型文件不存在: {full_path}")
                if os.path.exists(model_path):
                    print(f"  ✓ 但原路径存在: {model_path}")
    else:
        print(f"\n✗ 未找到 'profile_logged' 模型")

if __name__ == '__main__':
    print("=" * 60)
    print("测试页面-YOLO映射配置加载")
    print("=" * 60)
    test_mapping_load()
    print("\n" + "=" * 60)
