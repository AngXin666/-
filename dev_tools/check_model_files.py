"""
检查YOLO模型文件是否存在
"""
import json
import os

def check_model_files():
    """检查所有模型文件是否存在"""
    
    registry_path = 'models/yolo_model_registry.json'
    
    with open(registry_path, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)
    
    models = data.get('models', {})
    
    print(f"=" * 80)
    print(f"YOLO模型文件存在性检查")
    print(f"=" * 80)
    print(f"\n总模型数: {len(models)}")
    
    # 统计
    exists = []
    not_exists = []
    
    for key, model in models.items():
        path = model.get('model_path', '')
        full_path = os.path.join('models', path)
        
        if os.path.exists(full_path):
            exists.append((key, path))
        else:
            not_exists.append((key, path))
    
    # 输出结果
    print(f"\n" + "=" * 80)
    print("检查结果:")
    print("=" * 80)
    
    print(f"\n【文件存在】 ({len(exists)}个)")
    if len(exists) <= 20:
        for key, path in exists:
            print(f"  ✓ {key}")
    else:
        for key, path in exists[:10]:
            print(f"  ✓ {key}")
        print(f"  ... 还有 {len(exists) - 10} 个")
    
    if not_exists:
        print(f"\n【文件不存在】 ({len(not_exists)}个)")
        for key, path in not_exists:
            print(f"  ✗ {key}")
            print(f"    路径: {path}")
            print(f"    完整路径: models/{path}")
    
    # 总结
    print(f"\n" + "=" * 80)
    print("总结:")
    print("=" * 80)
    print(f"✓ 文件存在: {len(exists)}个")
    print(f"✗ 文件不存在: {len(not_exists)}个")
    
    if len(not_exists) == 0:
        print(f"\n🎉 所有模型文件都存在！")
    else:
        print(f"\n⚠️ 有 {len(not_exists)} 个模型文件不存在")

if __name__ == "__main__":
    check_model_files()
