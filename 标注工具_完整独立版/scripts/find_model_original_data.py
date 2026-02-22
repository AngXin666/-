"""
根据模型注册表，查找已训练模型的原始标注图位置
"""
import json
from pathlib import Path

def find_original_data_folder(page_type, original_count):
    """查找原始数据文件夹"""
    # 可能的位置
    possible_locations = [
        Path('training_data') / page_type,
        Path('training_data_completed') / page_type,
    ]
    
    # 检查 training_data_completed 中的相关文件夹
    completed_dir = Path('training_data_completed')
    if completed_dir.exists():
        for folder in completed_dir.iterdir():
            if folder.is_dir() and page_type.lower() in folder.name.lower():
                possible_locations.append(folder)
    
    found_locations = []
    for location in possible_locations:
        if location.exists():
            # 统计图片数量
            images = list(location.rglob('*.png')) + list(location.rglob('*.jpg'))
            if len(images) > 0:
                found_locations.append({
                    'path': str(location),
                    'image_count': len(images)
                })
    
    return found_locations

def main():
    """主函数"""
    # 读取模型注册表
    registry_path = Path('yolo_model_registry.json')
    
    if not registry_path.exists():
        print("模型注册表不存在！")
        return
    
    with open(registry_path, 'r', encoding='utf-8') as f:
        registry = json.load(f)
    
    print("=" * 100)
    print("已训练模型的原始标注图位置")
    print("=" * 100)
    
    models = registry['models']
    
    print(f"\n{'模型名称':<25} {'页面类型':<30} {'原始图':<8} {'找到的位置':<40}")
    print("-" * 100)
    
    found_count = 0
    not_found_count = 0
    
    for model_key, model_info in models.items():
        page_type = model_info['page_type']
        original_count = model_info['dataset_size']['original']
        
        locations = find_original_data_folder(page_type, original_count)
        
        if locations:
            found_count += 1
            for i, loc in enumerate(locations):
                if i == 0:
                    print(f"{model_info['name']:<25} "
                          f"{page_type:<30} "
                          f"{original_count:<8} "
                          f"{loc['path']:<40} ({loc['image_count']}张)")
                else:
                    print(f"{'':25} {'':30} {'':8} {loc['path']:<40} ({loc['image_count']}张)")
        else:
            not_found_count += 1
            print(f"{model_info['name']:<25} "
                  f"{page_type:<30} "
                  f"{original_count:<8} "
                  f"❌ 未找到")
    
    print("-" * 100)
    print(f"\n总结:")
    print(f"  已训练模型: {len(models)} 个")
    print(f"  找到原始数据: {found_count} 个")
    print(f"  未找到原始数据: {not_found_count} 个")
    
    # 列出未找到的模型
    if not_found_count > 0:
        print(f"\n未找到原始数据的模型:")
        for model_key, model_info in models.items():
            page_type = model_info['page_type']
            original_count = model_info['dataset_size']['original']
            locations = find_original_data_folder(page_type, original_count)
            
            if not locations:
                print(f"  - {model_info['name']} ({page_type}, 原始图: {original_count}张)")

if __name__ == '__main__':
    main()
