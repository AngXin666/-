"""检查增强数据集"""
from pathlib import Path
import random

def check_augmented_dataset():
    """检查增强数据集的详细信息"""
    data_path = Path('page_classifier_dataset_augmented')
    
    if not data_path.exists():
        print(f"❌ 增强数据集目录不存在: {data_path}")
        return
    
    print("=" * 60)
    print("增强数据集检查")
    print("=" * 60)
    
    total = 0
    stats = {}
    
    # 统计每个类别
    for class_dir in sorted(data_path.iterdir()):
        if not class_dir.is_dir():
            continue
        
        all_files = list(class_dir.glob('*.png'))
        original_files = [f for f in all_files if not f.name.startswith('aug_')]
        augmented_files = [f for f in all_files if f.name.startswith('aug_')]
        
        stats[class_dir.name] = {
            'total': len(all_files),
            'original': len(original_files),
            'augmented': len(augmented_files)
        }
        total += len(all_files)
    
    print(f"\n总图片数: {total} 张")
    print(f"类别数: {len(stats)} 个")
    
    print("\n各类别统计:")
    for class_name in sorted(stats.keys()):
        s = stats[class_name]
        print(f"  {class_name}:")
        print(f"    原始: {s['original']} 张")
        print(f"    增强: {s['augmented']} 张")
        print(f"    总计: {s['total']} 张")
        if s['original'] > 0:
            ratio = s['augmented'] / s['original']
            print(f"    增强倍数: {ratio:.1f}x")
    
    # 随机抽样检查
    print("\n" + "=" * 60)
    print("随机抽样检查")
    print("=" * 60)
    
    # 随机选择3个类别
    sample_classes = random.sample(list(stats.keys()), min(3, len(stats)))
    
    for class_name in sample_classes:
        class_dir = data_path / class_name
        all_files = list(class_dir.glob('*.png'))
        
        print(f"\n类别: {class_name}")
        print(f"总文件数: {len(all_files)}")
        
        # 显示前5个原始文件
        original_files = [f for f in all_files if not f.name.startswith('aug_')][:5]
        if original_files:
            print("\n  原始文件示例:")
            for f in original_files:
                print(f"    - {f.name}")
        
        # 显示前5个增强文件
        augmented_files = [f for f in all_files if f.name.startswith('aug_')][:5]
        if augmented_files:
            print("\n  增强文件示例:")
            for f in augmented_files:
                print(f"    - {f.name}")
    
    # 检查是否有问题
    print("\n" + "=" * 60)
    print("数据质量检查")
    print("=" * 60)
    
    issues = []
    
    for class_name, s in stats.items():
        if s['total'] == 0:
            issues.append(f"❌ {class_name}: 没有图片")
        elif s['original'] == 0:
            issues.append(f"⚠️ {class_name}: 没有原始图片，只有增强图片")
        elif s['augmented'] == 0:
            issues.append(f"⚠️ {class_name}: 没有增强图片")
        elif s['augmented'] < s['original'] * 3:
            issues.append(f"⚠️ {class_name}: 增强倍数过低 ({s['augmented'] / s['original']:.1f}x)")
    
    if issues:
        print("\n发现问题:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✓ 数据集检查通过，没有发现问题")
    
    print("\n" + "=" * 60)
    print("检查完成")
    print("=" * 60)

if __name__ == '__main__':
    check_augmented_dataset()
