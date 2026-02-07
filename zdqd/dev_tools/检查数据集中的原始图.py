"""
检查 yolo_dataset_* 中是否包含原始标注图

对比 yolo_dataset_* 和 原始标注图/ 中的图片
找出只存在于 yolo_dataset_* 中的原始图
"""

from pathlib import Path
from collections import defaultdict

def get_image_files(directory):
    """获取目录中的所有图片文件名（不含路径）"""
    images = set()
    if directory.exists():
        for img in directory.rglob('*.png'):
            images.add(img.name)
        for img in directory.rglob('*.jpg'):
            images.add(img.name)
    return images

def check_datasets():
    """检查数据集中的原始图"""
    print("=" * 80)
    print("检查 yolo_dataset_* 中是否包含原始标注图")
    print("=" * 80)
    
    # 获取原始标注图文件夹中的所有图片
    original_dir = Path("原始标注图")
    original_images = get_image_files(original_dir)
    
    print(f"\n原始标注图/ 文件夹中的图片: {len(original_images)} 张")
    
    # 查找所有 yolo_dataset_* 文件夹
    datasets = list(Path(".").glob("yolo_dataset_*"))
    
    if not datasets:
        print("\n没有找到 yolo_dataset_* 文件夹")
        return
    
    print(f"找到 {len(datasets)} 个训练数据集文件夹")
    
    # 检查每个数据集
    all_unique_images = defaultdict(list)  # 只在数据集中存在的图片
    
    for dataset in sorted(datasets):
        print(f"\n" + "-" * 80)
        print(f"检查: {dataset.name}")
        print("-" * 80)
        
        # 获取数据集中的所有图片
        dataset_images = get_image_files(dataset)
        
        print(f"数据集中的图片: {len(dataset_images)} 张")
        
        # 找出只在数据集中存在的图片（可能是原始图）
        unique_images = dataset_images - original_images
        
        # 过滤掉明显的增强图片（文件名包含 train_, val_ 等）
        potential_original = []
        for img in unique_images:
            # 如果文件名不是 train_数字 或 val_数字 格式，可能是原始图
            if not (img.startswith('train_') or img.startswith('val_')):
                potential_original.append(img)
        
        if potential_original:
            print(f"⚠️  可能包含原始图: {len(potential_original)} 张")
            all_unique_images[dataset.name] = potential_original
            
            # 显示前 10 个文件名
            print("\n示例文件名:")
            for img in sorted(potential_original)[:10]:
                print(f"  - {img}")
            if len(potential_original) > 10:
                print(f"  ... 还有 {len(potential_original) - 10} 张")
        else:
            print("✓ 没有发现原始图（所有图片都是 train_/val_ 格式或已在原始标注图/中）")
    
    # 总结
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    
    if all_unique_images:
        print("\n⚠️  以下数据集可能包含原始标注图：")
        total_unique = 0
        for dataset_name, images in all_unique_images.items():
            print(f"\n{dataset_name}: {len(images)} 张")
            total_unique += len(images)
        
        print(f"\n总计: {total_unique} 张可能的原始图")
        print("\n建议:")
        print("  1. 检查这些图片是否是原始标注图")
        print("  2. 如果是，复制到 原始标注图/ 文件夹")
        print("  3. 然后再删除 yolo_dataset_* 文件夹")
    else:
        print("\n✓ 所有数据集都不包含原始标注图")
        print("✓ 可以安全删除 yolo_dataset_* 文件夹")

if __name__ == '__main__':
    check_datasets()
