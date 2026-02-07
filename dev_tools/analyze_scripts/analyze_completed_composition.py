"""
分析 training_data_completed 文件夹中图片的构成
区分原始图和增强图
"""
import os
from pathlib import Path

def is_augmented_image(filename):
    """判断是否为增强图片"""
    # 增强图片的特征标识
    augment_markers = [
        '_bright_', '_color_', '_contrast_', '_saturation_',
        '_combo_bc_', '_combo_cs_', '_combo_bcol_', '_combo_triple_',
        '副本'
    ]
    
    for marker in augment_markers:
        if marker in filename:
            return True
    return False

def analyze_folder_composition(folder_path):
    """分析文件夹中图片的构成"""
    folder = Path(folder_path)
    
    if not folder.exists():
        return None
    
    # 检查是否有 images 子文件夹（YOLO格式）
    images_train = folder / 'images' / 'train'
    images_val = folder / 'images' / 'val'
    
    all_images = []
    
    if images_train.exists():
        all_images.extend(list(images_train.glob('*.png')))
        all_images.extend(list(images_train.glob('*.jpg')))
    
    if images_val.exists():
        all_images.extend(list(images_val.glob('*.png')))
        all_images.extend(list(images_val.glob('*.jpg')))
    
    # 如果没有 images 子文件夹，直接在文件夹中查找
    if not all_images:
        all_images.extend(list(folder.glob('*.png')))
        all_images.extend(list(folder.glob('*.jpg')))
    
    if not all_images:
        return None
    
    original_count = 0
    augmented_count = 0
    
    for img in all_images:
        if is_augmented_image(img.name):
            augmented_count += 1
        else:
            original_count += 1
    
    return {
        'folder_name': folder.name,
        'total': len(all_images),
        'original': original_count,
        'augmented': augmented_count,
        'augmented_ratio': augmented_count / len(all_images) * 100 if len(all_images) > 0 else 0
    }

def main():
    """主函数"""
    completed_dir = Path('training_data_completed')
    
    if not completed_dir.exists():
        print("training_data_completed 文件夹不存在！")
        return
    
    print("=" * 90)
    print("training_data_completed 图片构成分析")
    print("=" * 90)
    
    results = []
    
    # 遍历所有子文件夹
    for folder in sorted(completed_dir.iterdir()):
        if folder.is_dir():
            result = analyze_folder_composition(folder)
            if result:
                results.append(result)
    
    # 按总图片数排序
    results.sort(key=lambda x: x['total'], reverse=True)
    
    # 显示结果
    print(f"\n{'文件夹名称':<45} {'总数':>6} {'原始':>6} {'增强':>6} {'增强比例':>10}")
    print("-" * 90)
    
    total_images = 0
    total_original = 0
    total_augmented = 0
    
    for result in results:
        print(f"{result['folder_name']:<45} "
              f"{result['total']:>6} "
              f"{result['original']:>6} "
              f"{result['augmented']:>6} "
              f"{result['augmented_ratio']:>9.1f}%")
        
        total_images += result['total']
        total_original += result['original']
        total_augmented += result['augmented']
    
    print("-" * 90)
    total_ratio = total_augmented / total_images * 100 if total_images > 0 else 0
    print(f"{'总计':<45} "
          f"{total_images:>6} "
          f"{total_original:>6} "
          f"{total_augmented:>6} "
          f"{total_ratio:>9.1f}%")
    
    print("\n" + "=" * 90)
    print("分析完成！")
    print("=" * 90)
    print(f"总文件夹数: {len(results)} 个")
    print(f"总图片数: {total_images} 张")
    print(f"原始图片: {total_original} 张 ({total_original/total_images*100:.1f}%)")
    print(f"增强图片: {total_augmented} 张 ({total_augmented/total_images*100:.1f}%)")
    
    # 统计纯原始和纯增强的文件夹
    pure_original = sum(1 for r in results if r['augmented'] == 0)
    pure_augmented = sum(1 for r in results if r['original'] == 0)
    mixed = sum(1 for r in results if r['original'] > 0 and r['augmented'] > 0)
    
    print(f"\n文件夹类型:")
    print(f"  纯原始图: {pure_original} 个")
    print(f"  纯增强图: {pure_augmented} 个")
    print(f"  混合图片: {mixed} 个")

if __name__ == '__main__':
    main()
