"""
找出所有包含原始标注图的文件夹
统计每个文件夹中原始图片和副本的数量
"""
import os
from pathlib import Path

def is_copy_file(filename):
    """判断是否为副本文件"""
    return '副本' in filename

def has_label_file(image_path):
    """检查图片是否有对应的标签文件"""
    label_path = image_path.with_suffix('.txt')
    return label_path.exists()

def analyze_folder(folder_path):
    """分析文件夹中的标注图"""
    folder = Path(folder_path)
    
    if not folder.exists() or not folder.is_dir():
        return None
    
    # 统计图片
    all_images = list(folder.glob('*.png')) + list(folder.glob('*.jpg'))
    
    if not all_images:
        return None
    
    original_images = []
    copy_images = []
    original_with_labels = []
    copy_with_labels = []
    
    for img in all_images:
        has_label = has_label_file(img)
        
        if is_copy_file(img.name):
            copy_images.append(img)
            if has_label:
                copy_with_labels.append(img)
        else:
            original_images.append(img)
            if has_label:
                original_with_labels.append(img)
    
    return {
        'folder_name': folder.name,
        'total_images': len(all_images),
        'original_count': len(original_images),
        'copy_count': len(copy_images),
        'original_labeled': len(original_with_labels),
        'copy_labeled': len(copy_with_labels),
        'has_annotations': len(original_with_labels) > 0 or len(copy_with_labels) > 0
    }

def main():
    """主函数"""
    training_data_dir = Path('training_data')
    
    if not training_data_dir.exists():
        print("training_data 文件夹不存在！")
        return
    
    print("=" * 80)
    print("原始标注图文件夹统计")
    print("=" * 80)
    
    folders_with_annotations = []
    folders_without_annotations = []
    
    # 遍历所有子文件夹
    for folder in sorted(training_data_dir.iterdir()):
        if folder.is_dir():
            result = analyze_folder(folder)
            
            if result and result['total_images'] > 0:
                if result['has_annotations']:
                    folders_with_annotations.append(result)
                else:
                    folders_without_annotations.append(result)
    
    # 显示有标注的文件夹
    print("\n【有标注的文件夹】")
    print("-" * 80)
    print(f"{'文件夹名称':<30} {'原始图':<8} {'副本':<8} {'原始已标注':<12} {'副本已标注':<12}")
    print("-" * 80)
    
    total_original = 0
    total_copy = 0
    total_original_labeled = 0
    total_copy_labeled = 0
    
    for result in folders_with_annotations:
        print(f"{result['folder_name']:<30} "
              f"{result['original_count']:<8} "
              f"{result['copy_count']:<8} "
              f"{result['original_labeled']:<12} "
              f"{result['copy_labeled']:<12}")
        
        total_original += result['original_count']
        total_copy += result['copy_count']
        total_original_labeled += result['original_labeled']
        total_copy_labeled += result['copy_labeled']
    
    print("-" * 80)
    print(f"{'总计':<30} "
          f"{total_original:<8} "
          f"{total_copy:<8} "
          f"{total_original_labeled:<12} "
          f"{total_copy_labeled:<12}")
    
    # 显示没有标注的文件夹
    if folders_without_annotations:
        print("\n【没有标注的文件夹】")
        print("-" * 80)
        for result in folders_without_annotations:
            print(f"{result['folder_name']:<30} "
                  f"原始图: {result['original_count']}, "
                  f"副本: {result['copy_count']}")
    
    print("\n" + "=" * 80)
    print("统计完成！")
    print("=" * 80)
    print(f"有标注的文件夹: {len(folders_with_annotations)} 个")
    print(f"没有标注的文件夹: {len(folders_without_annotations)} 个")
    print(f"原始已标注图片: {total_original_labeled} 张")
    print(f"副本已标注图片: {total_copy_labeled} 张")
    print(f"总已标注图片: {total_original_labeled + total_copy_labeled} 张")

if __name__ == '__main__':
    main()
