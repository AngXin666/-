"""
从恢复的数据中筛选出原始图片

删除增强图片，只保留原始标注图
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict

def is_augmented_image(filename):
    """
    判断是否为增强图片
    
    增强图片的特征：
    - 文件名包含 _copy
    - 文件名包含 _augmented
    - 文件名包含 _aug
    - 文件名包含 _flip
    - 文件名包含 _rotate
    - 文件名包含 _bright
    - 文件名包含 _contrast
    """
    augmented_markers = [
        '_copy', '_augmented', '_aug', '_flip', 
        '_rotate', '_bright', '_contrast', '_noise',
        '_blur', '_crop', '_scale', '_shift'
    ]
    
    filename_lower = filename.lower()
    return any(marker in filename_lower for marker in augmented_markers)

def filter_original_images(folder_path):
    """
    筛选原始图片，删除增强图片
    
    Args:
        folder_path: 文件夹路径
    
    Returns:
        dict: 统计信息
    """
    folder = Path(folder_path)
    images_dir = folder / "images"
    labels_dir = folder / "labels"
    
    stats = {
        'original_images': 0,
        'augmented_images_deleted': 0,
        'orphan_labels_deleted': 0
    }
    
    if not images_dir.exists():
        return stats
    
    # 获取所有图片文件
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(images_dir.glob(ext))
    
    # 删除增强图片
    for img_file in image_files:
        if is_augmented_image(img_file.stem):
            # 删除图片
            img_file.unlink()
            stats['augmented_images_deleted'] += 1
            
            # 删除对应的标签文件
            label_file = labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                label_file.unlink()
        else:
            stats['original_images'] += 1
    
    # 删除孤立的标签文件（没有对应图片的标签）
    if labels_dir.exists():
        for label_file in labels_dir.glob("*.txt"):
            # 检查是否有对应的图片
            img_exists = False
            for ext in ['.png', '.jpg', '.jpeg']:
                if (images_dir / f"{label_file.stem}{ext}").exists():
                    img_exists = True
                    break
            
            if not img_exists:
                label_file.unlink()
                stats['orphan_labels_deleted'] += 1
    
    return stats

def main():
    """主函数"""
    print("=" * 80)
    print("从恢复的数据中筛选原始图片")
    print("=" * 80)
    
    # 需要处理的文件夹
    folders_to_filter = [
        '个人页_已登录_余额积分',
        '个人页_已登录_头像首页',
        '启动页服务弹窗',
        '登录异常'
    ]
    
    base_dir = Path("training_data_completed")
    
    # 总统计
    total_stats = defaultdict(int)
    
    # 处理每个文件夹
    for folder_name in folders_to_filter:
        folder_path = base_dir / folder_name
        
        if not folder_path.exists():
            print(f"\n⚠ {folder_name}: 文件夹不存在，跳过")
            continue
        
        print(f"\n处理: {folder_name}")
        
        # 筛选原始图片
        stats = filter_original_images(folder_path)
        
        # 更新总统计
        for key, value in stats.items():
            total_stats[key] += value
        
        # 打印统计
        print(f"  ✓ 保留原始图片: {stats['original_images']} 张")
        print(f"  ✗ 删除增强图片: {stats['augmented_images_deleted']} 张")
        if stats['orphan_labels_deleted'] > 0:
            print(f"  ✗ 删除孤立标签: {stats['orphan_labels_deleted']} 个")
    
    # 打印总统计
    print("\n" + "=" * 80)
    print("筛选完成！")
    print("=" * 80)
    print(f"总计:")
    print(f"  保留原始图片: {total_stats['original_images']} 张")
    print(f"  删除增强图片: {total_stats['augmented_images_deleted']} 张")
    print(f"  删除孤立标签: {total_stats['orphan_labels_deleted']} 个")
    
    # 验证最终结果
    print("\n" + "=" * 80)
    print("验证最终结果")
    print("=" * 80)
    
    for folder_name in folders_to_filter:
        folder_path = base_dir / folder_name
        if folder_path.exists():
            images_dir = folder_path / "images"
            labels_dir = folder_path / "labels"
            
            num_images = len(list(images_dir.glob("*.png"))) + len(list(images_dir.glob("*.jpg"))) if images_dir.exists() else 0
            num_labels = len(list(labels_dir.glob("*.txt"))) if labels_dir.exists() else 0
            
            print(f"\n{folder_name}:")
            print(f"  原始图片: {num_images} 张")
            print(f"  标签文件: {num_labels} 个")

if __name__ == '__main__':
    main()
