"""
删除已完成文件夹中的增强图片
只保留原始标注图片
"""
import os
import shutil
from pathlib import Path

def is_augmented_image(filename):
    """判断是否为增强图片"""
    # 增强图片的特征标识
    augment_markers = [
        '_bright_', '_color_', '_contrast_', '_saturation_',
        '_combo_bc_', '_combo_cs_', '_combo_bcol_', '_combo_triple_',
        ' - 副本', '- 副本'
    ]
    
    for marker in augment_markers:
        if marker in filename:
            return True
    return False

def delete_augmented_from_folder(folder_path):
    """删除指定文件夹中的增强图片和对应标签"""
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"文件夹不存在: {folder}")
        return
    
    print(f"\n处理文件夹: {folder.name}")
    print("=" * 60)
    
    deleted_count = 0
    kept_count = 0
    
    # 处理 images/train 和 images/val
    for split in ['train', 'val']:
        images_dir = folder / 'images' / split
        labels_dir = folder / 'labels' / split
        
        if not images_dir.exists():
            continue
        
        print(f"\n处理 {split} 集...")
        
        for img_file in images_dir.glob('*'):
            if img_file.is_file():
                if is_augmented_image(img_file.name):
                    # 删除图片
                    img_file.unlink()
                    
                    # 删除对应的标签文件
                    label_file = labels_dir / (img_file.stem + '.txt')
                    if label_file.exists():
                        label_file.unlink()
                    
                    deleted_count += 1
                    if deleted_count <= 5:  # 只显示前5个
                        print(f"  删除: {img_file.name}")
                else:
                    kept_count += 1
    
    # 删除缓存文件
    for cache_file in (folder / 'labels').glob('*.cache'):
        cache_file.unlink()
        print(f"\n删除缓存: {cache_file.name}")
    
    print(f"\n统计:")
    print(f"  删除增强图片: {deleted_count} 张")
    print(f"  保留原始图片: {kept_count} 张")
    
    return deleted_count, kept_count

def main():
    """主函数"""
    completed_dir = Path('training_data_completed')
    
    if not completed_dir.exists():
        print("training_data_completed 文件夹不存在！")
        return
    
    print("开始删除增强图片...")
    print("=" * 60)
    
    # 获取所有子文件夹
    all_folders = [f for f in completed_dir.iterdir() if f.is_dir()]
    
    print(f"\n找到 {len(all_folders)} 个文件夹:")
    for folder in all_folders:
        print(f"  - {folder.name}")
    
    print("\n" + "=" * 60)
    
    total_deleted = 0
    total_kept = 0
    
    for folder_path in all_folders:
        deleted, kept = delete_augmented_from_folder(folder_path)
        total_deleted += deleted
        total_kept += kept
    
    print("\n" + "=" * 60)
    print("删除完成！")
    print("=" * 60)
    print(f"总共删除: {total_deleted} 张增强图片")
    print(f"总共保留: {total_kept} 张原始图片")

if __name__ == '__main__':
    main()
