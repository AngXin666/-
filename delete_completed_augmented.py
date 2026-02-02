"""
删除 training_data_completed 文件夹中的所有增强图片
只保留原始标注图
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
        '副本'
    ]
    
    for marker in augment_markers:
        if marker in filename:
            return True
    return False

def delete_augmented_in_folder(folder_path):
    """删除文件夹中的增强图片和对应标签"""
    folder = Path(folder_path)
    
    if not folder.exists():
        return 0, 0
    
    deleted_count = 0
    kept_count = 0
    
    # 检查是否有 images 子文件夹（YOLO格式）
    images_train = folder / 'images' / 'train'
    images_val = folder / 'images' / 'val'
    labels_train = folder / 'labels' / 'train'
    labels_val = folder / 'labels' / 'val'
    
    folders_to_check = []
    
    if images_train.exists():
        folders_to_check.append((images_train, labels_train))
    if images_val.exists():
        folders_to_check.append((images_val, labels_val))
    
    # 如果没有 images 子文件夹，直接在文件夹中处理
    if not folders_to_check:
        folders_to_check.append((folder, folder))
    
    for img_folder, label_folder in folders_to_check:
        for img_file in list(img_folder.glob('*.png')) + list(img_folder.glob('*.jpg')):
            if is_augmented_image(img_file.name):
                # 删除图片
                img_file.unlink()
                
                # 删除对应的标签文件
                if label_folder.exists():
                    label_file = label_folder / (img_file.stem + '.txt')
                    if label_file.exists():
                        label_file.unlink()
                
                deleted_count += 1
            else:
                kept_count += 1
    
    # 删除缓存文件
    if (folder / 'labels').exists():
        for cache_file in (folder / 'labels').glob('*.cache'):
            cache_file.unlink()
    
    return deleted_count, kept_count

def main():
    """主函数"""
    completed_dir = Path('training_data_completed')
    
    if not completed_dir.exists():
        print("training_data_completed 文件夹不存在！")
        return
    
    print("=" * 80)
    print("开始删除 training_data_completed 中的所有增强图片...")
    print("=" * 80)
    
    total_deleted = 0
    total_kept = 0
    deleted_folders = []
    
    # 遍历所有子文件夹
    for folder in sorted(completed_dir.iterdir()):
        if folder.is_dir():
            # 删除 _augmented 文件夹
            if folder.name.endswith('_augmented'):
                print(f"\n删除增强文件夹: {folder.name}")
                try:
                    shutil.rmtree(folder)
                    deleted_folders.append(folder.name)
                    print(f"  ✓ 已删除整个文件夹")
                except Exception as e:
                    print(f"  ✗ 删除失败: {e}")
            else:
                # 删除文件夹中的增强图片
                print(f"\n处理文件夹: {folder.name}")
                deleted, kept = delete_augmented_in_folder(folder)
                
                if deleted > 0:
                    total_deleted += deleted
                    total_kept += kept
                    print(f"  删除增强图: {deleted} 张")
                    print(f"  保留原始图: {kept} 张")
                elif kept > 0:
                    total_kept += kept
                    print(f"  保留原始图: {kept} 张（无增强图）")
                else:
                    print(f"  空文件夹")
    
    print("\n" + "=" * 80)
    print("删除完成！")
    print("=" * 80)
    print(f"删除增强图片: {total_deleted} 张")
    print(f"保留原始图片: {total_kept} 张")
    print(f"删除增强文件夹: {len(deleted_folders)} 个")
    if deleted_folders:
        for folder_name in deleted_folders:
            print(f"  - {folder_name}")

if __name__ == '__main__':
    main()
