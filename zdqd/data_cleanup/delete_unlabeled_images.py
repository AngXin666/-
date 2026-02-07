"""
删除 training_data 和 training_data_completed 中所有未标记的图片
只保留有对应 .txt 标签文件的图片
"""
import os
from pathlib import Path

def delete_unlabeled_in_folder(folder_path):
    """删除文件夹中没有标签的图片"""
    folder = Path(folder_path)
    
    if not folder.exists():
        return 0, 0, 0
    
    deleted_images = 0
    kept_images = 0
    orphan_labels = 0
    
    # 检查是否有 images 子文件夹（YOLO格式）
    images_folders = []
    labels_folders = []
    
    images_train = folder / 'images' / 'train'
    images_val = folder / 'images' / 'val'
    labels_train = folder / 'labels' / 'train'
    labels_val = folder / 'labels' / 'val'
    
    if images_train.exists():
        images_folders.append((images_train, labels_train))
    if images_val.exists():
        images_folders.append((images_val, labels_val))
    
    # 如果没有 images 子文件夹，直接在文件夹中处理
    if not images_folders:
        images_folders.append((folder, folder))
    
    for img_folder, label_folder in images_folders:
        # 获取所有图片
        all_images = list(img_folder.glob('*.png')) + list(img_folder.glob('*.jpg'))
        
        for img_file in all_images:
            # 检查是否有对应的标签文件
            label_file = label_folder / (img_file.stem + '.txt')
            
            if not label_file.exists():
                # 没有标签，删除图片
                img_file.unlink()
                deleted_images += 1
            else:
                kept_images += 1
        
        # 检查孤立的标签文件（有标签但没有图片）
        if label_folder.exists():
            all_labels = list(label_folder.glob('*.txt'))
            for label_file in all_labels:
                # 检查是否有对应的图片
                img_png = img_folder / (label_file.stem + '.png')
                img_jpg = img_folder / (label_file.stem + '.jpg')
                
                if not img_png.exists() and not img_jpg.exists():
                    # 孤立标签，删除
                    label_file.unlink()
                    orphan_labels += 1
    
    return deleted_images, kept_images, orphan_labels

def process_directory(dir_path, dir_name):
    """处理目录中的所有子文件夹"""
    directory = Path(dir_path)
    
    if not directory.exists():
        print(f"{dir_name} 文件夹不存在！")
        return
    
    print(f"\n{'=' * 80}")
    print(f"处理 {dir_name}")
    print(f"{'=' * 80}")
    
    total_deleted = 0
    total_kept = 0
    total_orphan = 0
    
    for subfolder in sorted(directory.iterdir()):
        if subfolder.is_dir():
            print(f"\n处理文件夹: {subfolder.name}")
            deleted, kept, orphan = delete_unlabeled_in_folder(subfolder)
            
            if deleted > 0 or orphan > 0:
                print(f"  删除未标记图片: {deleted} 张")
                print(f"  保留已标记图片: {kept} 张")
                if orphan > 0:
                    print(f"  删除孤立标签: {orphan} 个")
            elif kept > 0:
                print(f"  保留已标记图片: {kept} 张（无未标记图片）")
            else:
                print(f"  空文件夹或无图片")
            
            total_deleted += deleted
            total_kept += kept
            total_orphan += orphan
    
    print(f"\n{'-' * 80}")
    print(f"{dir_name} 总计:")
    print(f"  删除未标记图片: {total_deleted} 张")
    print(f"  保留已标记图片: {total_kept} 张")
    print(f"  删除孤立标签: {total_orphan} 个")
    
    return total_deleted, total_kept, total_orphan

def main():
    """主函数"""
    print("=" * 80)
    print("删除未标记图片")
    print("=" * 80)
    
    # 处理 training_data
    deleted1, kept1, orphan1 = process_directory('training_data', 'training_data')
    
    # 处理 training_data_completed
    deleted2, kept2, orphan2 = process_directory('training_data_completed', 'training_data_completed')
    
    # 总结
    print(f"\n{'=' * 80}")
    print("总结")
    print(f"{'=' * 80}")
    print(f"总共删除未标记图片: {deleted1 + deleted2} 张")
    print(f"总共保留已标记图片: {kept1 + kept2} 张")
    print(f"总共删除孤立标签: {orphan1 + orphan2} 个")

if __name__ == '__main__':
    main()
