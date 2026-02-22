"""
整理原始标注图到统一的文件夹结构

将所有原始标注图整理到 original_annotations/ 文件夹中
每个模型一个子文件夹，包含 images/ 和 labels/ 子目录
"""

import shutil
from pathlib import Path

def organize_annotations():
    """整理原始标注图"""
    print("=" * 80)
    print("整理原始标注图到统一文件夹结构")
    print("=" * 80)
    
    # 创建目标目录
    target_base = Path("original_annotations")
    target_base.mkdir(exist_ok=True)
    
    # 源目录
    source_base = Path("training_data_completed")
    
    if not source_base.exists():
        print("⚠ training_data_completed 文件夹不存在")
        return
    
    # 统计信息
    total_folders = 0
    total_images = 0
    total_labels = 0
    
    # 遍历所有子文件夹
    for source_folder in source_base.iterdir():
        if not source_folder.is_dir():
            continue
        
        folder_name = source_folder.name
        print(f"\n处理: {folder_name}")
        
        # 创建目标文件夹
        target_folder = target_base / folder_name
        target_images = target_folder / "images"
        target_labels = target_folder / "labels"
        
        target_images.mkdir(parents=True, exist_ok=True)
        target_labels.mkdir(parents=True, exist_ok=True)
        
        # 复制图片和标签
        images_copied = 0
        labels_copied = 0
        
        # 情况1: 图片在 images/ 子文件夹中
        source_images = source_folder / "images"
        if source_images.exists():
            for img_file in source_images.glob("*.png"):
                shutil.copy2(img_file, target_images / img_file.name)
                images_copied += 1
            for img_file in source_images.glob("*.jpg"):
                shutil.copy2(img_file, target_images / img_file.name)
                images_copied += 1
        
        # 情况2: 图片直接在根目录下
        else:
            for img_file in source_folder.glob("*.png"):
                shutil.copy2(img_file, target_images / img_file.name)
                images_copied += 1
            for img_file in source_folder.glob("*.jpg"):
                shutil.copy2(img_file, target_images / img_file.name)
                images_copied += 1
        
        # 复制标签文件
        source_labels = source_folder / "labels"
        if source_labels.exists():
            for label_file in source_labels.glob("*.txt"):
                shutil.copy2(label_file, target_labels / label_file.name)
                labels_copied += 1
        else:
            # 标签直接在根目录下
            for label_file in source_folder.glob("*.txt"):
                shutil.copy2(label_file, target_labels / label_file.name)
                labels_copied += 1
        
        # 复制 annotations.json（如果存在）
        ann_file = source_folder / "annotations.json"
        if ann_file.exists():
            shutil.copy2(ann_file, target_folder / "annotations.json")
        
        print(f"  图片: {images_copied} 张")
        print(f"  标签: {labels_copied} 个")
        
        total_folders += 1
        total_images += images_copied
        total_labels += labels_copied
    
    # 打印总结
    print("\n" + "=" * 80)
    print("整理完成！")
    print("=" * 80)
    print(f"文件夹数: {total_folders} 个")
    print(f"图片总数: {total_images} 张")
    print(f"标签总数: {total_labels} 个")
    print(f"\n所有原始标注图已整理到: {target_base.absolute()}")
    
    # 列出所有文件夹
    print("\n" + "=" * 80)
    print("文件夹列表：")
    print("=" * 80)
    
    for folder in sorted(target_base.iterdir()):
        if folder.is_dir():
            images_dir = folder / "images"
            labels_dir = folder / "labels"
            
            num_images = len(list(images_dir.glob("*.png"))) + len(list(images_dir.glob("*.jpg")))
            num_labels = len(list(labels_dir.glob("*.txt")))
            
            print(f"\n{folder.name}:")
            print(f"  图片: {num_images} 张")
            print(f"  标签: {num_labels} 个")

if __name__ == '__main__':
    organize_annotations()
