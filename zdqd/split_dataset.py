"""
分割数据集为训练集和验证集
Split dataset into train and validation sets
"""

from pathlib import Path
import shutil
import random


def split_dataset(train_ratio=0.8):
    """分割数据集"""
    
    print("=" * 70)
    print("分割数据集为训练集和验证集")
    print("=" * 70)
    
    # 目录
    train_img_dir = Path("yolo_dataset/profile_numbers/images/train")
    train_label_dir = Path("yolo_dataset/profile_numbers/labels/train")
    val_img_dir = Path("yolo_dataset/profile_numbers/images/val")
    val_label_dir = Path("yolo_dataset/profile_numbers/labels/val")
    
    # 创建验证集目录
    val_img_dir.mkdir(parents=True, exist_ok=True)
    val_label_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图片
    image_files = list(train_img_dir.glob("*.png"))
    
    if not image_files:
        print(f"\n❌ 没有找到图片文件")
        return
    
    print(f"\n[数据集信息]")
    print(f"  总图片数: {len(image_files)}")
    print(f"  训练比例: {train_ratio:.0%}")
    print(f"  验证比例: {1-train_ratio:.0%}")
    
    # 随机打乱
    random.seed(42)
    random.shuffle(image_files)
    
    # 计算分割点
    split_idx = int(len(image_files) * train_ratio)
    val_images = image_files[split_idx:]
    
    print(f"\n[分割结果]")
    print(f"  训练集: {split_idx} 张")
    print(f"  验证集: {len(val_images)} 张")
    
    # 移动验证集文件
    print(f"\n[移动文件]")
    for img_file in val_images:
        # 移动图片
        shutil.move(str(img_file), str(val_img_dir / img_file.name))
        
        # 移动标注
        label_file = train_label_dir / (img_file.stem + ".txt")
        if label_file.exists():
            shutil.move(str(label_file), str(val_label_dir / label_file.name))
        
        print(f"  ✓ {img_file.name}")
    
    print(f"\n✅ 数据集分割完成！")
    print(f"  训练集: {train_img_dir}")
    print(f"  验证集: {val_img_dir}")


if __name__ == '__main__':
    split_dataset(train_ratio=0.8)
