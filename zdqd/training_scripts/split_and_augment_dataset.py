"""
分割数据集并进行数据增强
Split dataset and apply data augmentation
"""

from pathlib import Path
import shutil
import random
import cv2
import numpy as np
from PIL import Image


def augment_image_and_label(img_path, label_path, output_img_dir, output_label_dir, aug_count=3):
    """增强单张图片和标注"""
    
    # 读取图片
    img = cv2.imread(str(img_path))
    if img is None:
        return 0
    
    h, w = img.shape[:2]
    
    # 读取标注
    with open(label_path, 'r') as f:
        labels = f.readlines()
    
    augmented = 0
    
    # 原始图片
    base_name = img_path.stem
    
    # 增强1: 亮度调整
    for i, brightness in enumerate([0.8, 1.2]):
        aug_img = cv2.convertScaleAbs(img, alpha=brightness, beta=0)
        
        aug_img_path = output_img_dir / f"{base_name}_bright{i}.png"
        aug_label_path = output_label_dir / f"{base_name}_bright{i}.txt"
        
        cv2.imwrite(str(aug_img_path), aug_img)
        shutil.copy2(label_path, aug_label_path)
        augmented += 1
    
    # 增强2: 对比度调整
    for i, contrast in enumerate([0.8, 1.2]):
        aug_img = cv2.convertScaleAbs(img, alpha=contrast, beta=0)
        
        aug_img_path = output_img_dir / f"{base_name}_contrast{i}.png"
        aug_label_path = output_label_dir / f"{base_name}_contrast{i}.txt"
        
        cv2.imwrite(str(aug_img_path), aug_img)
        shutil.copy2(label_path, aug_label_path)
        augmented += 1
    
    # 增强3: 轻微旋转 (-5到5度)
    for i, angle in enumerate([-3, 3]):
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aug_img = cv2.warpAffine(img, M, (w, h))
        
        aug_img_path = output_img_dir / f"{base_name}_rotate{i}.png"
        aug_label_path = output_label_dir / f"{base_name}_rotate{i}.txt"
        
        cv2.imwrite(str(aug_img_path), aug_img)
        shutil.copy2(label_path, aug_label_path)
        augmented += 1
    
    return augmented


def split_and_augment_dataset(dataset_name="profile_regions", train_ratio=0.8, augment=True):
    """分割并增强数据集"""
    
    print("=" * 70)
    print(f"分割并增强数据集: {dataset_name}")
    print("=" * 70)
    
    # 目录
    dataset_dir = Path(f"yolo_dataset/{dataset_name}")
    train_img_dir = dataset_dir / "images" / "train"
    train_label_dir = dataset_dir / "labels" / "train"
    val_img_dir = dataset_dir / "images" / "val"
    val_label_dir = dataset_dir / "labels" / "val"
    
    # 创建验证集目录
    val_img_dir.mkdir(parents=True, exist_ok=True)
    val_label_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图片
    image_files = list(train_img_dir.glob("*.png"))
    
    if not image_files:
        print(f"\n❌ 没有找到图片文件")
        return
    
    print(f"\n[数据集信息]")
    print(f"  原始图片数: {len(image_files)}")
    print(f"  训练比例: {train_ratio:.0%}")
    print(f"  验证比例: {1-train_ratio:.0%}")
    print(f"  数据增强: {'是' if augment else '否'}")
    
    # 随机打乱
    random.seed(42)
    random.shuffle(image_files)
    
    # 计算分割点
    split_idx = int(len(image_files) * train_ratio)
    train_images = image_files[:split_idx]
    val_images = image_files[split_idx:]
    
    print(f"\n[分割结果]")
    print(f"  训练集: {len(train_images)} 张")
    print(f"  验证集: {len(val_images)} 张")
    
    # 移动验证集文件
    print(f"\n[移动验证集]")
    for img_file in val_images:
        # 移动图片
        shutil.move(str(img_file), str(val_img_dir / img_file.name))
        
        # 移动标注
        label_file = train_label_dir / (img_file.stem + ".txt")
        if label_file.exists():
            shutil.move(str(label_file), str(val_label_dir / label_file.name))
    
    print(f"  ✓ 已移动 {len(val_images)} 张到验证集")
    
    # 数据增强
    if augment:
        print(f"\n[数据增强]")
        print(f"  增强训练集...")
        
        # 重新获取训练集图片（已经移除了验证集）
        train_images = list(train_img_dir.glob("*.png"))
        
        total_augmented = 0
        for i, img_file in enumerate(train_images):
            label_file = train_label_dir / (img_file.stem + ".txt")
            
            if not label_file.exists():
                continue
            
            aug_count = augment_image_and_label(
                img_file, label_file,
                train_img_dir, train_label_dir
            )
            total_augmented += aug_count
            
            if (i + 1) % 10 == 0:
                print(f"    已处理: {i+1}/{len(train_images)}")
        
        print(f"  ✓ 增强完成")
        print(f"    原始: {len(train_images)} 张")
        print(f"    增强: {total_augmented} 张")
        print(f"    总计: {len(train_images) + total_augmented} 张")
    
    # 统计最终数据
    final_train_count = len(list(train_img_dir.glob("*.png")))
    final_val_count = len(list(val_img_dir.glob("*.png")))
    
    print(f"\n✅ 数据集准备完成！")
    print(f"  训练集: {final_train_count} 张")
    print(f"  验证集: {final_val_count} 张")
    print(f"  总计: {final_train_count + final_val_count} 张")
    print(f"\n下一步：")
    print(f"  运行训练: python train_profile_regions_yolo.py")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='分割并增强数据集')
    parser.add_argument('--dataset', default='profile_regions', help='数据集名称')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='训练集比例')
    parser.add_argument('--no-augment', action='store_true', help='不进行数据增强')
    
    args = parser.parse_args()
    
    split_and_augment_dataset(
        dataset_name=args.dataset,
        train_ratio=args.train_ratio,
        augment=not args.no_augment
    )
