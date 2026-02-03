"""
数据增强 - 个人页详细标注数据集
Data Augmentation - Profile Detailed Annotation Dataset
"""

import os
import cv2
import numpy as np
from pathlib import Path
import random

# 数据集目录
DATASET_DIR = "yolo_dataset/profile_detailed"

# 增强倍数
AUGMENTATION_FACTOR = 30  # 每张图片生成30个增强版本

def augment_image(image, boxes):
    """对图片和标注框进行增强
    
    Args:
        image: 原始图片
        boxes: YOLO格式的标注框 [[class_id, x_center, y_center, width, height], ...]
    
    Returns:
        augmented_image: 增强后的图片
        augmented_boxes: 增强后的标注框
    """
    h, w = image.shape[:2]
    
    # 随机选择增强方法
    aug_type = random.choice([
        'brightness', 'contrast', 'rotation', 'flip', 
        'blur', 'noise', 'combo_bc', 'combo_br', 
        'combo_triple', 'combo_all'
    ])
    
    if aug_type == 'brightness':
        # 亮度调整 (-50 到 +50)
        value = random.randint(-50, 50)
        image = cv2.convertScaleAbs(image, alpha=1, beta=value)
        return image, boxes
    
    elif aug_type == 'contrast':
        # 对比度调整 (0.5 到 1.5)
        alpha = random.uniform(0.5, 1.5)
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        return image, boxes
    
    elif aug_type == 'rotation':
        # 小角度旋转 (-5 到 +5 度)
        angle = random.uniform(-5, 5)
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
        return image, boxes
    
    elif aug_type == 'flip':
        # 水平翻转
        image = cv2.flip(image, 1)
        # 翻转标注框
        new_boxes = []
        for box in boxes:
            class_id, x_center, y_center, width, height = box
            x_center = 1.0 - x_center  # 翻转x坐标
            new_boxes.append([class_id, x_center, y_center, width, height])
        return image, new_boxes
    
    elif aug_type == 'blur':
        # 轻微模糊
        kernel_size = random.choice([3, 5])
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return image, boxes
    
    elif aug_type == 'noise':
        # 添加高斯噪声
        noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
        image = cv2.add(image, noise)
        return image, boxes
    
    elif aug_type == 'combo_bc':
        # 组合：亮度 + 对比度
        value = random.randint(-30, 30)
        alpha = random.uniform(0.7, 1.3)
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=value)
        return image, boxes
    
    elif aug_type == 'combo_br':
        # 组合：亮度 + 旋转
        value = random.randint(-30, 30)
        image = cv2.convertScaleAbs(image, alpha=1, beta=value)
        angle = random.uniform(-3, 3)
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
        return image, boxes
    
    elif aug_type == 'combo_triple':
        # 组合：亮度 + 对比度 + 模糊
        value = random.randint(-20, 20)
        alpha = random.uniform(0.8, 1.2)
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=value)
        image = cv2.GaussianBlur(image, (3, 3), 0)
        return image, boxes
    
    elif aug_type == 'combo_all':
        # 组合：亮度 + 对比度 + 噪声
        value = random.randint(-20, 20)
        alpha = random.uniform(0.8, 1.2)
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=value)
        noise = np.random.normal(0, 5, image.shape).astype(np.uint8)
        image = cv2.add(image, noise)
        return image, boxes
    
    return image, boxes


def augment_dataset():
    """对数据集进行增强"""
    
    print("=" * 70)
    print("数据增强 - 个人页详细标注数据集")
    print("=" * 70)
    
    dataset_path = Path(DATASET_DIR)
    
    # 只增强训练集
    train_images_dir = dataset_path / 'train' / 'images'
    train_labels_dir = dataset_path / 'train' / 'labels'
    
    if not train_images_dir.exists():
        print(f"❌ 训练集目录不存在: {train_images_dir}")
        return
    
    # 获取所有训练图片
    image_files = list(train_images_dir.glob('*.png')) + list(train_images_dir.glob('*.jpg'))
    
    print(f"\n[1] 找到 {len(image_files)} 张训练图片")
    print(f"[2] 每张图片生成 {AUGMENTATION_FACTOR} 个增强版本")
    print(f"[3] 预计生成 {len(image_files) * AUGMENTATION_FACTOR} 张增强图片")
    
    # 开始增强
    print(f"\n开始数据增强...")
    
    total_generated = 0
    
    for img_idx, img_path in enumerate(image_files, 1):
        # 读取图片
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  ⚠️  无法读取图片: {img_path.name}")
            continue
        
        # 读取标注
        label_path = train_labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            print(f"  ⚠️  标注文件不存在: {label_path.name}")
            continue
        
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        boxes = []
        for line in lines:
            if line.strip():
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                boxes.append([class_id, x_center, y_center, width, height])
        
        # 生成增强版本
        for aug_idx in range(AUGMENTATION_FACTOR):
            # 增强图片和标注
            aug_image, aug_boxes = augment_image(image.copy(), boxes.copy())
            
            # 保存增强后的图片
            aug_img_name = f"{img_path.stem}_aug_{aug_idx}{img_path.suffix}"
            aug_img_path = train_images_dir / aug_img_name
            cv2.imwrite(str(aug_img_path), aug_image)
            
            # 保存增强后的标注
            aug_label_name = f"{img_path.stem}_aug_{aug_idx}.txt"
            aug_label_path = train_labels_dir / aug_label_name
            
            with open(aug_label_path, 'w', encoding='utf-8') as f:
                for box in aug_boxes:
                    class_id, x_center, y_center, width, height = box
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            total_generated += 1
        
        if img_idx % 5 == 0:
            print(f"  进度: {img_idx}/{len(image_files)} ({total_generated} 张已生成)")
    
    print(f"\n✓ 数据增强完成！")
    print(f"  原始图片: {len(image_files)} 张")
    print(f"  增强图片: {total_generated} 张")
    print(f"  总计: {len(image_files) + total_generated} 张")
    
    print("\n" + "=" * 70)
    print("下一步: 运行训练脚本开始训练")
    print("=" * 70)


if __name__ == '__main__':
    augment_dataset()
