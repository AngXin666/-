"""
清理和增强训练数据脚本

功能：
1. 删除所有类别中的增强图片（文件名包含 _aug）
2. 重新统计所有类别的图片数量
3. 对低于100张的类别进行5倍增强
4. 对大于等于100张的类别进行1倍增强
"""

import os
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

# 训练数据目录
TRAINING_DATA_DIR = Path("标注工具_完整独立版/training_data")

def count_images_in_category(category_path):
    """统计类别中的图片数量"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    count = 0
    for file in category_path.iterdir():
        if file.suffix.lower() in image_extensions:
            count += 1
    return count

def delete_augmented_images(category_path):
    """删除增强图片（文件名包含 _aug）"""
    deleted_count = 0
    for file in category_path.iterdir():
        if '_aug' in file.stem:
            file.unlink()
            deleted_count += 1
    return deleted_count

def augment_image(image_path, output_dir, num_augmentations=5):
    """对单张图片进行增强"""
    img = Image.open(image_path)
    img_array = np.array(img)
    
    base_name = image_path.stem
    ext = image_path.suffix
    
    for i in range(num_augmentations):
        # 随机增强：旋转、翻转、亮度调整
        augmented = img_array.copy()
        
        # 随机旋转 (-10 到 10 度)
        angle = np.random.uniform(-10, 10)
        augmented_img = Image.fromarray(augmented).rotate(angle, fillcolor=(255, 255, 255))
        
        # 随机水平翻转
        if np.random.random() > 0.5:
            augmented_img = augmented_img.transpose(Image.FLIP_LEFT_RIGHT)
        
        # 随机亮度调整
        brightness_factor = np.random.uniform(0.8, 1.2)
        augmented_array = np.array(augmented_img) * brightness_factor
        augmented_array = np.clip(augmented_array, 0, 255).astype(np.uint8)
        
        # 保存增强图片
        output_path = output_dir / f"{base_name}_aug{i+1}{ext}"
        Image.fromarray(augmented_array).save(output_path)

def main():
    print("=" * 60)
    print("步骤 1: 删除所有类别中的增强图片")
    print("=" * 60)
    
    categories = sorted([d for d in TRAINING_DATA_DIR.iterdir() if d.is_dir()])
    
    # 统计初始状态
    initial_stats = {}
    for category_dir in categories:
        count = count_images_in_category(category_dir)
        initial_stats[category_dir.name] = count
    
    # 删除所有类别的增强图片
    deleted_total = 0
    for category_dir in categories:
        count = initial_stats[category_dir.name]
        deleted = delete_augmented_images(category_dir)
        if deleted > 0:
            deleted_total += deleted
            print(f"  {category_dir.name}: {count}张 → 删除了 {deleted} 张增强图片")
    
    print(f"\n共删除 {deleted_total} 张增强图片")
    
    print("\n" + "=" * 60)
    print("步骤 2: 重新统计所有类别的图片数量")
    print("=" * 60)
    
    # 重新统计
    current_stats = {}
    for category_dir in categories:
        count = count_images_in_category(category_dir)
        current_stats[category_dir.name] = count
        print(f"  {category_dir.name}: {count}张")
    
    print("\n" + "=" * 60)
    print("步骤 3: 对所有类别进行增强")
    print("=" * 60)
    
    # 对所有类别进行增强
    for category_dir in categories:
        count = current_stats[category_dir.name]
        
        # 低于100张的类别：5倍增强
        if count < 100:
            num_aug = 5
            print(f"\n正在增强 {category_dir.name} ({count}张) - 5倍增强...")
        # 大于等于100张的类别：1倍增强
        else:
            num_aug = 1
            print(f"\n正在增强 {category_dir.name} ({count}张) - 1倍增强...")
        
        # 获取所有原始图片（不包含 _aug）
        image_files = [
            f for f in category_dir.iterdir()
            if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}
            and '_aug' not in f.stem
        ]
        
        # 对每张原始图片进行增强
        for img_file in tqdm(image_files, desc=f"  增强 {category_dir.name}"):
            try:
                augment_image(img_file, category_dir, num_augmentations=num_aug)
            except Exception as e:
                print(f"    ⚠️ 增强失败: {img_file.name} - {e}")
        
        # 统计增强后的数量
        new_count = count_images_in_category(category_dir)
        print(f"  ✓ {category_dir.name}: {count}张 → {new_count}张")
    
    print("\n" + "=" * 60)
    print("步骤 4: 最终统计")
    print("=" * 60)
    
    # 最终统计
    final_stats = {}
    total_images = 0
    for category_dir in categories:
        count = count_images_in_category(category_dir)
        final_stats[category_dir.name] = count
        total_images += count
        print(f"  {category_dir.name}: {count}张")
    
    print(f"\n总计: {len(categories)} 个类别, {total_images} 张图片")
    print("\n✓ 完成！")

if __name__ == "__main__":
    main()
