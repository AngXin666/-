"""
对所有类别进行5倍多样化增强
"""

import os
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from tqdm import tqdm
import random

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

def augment_image_diverse(image_path, output_dir, aug_index):
    """对单张图片进行多样化增强（加大尺度）"""
    img = Image.open(image_path)
    
    base_name = image_path.stem
    ext = image_path.suffix
    
    # 5种不同的增强方式（加大尺度）
    if aug_index == 1:
        # 增强1: 大幅旋转 + 亮度调整
        angle = random.uniform(-25, 25)
        augmented = img.rotate(angle, fillcolor=(255, 255, 255))
        enhancer = ImageEnhance.Brightness(augmented)
        augmented = enhancer.enhance(random.uniform(0.6, 1.4))
        
    elif aug_index == 2:
        # 增强2: 水平翻转 + 大幅对比度调整
        augmented = img.transpose(Image.FLIP_LEFT_RIGHT)
        enhancer = ImageEnhance.Contrast(augmented)
        augmented = enhancer.enhance(random.uniform(0.6, 1.5))
        
    elif aug_index == 3:
        # 增强3: 模糊 + 色彩调整
        augmented = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.5)))
        enhancer = ImageEnhance.Color(augmented)
        augmented = enhancer.enhance(random.uniform(0.7, 1.3))
        
    elif aug_index == 4:
        # 增强4: 锐化 + 大幅亮度调整
        augmented = img.filter(ImageFilter.SHARPEN)
        enhancer = ImageEnhance.Brightness(augmented)
        augmented = enhancer.enhance(random.uniform(0.7, 1.4))
        
    else:  # aug_index == 5
        # 增强5: 大幅旋转 + 对比度 + 色彩
        angle = random.uniform(-20, 20)
        augmented = img.rotate(angle, fillcolor=(255, 255, 255))
        enhancer = ImageEnhance.Contrast(augmented)
        augmented = enhancer.enhance(random.uniform(0.7, 1.3))
        enhancer = ImageEnhance.Color(augmented)
        augmented = enhancer.enhance(random.uniform(0.8, 1.2))
    
    # 保存增强图片
    output_path = output_dir / f"{base_name}_aug{aug_index}{ext}"
    augmented.save(output_path)

def main():
    print("=" * 60)
    print("对所有类别进行5倍多样化增强")
    print("=" * 60)
    
    categories = sorted([d for d in TRAINING_DATA_DIR.iterdir() if d.is_dir()])
    
    # 对所有类别进行增强
    for category_dir in categories:
        count = count_images_in_category(category_dir)
        print(f"\n正在增强 {category_dir.name} ({count}张原始图片)...")
        
        # 获取所有图片（包括副本和aug_开头的，但不包含 _aug）
        image_files = [
            f for f in category_dir.iterdir()
            if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}
            and '_aug' not in f.stem
        ]
        
        # 对每张原始图片进行5倍增强
        for img_file in tqdm(image_files, desc=f"  增强 {category_dir.name}"):
            try:
                for i in range(1, 6):  # 5倍增强
                    augment_image_diverse(img_file, category_dir, i)
            except Exception as e:
                print(f"    ⚠️ 增强失败: {img_file.name} - {e}")
        
        # 统计增强后的数量
        new_count = count_images_in_category(category_dir)
        print(f"  ✓ {category_dir.name}: {count}张 → {new_count}张")
    
    print("\n" + "=" * 60)
    print("最终统计")
    print("=" * 60)
    
    # 最终统计
    total_images = 0
    for category_dir in categories:
        count = count_images_in_category(category_dir)
        total_images += count
        print(f"  {category_dir.name}: {count}张")
    
    print(f"\n总计: {len(categories)} 个类别, {total_images} 张图片")
    print("\n✓ 增强完成！")

if __name__ == "__main__":
    main()
