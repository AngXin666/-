"""
数据增强 - 为4类页面分类器增强数据

用法：
    python augment_4class_data.py
"""
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance
import shutil


def augment_image(image_path, output_dir, base_name, augment_factor=10):
    """对单张图片进行数据增强"""
    img = Image.open(image_path)
    augmented_images = []
    
    # 1. 原图
    original_path = output_dir / f"{base_name}_original.png"
    img.save(original_path)
    augmented_images.append(original_path)
    
    # 2. 亮度调整 (4张)
    for i, factor in enumerate([0.7, 0.85, 1.15, 1.3], 1):
        enhancer = ImageEnhance.Brightness(img)
        bright_img = enhancer.enhance(factor)
        path = output_dir / f"{base_name}_bright_{i}.png"
        bright_img.save(path)
        augmented_images.append(path)
    
    # 3. 对比度调整 (3张)
    for i, factor in enumerate([0.7, 1.2, 1.4], 1):
        enhancer = ImageEnhance.Contrast(img)
        contrast_img = enhancer.enhance(factor)
        path = output_dir / f"{base_name}_contrast_{i}.png"
        contrast_img.save(path)
        augmented_images.append(path)
    
    # 4. 色彩调整 (2张)
    for i, factor in enumerate([0.8, 1.2], 1):
        enhancer = ImageEnhance.Color(img)
        color_img = enhancer.enhance(factor)
        path = output_dir / f"{base_name}_color_{i}.png"
        color_img.save(path)
        augmented_images.append(path)
    
    # 只返回需要的数量
    return augmented_images[:augment_factor]


def augment_dataset():
    """增强4类页面分类器数据集"""
    print("=" * 60)
    print("数据增强 - 4类页面分类器")
    print("=" * 60)
    
    source_dir = Path("page_classifier_dataset_4classes")
    target_dir = Path("page_classifier_dataset_4classes_augmented")
    
    if not source_dir.exists():
        print(f"❌ 源目录不存在: {source_dir}")
        return
    
    # 删除旧的增强数据集
    if target_dir.exists():
        print(f"\n🗑️  删除旧的增强数据集...")
        shutil.rmtree(target_dir)
    
    print(f"\n📂 源目录: {source_dir}")
    print(f"📂 目标目录: {target_dir}")
    
    # 统计原始数据
    print(f"\n📊 原始数据统计:")
    total_original = 0
    for class_dir in sorted(source_dir.iterdir()):
        if class_dir.is_dir():
            count = len(list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg")))
            print(f"  {class_dir.name}: {count}张")
            total_original += count
    
    print(f"  总计: {total_original}张")
    
    # 设置增强倍数
    augment_factor = 10
    print(f"\n🎨 数据增强配置:")
    print(f"  增强倍数: {augment_factor}x")
    print(f"  预计生成: {total_original * augment_factor}张")
    
    # 开始增强
    print(f"\n🚀 开始数据增强...")
    total_augmented = 0
    
    for class_dir in sorted(source_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        print(f"\n  处理类别: {class_name}")
        
        # 创建目标目录
        target_class_dir = target_dir / class_name
        target_class_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取所有图片
        images = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
        
        # 增强每张图片
        for i, img_path in enumerate(images, 1):
            base_name = img_path.stem
            augmented = augment_image(img_path, target_class_dir, base_name, augment_factor)
            total_augmented += len(augmented)
            
            if i % 10 == 0:
                print(f"    进度: {i}/{len(images)}")
        
        augmented_count = len(list(target_class_dir.glob("*.png")))
        print(f"    ✓ 完成: {len(images)}张 → {augmented_count}张")
    
    print(f"\n✅ 数据增强完成!")
    print(f"  原始图片: {total_original}张")
    print(f"  增强后: {total_augmented}张")
    print(f"  位置: {target_dir}")
    
    # 统计增强后的数据
    print(f"\n📊 增强后数据统计:")
    for class_dir in sorted(target_dir.iterdir()):
        if class_dir.is_dir():
            count = len(list(class_dir.glob("*.png")))
            print(f"  {class_dir.name}: {count}张")
    
    print(f"\n🎯 下一步:")
    print(f"  训练模型: python train_4class_classifier.py")


if __name__ == "__main__":
    augment_dataset()
