"""
个人页详细标注数据增强 - 使用已验证的增强方法
"""
import json
import shutil
from pathlib import Path
from PIL import Image, ImageEnhance
import random


def augment_image_yolo(image_path, label_path, output_img_dir, output_label_dir, base_name):
    """对单张图片进行数据增强 - 30倍增强（YOLO格式）"""
    img = Image.open(image_path)
    
    # 读取YOLO标注
    with open(label_path, 'r', encoding='utf-8') as f:
        labels = f.read()
    
    augmented_count = 0
    
    # 1. 原图
    original_img_path = output_img_dir / f"{base_name}_original.png"
    original_label_path = output_label_dir / f"{base_name}_original.txt"
    img.save(original_img_path)
    with open(original_label_path, 'w', encoding='utf-8') as f:
        f.write(labels)
    augmented_count += 1
    
    # 2-5. 亮度调整（4个级别）
    for i, factor in enumerate([0.7, 0.85, 1.15, 1.3], 1):
        enhancer = ImageEnhance.Brightness(img)
        bright_img = enhancer.enhance(factor)
        
        bright_img_path = output_img_dir / f"{base_name}_bright{i}.png"
        bright_label_path = output_label_dir / f"{base_name}_bright{i}.txt"
        bright_img.save(bright_img_path)
        with open(bright_label_path, 'w', encoding='utf-8') as f:
            f.write(labels)
        augmented_count += 1
    
    # 6-9. 对比度调整（4个级别）
    for i, factor in enumerate([0.7, 0.85, 1.15, 1.3], 1):
        enhancer = ImageEnhance.Contrast(img)
        contrast_img = enhancer.enhance(factor)
        
        contrast_img_path = output_img_dir / f"{base_name}_contrast{i}.png"
        contrast_label_path = output_label_dir / f"{base_name}_contrast{i}.txt"
        contrast_img.save(contrast_img_path)
        with open(contrast_label_path, 'w', encoding='utf-8') as f:
            f.write(labels)
        augmented_count += 1
    
    # 10-13. 色彩饱和度调整（4个级别）
    for i, factor in enumerate([0.7, 0.85, 1.15, 1.3], 1):
        enhancer = ImageEnhance.Color(img)
        color_img = enhancer.enhance(factor)
        
        color_img_path = output_img_dir / f"{base_name}_color{i}.png"
        color_label_path = output_label_dir / f"{base_name}_color{i}.txt"
        color_img.save(color_img_path)
        with open(color_label_path, 'w', encoding='utf-8') as f:
            f.write(labels)
        augmented_count += 1
    
    # 14-17. 锐度调整（4个级别）
    for i, factor in enumerate([0.5, 0.75, 1.25, 1.5], 1):
        enhancer = ImageEnhance.Sharpness(img)
        sharp_img = enhancer.enhance(factor)
        
        sharp_img_path = output_img_dir / f"{base_name}_sharp{i}.png"
        sharp_label_path = output_label_dir / f"{base_name}_sharp{i}.txt"
        sharp_img.save(sharp_img_path)
        with open(sharp_label_path, 'w', encoding='utf-8') as f:
            f.write(labels)
        augmented_count += 1
    
    # 18-23. 组合增强：亮度+对比度（6种组合）
    brightness_factors = [0.8, 0.9, 1.1]
    contrast_factors = [0.8, 1.2]
    combo_idx = 1
    for b_factor in brightness_factors:
        for c_factor in contrast_factors:
            enhancer = ImageEnhance.Brightness(img)
            combo_img = enhancer.enhance(b_factor)
            enhancer = ImageEnhance.Contrast(combo_img)
            combo_img = enhancer.enhance(c_factor)
            
            combo_img_path = output_img_dir / f"{base_name}_combo_bc_{combo_idx}.png"
            combo_label_path = output_label_dir / f"{base_name}_combo_bc_{combo_idx}.txt"
            combo_img.save(combo_img_path)
            with open(combo_label_path, 'w', encoding='utf-8') as f:
                f.write(labels)
            augmented_count += 1
            combo_idx += 1
    
    # 24-27. 组合增强：对比度+锐度（4种组合）
    contrast_factors = [0.85, 1.15]
    sharp_factors = [0.75, 1.25]
    combo_idx = 1
    for c_factor in contrast_factors:
        for s_factor in sharp_factors:
            enhancer = ImageEnhance.Contrast(img)
            combo_img = enhancer.enhance(c_factor)
            enhancer = ImageEnhance.Sharpness(combo_img)
            combo_img = enhancer.enhance(s_factor)
            
            combo_img_path = output_img_dir / f"{base_name}_combo_cs_{combo_idx}.png"
            combo_label_path = output_label_dir / f"{base_name}_combo_cs_{combo_idx}.txt"
            combo_img.save(combo_img_path)
            with open(combo_label_path, 'w', encoding='utf-8') as f:
                f.write(labels)
            augmented_count += 1
            combo_idx += 1
    
    # 28-30. 三重组合：亮度+对比度+色彩（3种组合）
    combos = [
        (0.9, 0.9, 0.9),
        (1.0, 1.1, 1.1),
        (1.1, 0.9, 1.0)
    ]
    for idx, (b, c, col) in enumerate(combos, 1):
        enhancer = ImageEnhance.Brightness(img)
        combo_img = enhancer.enhance(b)
        enhancer = ImageEnhance.Contrast(combo_img)
        combo_img = enhancer.enhance(c)
        enhancer = ImageEnhance.Color(combo_img)
        combo_img = enhancer.enhance(col)
        
        combo_img_path = output_img_dir / f"{base_name}_combo_triple_{idx}.png"
        combo_label_path = output_label_dir / f"{base_name}_combo_triple_{idx}.txt"
        combo_img.save(combo_img_path)
        with open(combo_label_path, 'w', encoding='utf-8') as f:
            f.write(labels)
        augmented_count += 1
    
    return augmented_count


def main():
    """主函数"""
    print("=" * 70)
    print("个人页详细标注数据增强 - 30倍增强")
    print("=" * 70)
    
    # 数据集路径
    dataset_path = Path("yolo_dataset/profile_detailed")
    train_images = dataset_path / "train" / "images"
    train_labels = dataset_path / "train" / "labels"
    
    if not train_images.exists():
        print(f"❌ 训练图片目录不存在: {train_images}")
        return
    
    # 获取所有原始图片（不包括已增强的）
    all_images = list(train_images.glob("*.png")) + list(train_images.glob("*.jpg"))
    original_images = [img for img in all_images if not any(
        suffix in img.stem for suffix in ['_original', '_bright', '_contrast', '_color', '_sharp', '_combo']
    )]
    
    print(f"\n[数据集信息]")
    print(f"  原始图片: {len(original_images)} 张")
    print(f"  增强倍数: 30倍")
    print(f"  预计生成: {len(original_images) * 30} 张")
    
    # 开始增强
    print(f"\n[开始增强]")
    total_augmented = 0
    
    for idx, img_path in enumerate(original_images, 1):
        # 获取对应的标注文件
        label_path = train_labels / f"{img_path.stem}.txt"
        
        if not label_path.exists():
            print(f"  ⚠️  跳过（无标注）: {img_path.name}")
            continue
        
        # 增强
        count = augment_image_yolo(
            img_path, 
            label_path, 
            train_images, 
            train_labels, 
            img_path.stem
        )
        total_augmented += count
        
        if idx % 10 == 0:
            print(f"  已处理: {idx}/{len(original_images)}")
    
    print(f"  已处理: {len(original_images)}/{len(original_images)}")
    
    # 统计最终数量
    final_count = len(list(train_images.glob("*.png"))) + len(list(train_images.glob("*.jpg")))
    
    print(f"\n✅ 数据增强完成！")
    print(f"  原始图片: {len(original_images)} 张")
    print(f"  增强图片: {total_augmented} 张")
    print(f"  训练集总计: {final_count} 张")
    
    print(f"\n下一步：")
    print(f"  重新训练: python train_profile_detailed.py")
    print("=" * 70)


if __name__ == '__main__':
    main()
