"""
个人页区域检测数据增强 - 30倍增强
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
    with open(label_path, 'r') as f:
        labels = f.read()
    
    augmented_count = 0
    
    # 1. 原图
    original_img_path = output_img_dir / f"{base_name}_original.png"
    original_label_path = output_label_dir / f"{base_name}_original.txt"
    img.save(original_img_path)
    with open(original_label_path, 'w') as f:
        f.write(labels)
    augmented_count += 1
    
    # 2-5. 亮度调整（4个级别）
    for i, factor in enumerate([0.7, 0.85, 1.15, 1.3], 1):
        enhancer = ImageEnhance.Brightness(img)
        aug_img = enhancer.enhance(factor)
        path = output_img_dir / f"{base_name}_bright_{i}.png"
        label_path_out = output_label_dir / f"{base_name}_bright_{i}.txt"
        aug_img.save(path)
        with open(label_path_out, 'w') as f:
            f.write(labels)
        augmented_count += 1
    
    # 6-9. 对比度调整（4个级别）
    for i, factor in enumerate([0.6, 0.8, 1.2, 1.4], 1):
        enhancer = ImageEnhance.Contrast(img)
        aug_img = enhancer.enhance(factor)
        path = output_img_dir / f"{base_name}_contrast_{i}.png"
        label_path_out = output_label_dir / f"{base_name}_contrast_{i}.txt"
        aug_img.save(path)
        with open(label_path_out, 'w') as f:
            f.write(labels)
        augmented_count += 1
    
    # 10-12. 色彩饱和度调整（3个级别）
    for i, factor in enumerate([0.7, 1.15, 1.3], 1):
        enhancer = ImageEnhance.Color(img)
        aug_img = enhancer.enhance(factor)
        path = output_img_dir / f"{base_name}_color_{i}.png"
        label_path_out = output_label_dir / f"{base_name}_color_{i}.txt"
        aug_img.save(path)
        with open(label_path_out, 'w') as f:
            f.write(labels)
        augmented_count += 1
    
    # 13-15. 锐度调整（3个级别）
    for i, factor in enumerate([0.5, 1.3, 1.6], 1):
        enhancer = ImageEnhance.Sharpness(img)
        aug_img = enhancer.enhance(factor)
        path = output_img_dir / f"{base_name}_sharp_{i}.png"
        label_path_out = output_label_dir / f"{base_name}_sharp_{i}.txt"
        aug_img.save(path)
        with open(label_path_out, 'w') as f:
            f.write(labels)
        augmented_count += 1
    
    # 16-21. 亮度+对比度组合（6种组合）
    brightness_factors = [0.8, 1.2]
    contrast_factors = [0.8, 1.2, 1.4]
    combo_idx = 1
    for b_factor in brightness_factors:
        for c_factor in contrast_factors:
            enhancer = ImageEnhance.Brightness(img)
            temp_img = enhancer.enhance(b_factor)
            enhancer = ImageEnhance.Contrast(temp_img)
            aug_img = enhancer.enhance(c_factor)
            path = output_img_dir / f"{base_name}_combo_bc_{combo_idx}.png"
            label_path_out = output_label_dir / f"{base_name}_combo_bc_{combo_idx}.txt"
            aug_img.save(path)
            with open(label_path_out, 'w') as f:
                f.write(labels)
            augmented_count += 1
            combo_idx += 1
    
    # 22-25. 色彩+锐度组合（4种组合）
    color_factors = [0.8, 1.2]
    sharp_factors = [0.7, 1.4]
    combo_idx = 1
    for col_factor in color_factors:
        for sh_factor in sharp_factors:
            enhancer = ImageEnhance.Color(img)
            temp_img = enhancer.enhance(col_factor)
            enhancer = ImageEnhance.Sharpness(temp_img)
            aug_img = enhancer.enhance(sh_factor)
            path = output_img_dir / f"{base_name}_combo_cs_{combo_idx}.png"
            label_path_out = output_label_dir / f"{base_name}_combo_cs_{combo_idx}.txt"
            aug_img.save(path)
            with open(label_path_out, 'w') as f:
                f.write(labels)
            augmented_count += 1
            combo_idx += 1
    
    # 26-29. 亮度+色彩组合（4种组合）
    brightness_factors = [0.85, 1.15]
    color_factors = [0.85, 1.15]
    combo_idx = 1
    for b_factor in brightness_factors:
        for col_factor in color_factors:
            enhancer = ImageEnhance.Brightness(img)
            temp_img = enhancer.enhance(b_factor)
            enhancer = ImageEnhance.Color(temp_img)
            aug_img = enhancer.enhance(col_factor)
            path = output_img_dir / f"{base_name}_combo_bcol_{combo_idx}.png"
            label_path_out = output_label_dir / f"{base_name}_combo_bcol_{combo_idx}.txt"
            aug_img.save(path)
            with open(label_path_out, 'w') as f:
                f.write(labels)
            augmented_count += 1
            combo_idx += 1
    
    # 30-31. 三重组合（2种）
    # 组合1: 亮+对比+锐
    enhancer = ImageEnhance.Brightness(img)
    temp_img = enhancer.enhance(1.1)
    enhancer = ImageEnhance.Contrast(temp_img)
    temp_img = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Sharpness(temp_img)
    aug_img = enhancer.enhance(1.3)
    path = output_img_dir / f"{base_name}_combo_triple_1.png"
    label_path_out = output_label_dir / f"{base_name}_combo_triple_1.txt"
    aug_img.save(path)
    with open(label_path_out, 'w') as f:
        f.write(labels)
    augmented_count += 1
    
    # 组合2: 暗+对比+色彩
    enhancer = ImageEnhance.Brightness(img)
    temp_img = enhancer.enhance(0.9)
    enhancer = ImageEnhance.Contrast(temp_img)
    temp_img = enhancer.enhance(1.3)
    enhancer = ImageEnhance.Color(temp_img)
    aug_img = enhancer.enhance(1.1)
    path = output_img_dir / f"{base_name}_combo_triple_2.png"
    label_path_out = output_label_dir / f"{base_name}_combo_triple_2.txt"
    aug_img.save(path)
    with open(label_path_out, 'w') as f:
        f.write(labels)
    augmented_count += 1
    
    return augmented_count


def augment_profile_regions():
    """增强个人页区域检测数据集"""
    
    print("=" * 70)
    print("个人页区域检测数据增强 - 30倍增强")
    print("=" * 70)
    
    # 目录
    dataset_dir = Path("yolo_dataset/profile_regions")
    train_img_dir = dataset_dir / "images" / "train"
    train_label_dir = dataset_dir / "labels" / "train"
    
    if not train_img_dir.exists():
        print(f"\n❌ 训练图片目录不存在: {train_img_dir}")
        print(f"   请先运行: python prepare_profile_region_data.py")
        return
    
    # 获取所有原始图片（不包含已增强的）
    all_images = list(train_img_dir.glob("*.png"))
    original_images = [img for img in all_images if not any(
        keyword in img.stem for keyword in 
        ['_bright_', '_contrast_', '_color_', '_sharp_', '_combo_', '_original', '_rotate']
    )]
    
    print(f"\n[数据集信息]")
    print(f"  原始图片: {len(original_images)} 张")
    print(f"  增强倍数: 30倍")
    print(f"  预计生成: {len(original_images) * 30} 张")
    
    print(f"\n[开始增强]")
    total_augmented = 0
    
    for i, img_file in enumerate(original_images):
        label_file = train_label_dir / (img_file.stem + ".txt")
        
        if not label_file.exists():
            print(f"  ⚠ 标注文件不存在: {label_file.name}")
            continue
        
        aug_count = augment_image_yolo(
            img_file, label_file,
            train_img_dir, train_label_dir,
            img_file.stem
        )
        total_augmented += aug_count
        
        if (i + 1) % 10 == 0:
            print(f"  已处理: {i+1}/{len(original_images)}")
    
    # 统计最终数据
    final_count = len(list(train_img_dir.glob("*.png")))
    
    print(f"\n✅ 数据增强完成！")
    print(f"  原始图片: {len(original_images)} 张")
    print(f"  增强图片: {total_augmented} 张")
    print(f"  训练集总计: {final_count} 张")
    
    print(f"\n下一步：")
    print(f"  运行训练: python train_profile_regions_yolo.py")


if __name__ == '__main__':
    augment_profile_regions()
