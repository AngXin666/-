"""
数据增强脚本 - 扩充训练数据
"""
import json
import shutil
from pathlib import Path
from PIL import Image, ImageEnhance
import random


def augment_image(image_path, output_dir, base_name, annotations):
    """对单张图片进行数据增强 - 30倍增强"""
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    augmented_data = []
    
    # 1. 原图
    original_path = output_dir / f"{base_name}_original.png"
    img.save(original_path)
    augmented_data.append((str(original_path), annotations))
    
    # 2-4. 亮度调整（3个级别）
    for i, factor in enumerate([0.7, 0.85, 1.15, 1.3], 1):
        enhancer = ImageEnhance.Brightness(img)
        bright_img = enhancer.enhance(factor)
        path = output_dir / f"{base_name}_bright_{i}.png"
        bright_img.save(path)
        augmented_data.append((str(path), annotations))
    
    # 5-8. 对比度调整（4个级别）
    for i, factor in enumerate([0.6, 0.8, 1.2, 1.4], 1):
        enhancer = ImageEnhance.Contrast(img)
        contrast_img = enhancer.enhance(factor)
        path = output_dir / f"{base_name}_contrast_{i}.png"
        contrast_img.save(path)
        augmented_data.append((str(path), annotations))
    
    # 9-11. 色彩饱和度调整（3个级别）
    for i, factor in enumerate([0.7, 1.15, 1.3], 1):
        enhancer = ImageEnhance.Color(img)
        color_img = enhancer.enhance(factor)
        path = output_dir / f"{base_name}_color_{i}.png"
        color_img.save(path)
        augmented_data.append((str(path), annotations))
    
    # 12-14. 锐度调整（3个级别）
    for i, factor in enumerate([0.5, 1.3, 1.6], 1):
        enhancer = ImageEnhance.Sharpness(img)
        sharp_img = enhancer.enhance(factor)
        path = output_dir / f"{base_name}_sharp_{i}.png"
        sharp_img.save(path)
        augmented_data.append((str(path), annotations))
    
    # 15-20. 亮度+对比度组合（6种组合）
    brightness_factors = [0.8, 1.2]
    contrast_factors = [0.8, 1.2, 1.4]
    combo_idx = 1
    for b_factor in brightness_factors:
        for c_factor in contrast_factors:
            enhancer = ImageEnhance.Brightness(img)
            temp_img = enhancer.enhance(b_factor)
            enhancer = ImageEnhance.Contrast(temp_img)
            combo_img = enhancer.enhance(c_factor)
            path = output_dir / f"{base_name}_combo_bc_{combo_idx}.png"
            combo_img.save(path)
            augmented_data.append((str(path), annotations))
            combo_idx += 1
    
    # 21-24. 色彩+锐度组合（4种组合）
    color_factors = [0.8, 1.2]
    sharp_factors = [0.7, 1.4]
    combo_idx = 1
    for col_factor in color_factors:
        for sh_factor in sharp_factors:
            enhancer = ImageEnhance.Color(img)
            temp_img = enhancer.enhance(col_factor)
            enhancer = ImageEnhance.Sharpness(temp_img)
            combo_img = enhancer.enhance(sh_factor)
            path = output_dir / f"{base_name}_combo_cs_{combo_idx}.png"
            combo_img.save(path)
            augmented_data.append((str(path), annotations))
            combo_idx += 1
    
    # 25-28. 亮度+色彩组合（4种组合）
    brightness_factors = [0.85, 1.15]
    color_factors = [0.85, 1.15]
    combo_idx = 1
    for b_factor in brightness_factors:
        for col_factor in color_factors:
            enhancer = ImageEnhance.Brightness(img)
            temp_img = enhancer.enhance(b_factor)
            enhancer = ImageEnhance.Color(temp_img)
            combo_img = enhancer.enhance(col_factor)
            path = output_dir / f"{base_name}_combo_bcol_{combo_idx}.png"
            combo_img.save(path)
            augmented_data.append((str(path), annotations))
            combo_idx += 1
    
    # 29-30. 三重组合（2种）
    # 组合1: 亮+对比+锐
    enhancer = ImageEnhance.Brightness(img)
    temp_img = enhancer.enhance(1.1)
    enhancer = ImageEnhance.Contrast(temp_img)
    temp_img = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Sharpness(temp_img)
    combo_img = enhancer.enhance(1.3)
    path = output_dir / f"{base_name}_combo_triple_1.png"
    combo_img.save(path)
    augmented_data.append((str(path), annotations))
    
    # 组合2: 暗+对比+色彩
    enhancer = ImageEnhance.Brightness(img)
    temp_img = enhancer.enhance(0.9)
    enhancer = ImageEnhance.Contrast(temp_img)
    temp_img = enhancer.enhance(1.3)
    enhancer = ImageEnhance.Color(temp_img)
    combo_img = enhancer.enhance(1.1)
    path = output_dir / f"{base_name}_combo_triple_2.png"
    combo_img.save(path)
    augmented_data.append((str(path), annotations))
    
    return augmented_data


def augment_page_data(page_type="首页"):
    """增强页面数据
    
    Args:
        page_type: 页面类型，如"首页"、"登录页"等
    """
    print("=" * 60)
    print(f"{page_type}数据增强")
    print("=" * 60)
    
    source_dir = Path(f"training_data/{page_type}")
    output_dir = Path(f"training_data/{page_type}_augmented")
    output_dir.mkdir(exist_ok=True)
    
    # 加载标注
    annotation_file = source_dir / "annotations.json"
    if not annotation_file.exists():
        print("\n❌ 找不到标注文件")
        return
    
    with open(annotation_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    print(f"\n原始数据: {len([k for k,v in annotations.items() if v])} 张已标注图片")
    
    # 增强后的标注数据
    augmented_annotations = {}
    total_augmented = 0
    
    for image_path_str, anns in annotations.items():
        if not anns:
            continue
        
        image_path = Path(image_path_str)
        if not image_path.exists():
            continue
        
        base_name = image_path.stem
        
        # 对图片进行增强
        augmented_list = augment_image(image_path, output_dir, base_name, anns)
        
        # 保存增强后的标注
        for aug_path, aug_anns in augmented_list:
            augmented_annotations[aug_path] = aug_anns
            total_augmented += 1
        
        if total_augmented % 10 == 0:
            print(f"  已处理 {total_augmented} 张...")
    
    # 保存增强后的标注文件
    aug_annotation_file = output_dir / "annotations.json"
    with open(aug_annotation_file, 'w', encoding='utf-8') as f:
        json.dump(augmented_annotations, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 数据增强完成！")
    print(f"原始图片: {len([k for k,v in annotations.items() if v])} 张")
    print(f"增强后图片: {total_augmented} 张")
    print(f"增强倍数: {total_augmented / len([k for k,v in annotations.items() if v]):.1f}x")
    print(f"\n增强数据保存在: {output_dir}")
    
    # 更新准备脚本使用增强后的数据
    print("\n下一步:")
    print(f"  1. 准备YOLO数据集")
    print(f"  2. 训练模型")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='数据增强脚本')
    parser.add_argument('--page_type', type=str, default='首页', help='页面类型，如"首页"、"登录页"')
    args = parser.parse_args()
    
    augment_page_data(args.page_type)
