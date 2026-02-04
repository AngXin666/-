"""
对 page_classifier_dataset_updated 中少于100张的类型进行数据增强
"""
from pathlib import Path
from PIL import Image, ImageEnhance
import random
import shutil


def augment_image(img, augment_factor=5):
    """对单张图片进行数据增强
    
    Args:
        img: PIL Image对象
        augment_factor: 增强倍数（每张原图生成多少张增强图）
    
    Returns:
        增强后的图片列表（包括原图）
    """
    augmented_images = [img]  # 包含原图
    
    # 生成 augment_factor-1 张增强图
    for i in range(augment_factor - 1):
        # 随机选择增强方式
        aug_img = img.copy()
        
        # 1. 亮度调整 (0.7-1.3)
        if random.random() > 0.3:
            enhancer = ImageEnhance.Brightness(aug_img)
            factor = random.uniform(0.7, 1.3)
            aug_img = enhancer.enhance(factor)
        
        # 2. 对比度调整 (0.7-1.4)
        if random.random() > 0.3:
            enhancer = ImageEnhance.Contrast(aug_img)
            factor = random.uniform(0.7, 1.4)
            aug_img = enhancer.enhance(factor)
        
        # 3. 色彩饱和度调整 (0.7-1.3)
        if random.random() > 0.5:
            enhancer = ImageEnhance.Color(aug_img)
            factor = random.uniform(0.7, 1.3)
            aug_img = enhancer.enhance(factor)
        
        # 4. 锐度调整 (0.5-1.6)
        if random.random() > 0.5:
            enhancer = ImageEnhance.Sharpness(aug_img)
            factor = random.uniform(0.5, 1.6)
            aug_img = enhancer.enhance(factor)
        
        augmented_images.append(aug_img)
    
    return augmented_images


def augment_class(source_dir, target_dir, class_name, augment_factor=5):
    """增强单个类别的数据
    
    Args:
        source_dir: 源数据集目录
        target_dir: 目标数据集目录
        class_name: 类别名称
        augment_factor: 增强倍数
    
    Returns:
        (原始数量, 增强后数量)
    """
    source_class_dir = source_dir / class_name
    target_class_dir = target_dir / class_name
    target_class_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图片
    image_files = list(source_class_dir.glob('*.png')) + list(source_class_dir.glob('*.jpg'))
    original_count = len(image_files)
    
    if original_count == 0:
        return 0, 0
    
    total_count = 0
    
    for idx, img_path in enumerate(image_files):
        try:
            img = Image.open(img_path)
            
            # 生成增强图片
            augmented_images = augment_image(img, augment_factor)
            
            # 保存增强后的图片
            base_name = img_path.stem
            for aug_idx, aug_img in enumerate(augmented_images):
                if aug_idx == 0:
                    # 原图
                    save_path = target_class_dir / f"{base_name}.png"
                else:
                    # 增强图
                    save_path = target_class_dir / f"{base_name}_aug_{aug_idx}.png"
                
                aug_img.save(save_path)
                total_count += 1
            
            if (idx + 1) % 10 == 0:
                print(f"    已处理 {idx + 1}/{original_count} 张...")
        
        except Exception as e:
            print(f"    ✗ 处理失败: {img_path.name} - {e}")
            continue
    
    return original_count, total_count


def main():
    """主函数"""
    print("=" * 60)
    print("页面分类器数据增强 - 只增强少于100张的类型")
    print("=" * 60)
    
    source_dir = Path('page_classifier_dataset_updated')
    target_dir = Path('page_classifier_dataset_updated')  # 直接在原目录增强
    
    if not source_dir.exists():
        print(f"\n❌ 源目录不存在: {source_dir}")
        return
    
    # 需要增强的类型（少于100张）
    classes_to_augment = {
        '首页': 49,
        '签到弹窗': 51,
        '温馨提示': 52,
        '签到页': 55,
        '转账确认弹窗': 56
    }
    
    print(f"\n需要增强的类型: {len(classes_to_augment)} 个")
    for class_name, count in classes_to_augment.items():
        print(f"  - {class_name}: {count}张")
    
    # 计算增强倍数（目标300-500张，与其他类型数量接近）
    augment_factor = 8  # 每张原图生成8张（原图+7张增强图）
    
    print(f"\n增强倍数: {augment_factor}x")
    print(f"目标: 每个类型300-500张图片（与其他类型数量接近）")
    print()
    
    # 对每个类别进行增强
    total_stats = {}
    
    for class_name in classes_to_augment.keys():
        print(f"正在增强: {class_name}")
        
        original_count, augmented_count = augment_class(
            source_dir, target_dir, class_name, augment_factor
        )
        
        total_stats[class_name] = {
            'original': original_count,
            'augmented': augmented_count
        }
        
        print(f"  ✓ {class_name}: {original_count} → {augmented_count} 张\n")
    
    # 打印统计信息
    print("=" * 60)
    print("增强完成统计")
    print("=" * 60)
    
    for class_name, stats in total_stats.items():
        print(f"  {class_name}: {stats['original']} → {stats['augmented']} 张")
    
    print(f"\n✓ 数据增强完成！")
    print(f"\n下一步:")
    print(f"  python train_page_classifier_pytorch.py")


if __name__ == '__main__':
    main()
