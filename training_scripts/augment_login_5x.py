"""
登录专用模型数据增强脚本 - 5倍增强
对低于100张的类别进行5倍数据增强
"""
# [2026-03-01] 修改原因：使用PIL库替代albumentations，与其他增强脚本保持一致

import os
import sys
from pathlib import Path
import random
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import time

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 需要增强的类别（低于100张）
CATEGORIES_TO_AUGMENT = [
    "个人页广告"
]

# 数据增强配置（5倍增强）
AUGMENTATION_FACTOR = 5


def augment_image(image, seed=None):
    """对图片进行随机增强
    
    Args:
        image: PIL Image对象
        seed: 随机种子（可选）
        
    Returns:
        增强后的PIL Image对象
    """
    if seed is not None:
        random.seed(seed)
    
    # 1. 随机亮度调整 (0.8-1.2)
    if random.random() > 0.3:
        enhancer = ImageEnhance.Brightness(image)
        factor = random.uniform(0.8, 1.2)
        image = enhancer.enhance(factor)
    
    # 2. 随机对比度调整 (0.8-1.2)
    if random.random() > 0.3:
        enhancer = ImageEnhance.Contrast(image)
        factor = random.uniform(0.8, 1.2)
        image = enhancer.enhance(factor)
    
    # 3. 随机饱和度调整 (0.8-1.2)
    if random.random() > 0.3:
        enhancer = ImageEnhance.Color(image)
        factor = random.uniform(0.8, 1.2)
        image = enhancer.enhance(factor)
    
    # 4. 随机锐度调整 (0.8-1.2)
    if random.random() > 0.3:
        enhancer = ImageEnhance.Sharpness(image)
        factor = random.uniform(0.8, 1.2)
        image = enhancer.enhance(factor)
    
    # 5. 随机轻微模糊 (10%概率)
    if random.random() > 0.9:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.0)))
    
    # 6. 随机轻微噪声 (10%概率)
    if random.random() > 0.9:
        img_array = np.array(image)
        noise = np.random.normal(0, 5, img_array.shape).astype(np.uint8)
        img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(img_array)
    
    return image


def augment_category(category_name: str, base_path: Path, augmentation_factor: int = 5):
    """对指定类别进行数据增强
    
    Args:
        category_name: 类别名称
        base_path: 训练数据基础路径
        augmentation_factor: 增强倍数
    """
    category_path = base_path / category_name
    
    if not category_path.exists():
        print(f"⚠️ 类别文件夹不存在: {category_path}")
        return
    
    # 获取所有原始图片（不包含增强图）
    original_images = [f for f in category_path.glob("*.png") if "_aug_" not in f.name and "aug_" not in f.name]
    
    if not original_images:
        print(f"⚠️ {category_name}: 没有找到原始图片")
        return
    
    print(f"\n{'='*60}")
    print(f"开始增强: {category_name}")
    print(f"原始图片数量: {len(original_images)}")
    print(f"增强倍数: {augmentation_factor}x")
    print(f"预计生成: {len(original_images) * augmentation_factor} 张图片")
    print(f"{'='*60}\n")
    
    augmented_count = 0
    need_generate = len(original_images) * augmentation_factor
    
    # 对每张原始图片进行增强
    for img_idx, img_path in enumerate(original_images):
        # 读取图片
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"⚠️ 无法读取图片: {img_path} - {e}")
            continue
        
        # 生成增强图片
        for i in range(augmentation_factor):
            try:
                # 应用数据增强
                aug_image = augment_image(image, seed=img_idx * 1000 + i)
                
                # 保存增强图片
                timestamp = int(time.time() * 1000000) % 1000000
                aug_filename = f"aug_{timestamp}_{augmented_count}.png"
                aug_path = category_path / aug_filename
                aug_image.save(aug_path)
                
                augmented_count += 1
                
                # 显示进度
                progress = augmented_count / need_generate * 100
                bar_length = 40
                filled = int(bar_length * augmented_count / need_generate)
                bar = '█' * filled + '░' * (bar_length - filled)
                print(f"\r  [{bar}] {progress:.1f}% ({augmented_count}/{need_generate})", end='', flush=True)
                
            except Exception as e:
                print(f"\n⚠️ 增强失败 {img_path.name} (第{i+1}次): {e}")
                continue
    
    # 统计最终数量
    total_images = len(list(category_path.glob("*.png")))
    print(f"\n\n✅ {category_name} 增强完成:")
    print(f"   原始图片: {len(original_images)} 张")
    print(f"   增强图片: {augmented_count} 张")
    print(f"   总计: {total_images} 张")


def main():
    """主函数"""
    print("="*60)
    print("🎨 登录专用模型 - 数据增强脚本")
    print("="*60)
    print(f"增强倍数: {AUGMENTATION_FACTOR}x")
    print(f"需要增强的类别: {len(CATEGORIES_TO_AUGMENT)} 个")
    print("="*60)
    
    # 训练数据路径
    base_path = project_root / "标注工具_完整独立版" / "training_data"
    
    if not base_path.exists():
        print(f"❌ 训练数据路径不存在: {base_path}")
        return
    
    total_original = 0
    total_augmented = 0
    
    # 对每个类别进行增强
    for category in CATEGORIES_TO_AUGMENT:
        category_path = base_path / category
        if category_path.exists():
            original_count = len([f for f in category_path.glob("*.png") if "_aug_" not in f.name and "aug_" not in f.name])
            total_original += original_count
        
        augment_category(category, base_path, AUGMENTATION_FACTOR)
        
        if category_path.exists():
            final_count = len(list(category_path.glob("*.png")))
            total_augmented += (final_count - original_count)
    
    print("\n" + "="*60)
    print("✅ 所有类别增强完成！")
    print("="*60)
    print(f"\n📊 统计:")
    print(f"  • 原始图片总数: {total_original} 张")
    print(f"  • 增强图片总数: {total_augmented} 张")
    print(f"  • 总计: {total_original + total_augmented} 张")


if __name__ == "__main__":
    main()
