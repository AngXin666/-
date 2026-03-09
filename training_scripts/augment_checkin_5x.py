"""
签到流程页面数据增强脚本 - 5倍增强
只增强指定的5个类别
"""
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import random

# 要增强的类别
CATEGORIES = [
    "首页",
    "签到页",
    "签到弹窗",
    "温馨提示",
    "登录页"
]

# 增强倍数
AUGMENT_MULTIPLIER = 5

def augment_image(image):
    """对图片进行随机增强"""
    # 随机选择增强方式
    aug_type = random.choice(['brightness', 'contrast', 'blur', 'rotate'])
    
    if aug_type == 'brightness':
        enhancer = ImageEnhance.Brightness(image)
        factor = random.uniform(0.7, 1.3)
        return enhancer.enhance(factor)
    elif aug_type == 'contrast':
        enhancer = ImageEnhance.Contrast(image)
        factor = random.uniform(0.7, 1.3)
        return enhancer.enhance(factor)
    elif aug_type == 'blur':
        return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    elif aug_type == 'rotate':
        angle = random.uniform(-5, 5)
        return image.rotate(angle, fillcolor=(255, 255, 255))
    
    return image

def main():
    base_dir = Path("标注工具_完整独立版/training_data")
    
    print("\n" + "=" * 60)
    print("🎯 签到流程页面数据增强 (5倍)")
    print("=" * 60)
    
    total_original = 0
    total_augmented = 0
    
    for category in CATEGORIES:
        category_dir = base_dir / category
        
        if not category_dir.exists():
            print(f"\n⚠️  {category}: 目录不存在，跳过")
            continue
        
        # 获取原始图片（不包括已增强的）
        original_images = [f for f in category_dir.glob("*.png") if "_aug_" not in f.name]
        original_count = len(original_images)
        total_original += original_count
        
        # 计算需要生成的增强图片数量
        augment_count = original_count * AUGMENT_MULTIPLIER
        
        print(f"\n📁 {category}")
        print(f"  • 原始图片: {original_count} 张")
        print(f"  • 需要生成: {augment_count} 张")
        
        # 生成增强图片
        generated = 0
        for i in range(augment_count):
            # 随机选择一张原始图片
            source_img_path = random.choice(original_images)
            
            # 加载图片
            img = Image.open(source_img_path)
            
            # 增强
            aug_img = augment_image(img)
            
            # 保存
            aug_filename = f"{source_img_path.stem}_aug_{i+1}.png"
            aug_path = category_dir / aug_filename
            aug_img.save(aug_path)
            
            generated += 1
            
            # 显示进度
            if (generated % 10 == 0) or (generated == augment_count):
                progress = generated / augment_count * 100
                print(f"\r  进度: {generated}/{augment_count} ({progress:.1f}%)", end='', flush=True)
        
        print(f"\n  ✓ 完成: 生成了 {generated} 张增强图片")
        total_augmented += generated
    
    print("\n" + "=" * 60)
    print("✅ 数据增强完成!")
    print("=" * 60)
    print(f"\n📊 统计:")
    print(f"  • 原始图片总数: {total_original} 张")
    print(f"  • 增强图片总数: {total_augmented} 张")
    print(f"  • 总计: {total_original + total_augmented} 张")

if __name__ == '__main__':
    main()
