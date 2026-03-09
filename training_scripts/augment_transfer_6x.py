"""
转账页面数据增强脚本 - 6倍增强
对转账相关页面进行数据增强，生成6倍数据
"""
import os
import sys
from pathlib import Path
import random
from PIL import Image, ImageEnhance, ImageFilter
import shutil

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============ 配置区域 ============
# 指定要增强的类别（转账流程需要的页面）
SELECTED_CLASSES = [
    "个人页已登陆",
    "个人页未登陆",
    "钱包页",
    "转账页",
    "转账弹窗",
    "首页"
]

# 增强倍数（5倍：原始数据 × 5）
AUGMENT_MULTIPLIER = 5

# 只增强低于此数量的类别
MIN_IMAGES_THRESHOLD = 100
# ==================================


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
        import numpy as np
        img_array = np.array(image)
        noise = np.random.normal(0, 5, img_array.shape).astype(np.uint8)
        img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(img_array)
    
    return image


def augment_dataset(training_data_dir, selected_classes, multiplier=6, min_threshold=100):
    """对数据集进行增强
    
    Args:
        training_data_dir: 训练数据目录
        selected_classes: 要增强的类别列表
        multiplier: 增强倍数
        min_threshold: 只增强低于此数量的类别
    """
    print("\n" + "=" * 80)
    print("🎨 转账页面数据增强")
    print("=" * 80)
    
    print(f"\n📁 数据目录: {training_data_dir}")
    print(f"🔢 增强倍数: {multiplier}x")
    print(f"📏 增强阈值: 只增强低于 {min_threshold} 张的类别")
    
    print(f"\n📋 检查类别:")
    for i, class_name in enumerate(selected_classes, 1):
        class_dir = Path(training_data_dir) / class_name
        if class_dir.exists():
            count = len(list(class_dir.glob("*.png")))
            status = "需要增强" if count < min_threshold else "跳过（已足够）"
            print(f"  {i}. {class_name}: {count} 张 - {status}")
    
    total_original = 0
    total_augmented = 0
    
    for class_name in selected_classes:
        class_dir = Path(training_data_dir) / class_name
        
        if not class_dir.exists():
            print(f"\n⚠️  警告: 类别目录不存在: {class_name}")
            continue
        
        # 获取所有现有图片（包括已增强的）
        all_images = list(class_dir.glob("*.png"))
        current_count = len(all_images)
        
        # 如果已经超过阈值，跳过
        if current_count >= min_threshold:
            print(f"\n⏭️  跳过 {class_name}: 已有 {current_count} 张（>= {min_threshold}）")
            continue
        
        print(f"\n{'=' * 80}")
        print(f"📂 处理类别: {class_name}")
        print(f"{'=' * 80}")
        print(f"  • 当前图片: {current_count} 张")
        
        # 计算需要生成的数量（5倍增强 = 原始数量 × 5）
        target_count = current_count * multiplier
        need_generate = target_count - current_count
        
        print(f"  • 目标数量: {target_count} 张")
        print(f"  • 需要生成: {need_generate} 张")
        
        class_augmented = 0
        
        # 随机选择图片进行增强，直到达到目标数量
        import random
        for i in range(need_generate):
            try:
                # 随机选择一张现有图片
                source_img_path = random.choice(all_images)
                
                # 加载图片
                image = Image.open(source_img_path).convert('RGB')
                
                # 生成增强图片
                aug_image = augment_image(image, seed=i * 1000 + random.randint(0, 9999))
                
                # 保存增强图片
                # 使用时间戳避免文件名冲突
                import time
                timestamp = int(time.time() * 1000000) % 1000000
                aug_name = f"aug_{timestamp}_{i}.png"
                aug_path = class_dir / aug_name
                
                aug_image.save(aug_path)
                class_augmented += 1
                
                # 显示进度
                progress = (i + 1) / need_generate * 100
                bar_length = 40
                filled = int(bar_length * (i + 1) / need_generate)
                bar = '█' * filled + '░' * (bar_length - filled)
                print(f"\r  [{bar}] {progress:.1f}% ({i + 1}/{need_generate})", end='', flush=True)
                
            except Exception as e:
                print(f"\n  ❌ 处理失败: {e}")
        
        print(f"\n  ✅ 完成! 生成了 {class_augmented} 张增强图")
        print(f"  📊 最终数量: {current_count + class_augmented} 张")
        
        total_original += current_count
        total_augmented += class_augmented
    
    print("\n" + "=" * 80)
    print("✅ 数据增强完成!")
    print("=" * 80)
    print(f"\n📊 统计:")
    print(f"  • 增强前总数: {total_original} 张")
    print(f"  • 新增图片: {total_augmented} 张")
    print(f"  • 增强后总数: {total_original + total_augmented} 张")


def main():
    """主函数"""
    # 配置
    script_dir = Path(__file__).parent.parent
    training_data_dir = script_dir / "标注工具_完整独立版" / "training_data"
    
    if not training_data_dir.exists():
        print(f"\n❌ 错误: 训练数据目录不存在: {training_data_dir}")
        return
    
    # 执行增强
    augment_dataset(training_data_dir, SELECTED_CLASSES, AUGMENT_MULTIPLIER, MIN_IMAGES_THRESHOLD)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户取消操作")
    except Exception as e:
        print(f"\n\n❌ 操作失败: {e}")
        import traceback
        traceback.print_exc()
