"""
只增强广告页数据
"""
import os
import sys
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import random
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 全局进度计数器（线程安全）
progress_lock = threading.Lock()
progress_counter = 0


def augment_image(image_path, output_dir, base_name, index):
    """对单张图片进行数据增强"""
    img = Image.open(image_path)
    
    # 随机选择增强方式
    augment_type = random.choice(['brightness', 'contrast', 'rotate', 'flip', 'blur', 'color'])
    
    if augment_type == 'brightness':
        enhancer = ImageEnhance.Brightness(img)
        factor = random.uniform(0.7, 1.3)
        img = enhancer.enhance(factor)
    elif augment_type == 'contrast':
        enhancer = ImageEnhance.Contrast(img)
        factor = random.uniform(0.7, 1.3)
        img = enhancer.enhance(factor)
    elif augment_type == 'rotate':
        angle = random.uniform(-5, 5)
        img = img.rotate(angle, fillcolor=(255, 255, 255))
    elif augment_type == 'flip':
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    elif augment_type == 'blur':
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    elif augment_type == 'color':
        enhancer = ImageEnhance.Color(img)
        factor = random.uniform(0.8, 1.2)
        img = enhancer.enhance(factor)
    
    # 保存增强后的图片
    output_path = output_dir / f"{base_name}_aug_{index}.png"
    img.save(output_path)
    return output_path


def main():
    print("\n" + "=" * 80)
    print("🎨 广告页数据增强")
    print("=" * 80)
    
    # 广告页目录
    ad_dir = Path("标注工具_完整独立版/training_data/广告页")
    
    if not ad_dir.exists():
        print(f"\n❌ 错误: 广告页目录不存在: {ad_dir}")
        return
    
    # 统计原始图片（排除增强图片）
    images = [f for f in ad_dir.glob("*.png") 
             if not f.stem.endswith(('_aug', '_augmented')) and not f.name.startswith('aug_')]
    
    if not images:
        print("\n❌ 错误: 没有找到原始图片")
        return
    
    original_count = len(images)
    target_total = 650  # 目标总数
    augment_count = max(0, target_total - original_count)
    
    print(f"\n📁 数据目录: {ad_dir}")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📊 统计信息:")
    print(f"  • 原始图片: {original_count}张")
    print(f"  • 需要生成: {augment_count}张")
    print(f"  • 目标总数: {target_total}张")
    
    if augment_count == 0:
        print("\n✓ 已达到目标数量，无需增强")
        return
    
    print(f"\n🎯 增强方法: 亮度、对比度、旋转、翻转、模糊、色彩")
    input("\n按 Enter 键开始增强，或 Ctrl+C 取消...")
    
    print("\n" + "=" * 80)
    print("🚀 开始数据增强...")
    print("=" * 80)
    
    start_time = datetime.now()
    
    # 计算每张原始图片需要生成多少增强图片
    images_per_original = augment_count // original_count
    extra_images = augment_count % original_count
    
    # 准备所有增强任务
    tasks = []
    aug_counter = 0
    
    for img_idx, img_path in enumerate(images, 1):
        base_name = img_path.stem
        
        # 计算这张图片需要生成多少增强图片
        num_augments = images_per_original
        if img_idx <= extra_images:
            num_augments += 1
        
        # 为每个增强操作创建任务
        for aug_idx in range(num_augments):
            tasks.append((img_path, ad_dir, base_name, aug_counter + 1))
            aug_counter += 1
    
    # 使用16线程并行处理
    total_processed = 0
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(augment_image, *task): task for task in tasks}
        
        for future in as_completed(futures):
            try:
                future.result()
                
                with progress_lock:
                    total_processed += 1
                
                # 每10张或完成时显示进度
                if total_processed % 10 == 0 or total_processed == augment_count:
                    progress = (total_processed / augment_count) * 100
                    bar_length = min(40, int(progress / 2.5))
                    print(f"\r  进度: [{total_processed}/{augment_count}] ({progress:.1f}%) "
                          f"[{'█' * bar_length}{' ' * (40 - bar_length)}]", end='', flush=True)
                
            except Exception as e:
                task = futures[future]
                print(f"\n  ⚠️  警告: 增强失败 {task[0].name}: {e}")
    
    print(f"\n  ✓ 完成: 生成了 {total_processed}张增强图片")
    
    # 完成统计
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 80)
    print("✅ 数据增强完成!")
    print("=" * 80)
    print(f"\n📊 统计信息:")
    print(f"  • 原始图片: {original_count}张")
    print(f"  • 生成图片: {total_processed}张")
    print(f"  • 总计图片: {original_count + total_processed}张")
    print(f"  • 耗时: {duration:.1f}秒")
    print(f"  • 平均速度: {total_processed / duration:.1f}张/秒")
    print(f"\n⏰ 完成时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户取消操作")
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
