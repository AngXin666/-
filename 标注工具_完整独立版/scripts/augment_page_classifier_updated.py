"""
页面分类器数据增强脚本 - 带详细日志
支持智能增强倍数和实时进度显示
支持16线程并行处理加速
"""
import os
import sys
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import random
import shutil
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 全局进度计数器（线程安全）
progress_lock = threading.Lock()
progress_counter = 0


def calculate_augment_count(image_count, mode='medium'):
    """根据增强模式计算目标图片数量
    
    Args:
        image_count: 原始图片数量
        mode: 增强模式 ('light'=轻度, 'medium'=中度, 'heavy'=重度)
    
    Returns:
        需要生成的增强图片数量
    """
    if mode == 'light':
        # 轻度: 目标200-300张
        target = 250
    elif mode == 'medium':
        # 中度: 目标500-800张
        target = 650
    elif mode == 'heavy':
        # 重度: 目标1000张左右
        target = 1000
    else:
        target = 650  # 默认中度
    
    # 计算需要生成的增强图片数量
    augment_count = max(0, target - image_count)
    return augment_count


def augment_image(image_path, output_dir, base_name, index):
    """对单张图片进行数据增强
    
    Args:
        image_path: 原始图片路径
        output_dir: 输出目录
        base_name: 基础文件名
        index: 增强索引
    
    Returns:
        增强后的图片路径
    """
    img = Image.open(image_path)
    
    # 随机选择增强方式
    augment_type = random.choice(['brightness', 'contrast', 'rotate', 'flip', 'blur', 'color'])
    
    if augment_type == 'brightness':
        # 亮度调整
        enhancer = ImageEnhance.Brightness(img)
        factor = random.uniform(0.7, 1.3)
        img = enhancer.enhance(factor)
    
    elif augment_type == 'contrast':
        # 对比度调整
        enhancer = ImageEnhance.Contrast(img)
        factor = random.uniform(0.7, 1.3)
        img = enhancer.enhance(factor)
    
    elif augment_type == 'rotate':
        # 轻微旋转
        angle = random.uniform(-5, 5)
        img = img.rotate(angle, fillcolor=(255, 255, 255))
    
    elif augment_type == 'flip':
        # 水平翻转
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    elif augment_type == 'blur':
        # 轻微模糊
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    
    elif augment_type == 'color':
        # 色彩调整
        enhancer = ImageEnhance.Color(img)
        factor = random.uniform(0.8, 1.2)
        img = enhancer.enhance(factor)
    
    # 保存增强后的图片
    output_path = output_dir / f"{base_name}_aug_{index}.png"
    img.save(output_path)
    
    return output_path


def augment_dataset(mode='medium', auto_confirm=False):
    """增强页面分类器数据集
    
    Args:
        mode: 增强模式 ('light'=轻度, 'medium'=中度, 'heavy'=重度)
        auto_confirm: 是否自动确认（GUI模式下使用）
    """
    print("\n" + "=" * 80)
    print("🎨 页面分类器数据增强")
    print("=" * 80)
    
    # 获取training_data目录
    script_dir = Path(__file__).parent.parent
    training_data_dir = script_dir / "training_data"
    
    if not training_data_dir.exists():
        print(f"\n❌ 错误: training_data目录不存在: {training_data_dir}")
        return
    
    print(f"\n📁 数据目录: {training_data_dir}")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 显示增强模式
    mode_info = {
        'light': '轻度 (目标: 200-300张)',
        'medium': '中度 (目标: 500-800张)',
        'heavy': '重度 (目标: 1000张左右)'
    }
    print(f"⚙️  增强模式: {mode_info.get(mode, mode_info['medium'])}")
    
    # 扫描所有类别
    categories = []
    for category_dir in sorted(training_data_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        
        # 统计原始图片数量（排除增强图片）
        images = [f for f in category_dir.glob("*.png") 
                 if not f.stem.endswith(('_aug', '_augmented')) and not f.name.startswith('aug_')]
        
        if images:
            categories.append({
                'name': category_dir.name,
                'dir': category_dir,
                'images': images,
                'count': len(images)
            })
    
    if not categories:
        print("\n❌ 错误: 没有找到任何训练数据")
        return
    
    print(f"\n📊 找到 {len(categories)} 个类别:")
    print("-" * 80)
    
    # 计算增强数量并显示统计信息
    total_original = 0
    total_augmented = 0
    
    for cat in categories:
        augment_count = calculate_augment_count(cat['count'], mode)
        cat['augment_count'] = augment_count
        cat['target_total'] = cat['count'] + augment_count
        
        total_original += cat['count']
        total_augmented += augment_count
        
        print(f"  {cat['name']:30s} | 原始: {cat['count']:3d}张 | 生成: {augment_count:4d}张 | 总计: {cat['target_total']:4d}张")
    
    print("-" * 80)
    print(f"  {'总计':30s} | 原始: {total_original:3d}张 | 生成: {total_augmented:4d}张 | 总计: {total_original + total_augmented:4d}张")
    print("-" * 80)
    
    # 确认开始增强
    print(f"\n🎯 增强方法: 亮度、对比度、旋转、翻转、模糊、色彩")
    
    if not auto_confirm:
        input("\n按 Enter 键开始增强，或 Ctrl+C 取消...")
    else:
        print("\n🚀 自动模式：开始增强...")
    
    # 开始增强
    print("\n" + "=" * 80)
    print("🚀 开始数据增强...")
    print("=" * 80)
    
    start_time = datetime.now()
    total_processed = 0
    total_to_process = total_augmented
    
    for cat_idx, cat in enumerate(categories, 1):
        if cat['augment_count'] == 0:
            print(f"\n[{cat_idx}/{len(categories)}] ⏭️  跳过类别: {cat['name']} (已达到目标数量)")
            continue
        
        print(f"\n[{cat_idx}/{len(categories)}] 📦 处理类别: {cat['name']}")
        print(f"  原始图片: {cat['count']}张")
        print(f"  需要生成: {cat['augment_count']}张")
        print(f"  目标总数: {cat['target_total']}张")
        
        # 计算每张原始图片需要生成多少增强图片
        images_per_original = cat['augment_count'] // cat['count']
        extra_images = cat['augment_count'] % cat['count']
        
        aug_counter = 0
        
        # 准备所有增强任务
        tasks = []
        for img_idx, img_path in enumerate(cat['images'], 1):
            base_name = img_path.stem
            
            # 计算这张图片需要生成多少增强图片
            num_augments = images_per_original
            if img_idx <= extra_images:
                num_augments += 1
            
            # 为每个增强操作创建任务
            for aug_idx in range(num_augments):
                tasks.append((img_path, cat['dir'], base_name, aug_counter + 1))
                aug_counter += 1
        
        # 使用16线程并行处理
        with ThreadPoolExecutor(max_workers=16) as executor:
            # 提交所有任务
            futures = {executor.submit(augment_image, *task): task for task in tasks}
            
            # 处理完成的任务
            for future in as_completed(futures):
                try:
                    future.result()
                    
                    # 线程安全地更新进度
                    with progress_lock:
                        total_processed += 1
                        current_progress = total_processed
                    
                    # 只在每10张或完成时显示进度
                    if current_progress % 10 == 0 or current_progress == total_to_process:
                        progress = (current_progress / total_to_process) * 100 if total_to_process > 0 else 100
                        bar_length = min(40, int(progress / 2.5))
                        print(f"\r  进度: [{current_progress - (total_processed - aug_counter)}/{cat['augment_count']}] "
                              f"总进度: {current_progress}/{total_to_process} ({progress:.1f}%) "
                              f"[{'█' * bar_length}{' ' * (40 - bar_length)}]", end='', flush=True)
                    
                except Exception as e:
                    task = futures[future]
                    print(f"\n  ⚠️  警告: 增强失败 {task[0].name}: {e}")
        
        print(f"\n  ✓ 完成: 生成了 {aug_counter}张增强图片，总计 {cat['target_total']}张")
    
    # 完成统计
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 80)
    print("✅ 数据增强完成!")
    print("=" * 80)
    print(f"\n📊 统计信息:")
    print(f"  • 处理类别: {len(categories)}个")
    print(f"  • 原始图片: {total_original}张")
    print(f"  • 生成图片: {total_augmented}张")
    print(f"  • 总计图片: {total_original + total_augmented}张")
    print(f"  • 耗时: {duration:.1f}秒")
    print(f"  • 平均速度: {total_augmented / duration:.1f}张/秒")
    print(f"\n⏰ 完成时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    print("\n💡 提示: 训练完成后，增强的图片会自动删除")


if __name__ == '__main__':
    try:
        # 检查命令行参数
        mode = 'medium'  # 默认中度
        if len(sys.argv) > 1:
            mode_arg = sys.argv[1].lower()
            if mode_arg in ['light', 'medium', 'heavy', '1', '2', '3']:
                if mode_arg == '1' or mode_arg == 'light':
                    mode = 'light'
                elif mode_arg == '2' or mode_arg == 'medium':
                    mode = 'medium'
                elif mode_arg == '3' or mode_arg == 'heavy':
                    mode = 'heavy'
        
        augment_dataset(mode)
    except KeyboardInterrupt:
        print("\n\n⚠️  用户取消操作")
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
