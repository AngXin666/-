"""
准备页面分类器训练数据
- 整合 training_data 和 training_data_completed 目录
- 排除增强数据文件夹（_temp_augmented）
- 复制到统一的 page_classifier_dataset 目录
"""
from pathlib import Path
import shutil

def prepare_dataset():
    """准备数据集"""
    print("=" * 60)
    print("准备页面分类器训练数据")
    print("=" * 60)
    
    # 源目录
    source_dirs = [
        Path('training_data'),
        Path('training_data_completed')
    ]
    
    # 目标目录
    target_dir = Path('page_classifier_dataset')
    
    # 删除旧的目标目录
    if target_dir.exists():
        print(f"\n删除旧的数据集目录: {target_dir}")
        shutil.rmtree(target_dir)
    
    # 创建新目录
    target_dir.mkdir(exist_ok=True)
    print(f"创建数据集目录: {target_dir}")
    
    # 统计信息
    total_images = 0
    total_classes = 0
    class_stats = {}
    
    # 遍历源目录
    for source_dir in source_dirs:
        if not source_dir.exists():
            continue
        
        print(f"\n处理目录: {source_dir}")
        
        for class_dir in sorted(source_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            
            # 跳过增强数据文件夹
            if '_temp_augmented' in class_name or '_augmented' in class_name:
                print(f"  跳过增强数据: {class_name}")
                continue
            
            # 跳过特殊文件夹
            if class_name.startswith('.') or class_name == '__pycache__':
                continue
            
            # 统计图片数量
            images = list(class_dir.glob('*.png'))
            if len(images) == 0:
                print(f"  跳过空文件夹: {class_name}")
                continue
            
            # 创建目标类别目录
            target_class_dir = target_dir / class_name
            if not target_class_dir.exists():
                target_class_dir.mkdir(parents=True)
                class_stats[class_name] = 0
                total_classes += 1
            
            # 复制图片
            copied = 0
            for img in images:
                # 生成唯一的目标文件名（避免重复）
                target_file = target_class_dir / f"{source_dir.name}_{img.name}"
                if not target_file.exists():
                    shutil.copy2(img, target_file)
                    copied += 1
            
            class_stats[class_name] += copied
            total_images += copied
            print(f"  ✓ {class_name}: 复制 {copied} 张图片")
    
    # 输出统计信息
    print("\n" + "=" * 60)
    print("数据集准备完成")
    print("=" * 60)
    print(f"\n总计: {total_images} 张图片，{total_classes} 个类别")
    print("\n各类别图片数量:")
    for class_name in sorted(class_stats.keys()):
        count = class_stats[class_name]
        status = "✓" if count >= 20 else "⚠️"
        print(f"  {status} {class_name}: {count} 张")
    
    # 检查数据不足的类别
    insufficient = [cls for cls, count in class_stats.items() if count < 20]
    if insufficient:
        print(f"\n⚠️ {len(insufficient)} 个类别图片数量不足20张:")
        for cls in insufficient:
            print(f"  - {cls}: {class_stats[cls]} 张")
        print("\n建议:")
        print("  1. 收集更多这些类别的截图")
        print("  2. 或者在训练时排除这些类别")
    
    print(f"\n数据集目录: {target_dir}")
    print("\n下一步:")
    print("  python train_page_classifier.py")
    
    return class_stats

if __name__ == '__main__':
    prepare_dataset()
