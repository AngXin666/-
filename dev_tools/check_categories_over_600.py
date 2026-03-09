"""
检查哪些类别的图片数量大于600张
"""

from pathlib import Path

# 训练数据目录
TRAINING_DATA_DIR = Path("标注工具_完整独立版/training_data")

def count_images_in_category(category_path):
    """统计类别中的图片数量"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    count = 0
    for file in category_path.iterdir():
        if file.suffix.lower() in image_extensions:
            count += 1
    return count

def main():
    categories = sorted([d for d in TRAINING_DATA_DIR.iterdir() if d.is_dir()])
    
    print("=" * 60)
    print("大于600张的类别：")
    print("=" * 60)
    
    over_600 = []
    for category_dir in categories:
        count = count_images_in_category(category_dir)
        if count > 600:
            over_600.append((category_dir.name, count))
            print(f"  {category_dir.name}: {count}张")
    
    if not over_600:
        print("  没有大于600张的类别")
    else:
        print(f"\n共 {len(over_600)} 个类别大于600张")

if __name__ == "__main__":
    main()
