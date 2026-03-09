"""
删除所有类别中的增强图片
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

def delete_augmented_images(category_path):
    """删除脚本生成的增强图片（文件名包含 _aug）"""
    deleted_count = 0
    for file in category_path.iterdir():
        if '_aug' in file.stem:
            file.unlink()
            deleted_count += 1
    return deleted_count

def main():
    print("=" * 60)
    print("删除所有类别中的增强图片")
    print("=" * 60)
    
    categories = sorted([d for d in TRAINING_DATA_DIR.iterdir() if d.is_dir()])
    
    # 删除所有类别的增强图片
    deleted_total = 0
    for category_dir in categories:
        before_count = count_images_in_category(category_dir)
        deleted = delete_augmented_images(category_dir)
        after_count = count_images_in_category(category_dir)
        
        if deleted > 0:
            deleted_total += deleted
            print(f"  {category_dir.name}: {before_count}张 → {after_count}张 (删除 {deleted} 张)")
    
    print(f"\n共删除 {deleted_total} 张增强图片")
    
    print("\n" + "=" * 60)
    print("删除后的统计")
    print("=" * 60)
    
    total_images = 0
    for category_dir in categories:
        count = count_images_in_category(category_dir)
        total_images += count
        print(f"  {category_dir.name}: {count}张")
    
    print(f"\n总计: {len(categories)} 个类别, {total_images} 张图片")

if __name__ == "__main__":
    main()
