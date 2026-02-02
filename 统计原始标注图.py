"""
统计原始标注图文件夹
"""

from pathlib import Path

def count_files():
    """统计原始标注图"""
    print("=" * 80)
    print("原始标注图统计")
    print("=" * 80)
    
    base_dir = Path("原始标注图")
    
    if not base_dir.exists():
        print("⚠ 原始标注图文件夹不存在")
        return
    
    total_images = 0
    total_labels = 0
    
    folders = sorted([f for f in base_dir.iterdir() if f.is_dir()])
    
    for folder in folders:
        images_dir = folder / "images"
        labels_dir = folder / "labels"
        
        num_images = 0
        num_labels = 0
        
        if images_dir.exists():
            num_images = len(list(images_dir.glob("*.png"))) + len(list(images_dir.glob("*.jpg")))
        
        if labels_dir.exists():
            num_labels = len(list(labels_dir.glob("*.txt")))
        
        print(f"\n{folder.name}:")
        print(f"  图片: {num_images} 张")
        print(f"  标签: {num_labels} 个")
        
        total_images += num_images
        total_labels += num_labels
    
    print("\n" + "=" * 80)
    print(f"总计: {len(folders)} 个模型")
    print(f"图片: {total_images} 张")
    print(f"标签: {total_labels} 个")
    print("=" * 80)

if __name__ == '__main__':
    count_files()
