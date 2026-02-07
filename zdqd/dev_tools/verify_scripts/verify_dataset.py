"""
验证数据集是否正确
"""
from pathlib import Path
from PIL import Image

def verify_dataset():
    """验证数据集"""
    print("=" * 60)
    print("验证转账页数据集")
    print("=" * 60)
    
    # 检查训练集
    train_img_dir = Path("yolo_dataset_transfer/images/train")
    train_label_dir = Path("yolo_dataset_transfer/labels/train")
    
    train_images = list(train_img_dir.glob("*.png"))
    train_labels = list(train_label_dir.glob("*.txt"))
    
    print(f"\n训练集:")
    print(f"  图片数量: {len(train_images)}")
    print(f"  标签数量: {len(train_labels)}")
    
    # 检查图片和标签是否匹配
    mismatched = []
    for img_path in train_images[:10]:  # 只检查前10个
        label_path = train_label_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            mismatched.append(img_path.name)
        else:
            # 检查标签文件是否为空
            with open(label_path, 'r') as f:
                content = f.read().strip()
                if not content:
                    print(f"  警告: {label_path.name} 是空文件")
                else:
                    lines = content.split('\n')
                    print(f"  {img_path.name}: {len(lines)} 个标注")
                    
                    # 检查图片尺寸
                    img = Image.open(img_path)
                    print(f"    图片尺寸: {img.size}")
                    
                    # 检查标签格式
                    for line in lines[:2]:  # 只显示前2个
                        parts = line.split()
                        if len(parts) == 5:
                            cls, x, y, w, h = parts
                            print(f"    类别{cls}: 中心({x}, {y}), 尺寸({w}, {h})")
    
    if mismatched:
        print(f"\n  不匹配的图片: {len(mismatched)}")
        for name in mismatched[:5]:
            print(f"    - {name}")
    else:
        print(f"\n  ✓ 所有图片都有对应的标签文件")
    
    # 检查验证集
    val_img_dir = Path("yolo_dataset_transfer/images/val")
    val_label_dir = Path("yolo_dataset_transfer/labels/val")
    
    val_images = list(val_img_dir.glob("*.png"))
    val_labels = list(val_label_dir.glob("*.txt"))
    
    print(f"\n验证集:")
    print(f"  图片数量: {len(val_images)}")
    print(f"  标签数量: {len(val_labels)}")
    
    print("\n" + "=" * 60)
    print("验证完成！")
    print("=" * 60)

if __name__ == "__main__":
    verify_dataset()
