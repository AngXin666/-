"""
检查余额积分标注数据
"""
import json
from pathlib import Path
import random
from PIL import Image

def check_balance_data():
    """检查余额积分标注数据"""
    
    # 读取原始标注
    ann_file = Path("training_data/个人页_已登录_余额积分/annotations.json")
    with open(ann_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("=" * 60)
    print("余额积分标注数据检查")
    print("=" * 60)
    
    print(f"\n总图片数: {len(data)}")
    
    # 随机抽查3张图片
    samples = random.sample(list(data.items()), min(3, len(data)))
    
    print("\n随机抽查图片标注:")
    for img_path, anns in samples:
        img_path = Path(img_path)
        print(f"\n图片: {img_path.name}")
        
        if img_path.exists():
            img = Image.open(img_path)
            print(f"  尺寸: {img.size}")
        
        for ann in anns:
            label = ann.get('label') or ann.get('class')
            if 'x1' in ann:
                x1, y1, x2, y2 = ann['x1'], ann['y1'], ann['x2'], ann['y2']
            else:
                bbox = ann['bbox']
                x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            
            width = x2 - x1
            height = y2 - y1
            print(f"  - {label}: 位置({x1}, {y1}) -> ({x2}, {y2}), 大小({width}x{height})")
    
    # 检查YOLO数据集
    print("\n" + "=" * 60)
    print("YOLO数据集检查")
    print("=" * 60)
    
    train_labels = list(Path("yolo_dataset_balance/labels/train").glob("*.txt"))
    val_labels = list(Path("yolo_dataset_balance/labels/val").glob("*.txt"))
    
    print(f"\n训练集标注文件数: {len(train_labels)}")
    print(f"验证集标注文件数: {len(val_labels)}")
    
    # 随机检查一个训练集标注文件
    if train_labels:
        sample_label = random.choice(train_labels)
        print(f"\n随机抽查训练集标注: {sample_label.name}")
        with open(sample_label, 'r') as f:
            lines = f.readlines()
        print(f"  标注行数: {len(lines)}")
        for line in lines[:3]:  # 只显示前3行
            parts = line.strip().split()
            if len(parts) == 5:
                cls_id, x_center, y_center, width, height = parts
                print(f"  类别{cls_id}: 中心({x_center}, {y_center}), 大小({width}x{height})")

if __name__ == "__main__":
    check_balance_data()
