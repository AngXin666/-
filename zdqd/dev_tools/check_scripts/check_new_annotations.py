"""检查新标注的数据"""
import json
from pathlib import Path
from collections import Counter

def check_annotations(data_dir):
    """检查标注数据"""
    data_path = Path(data_dir)
    ann_file = data_path / "annotations.json"
    
    if not ann_file.exists():
        print(f"❌ 标注文件不存在: {ann_file}")
        return
    
    with open(ann_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("=" * 80)
    print(f"检查标注数据: {data_dir}")
    print("=" * 80)
    
    # 统计图片数量
    print(f"\n总图片数: {len(data)}")
    
    # 统计各类别数量
    all_classes = []
    for img_path, anns in data.items():
        for ann in anns:
            label = ann.get('label') or ann.get('class')
            all_classes.append(label)
    
    counts = Counter(all_classes)
    print(f"\n各类别标注数:")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    
    print(f"\n总标注数: {sum(counts.values())}")
    
    # 检查每张图片的标注数量分布
    ann_counts = [len(anns) for anns in data.values()]
    ann_count_dist = Counter(ann_counts)
    print(f"\n每张图片的标注数量分布:")
    for count, num_imgs in sorted(ann_count_dist.items()):
        print(f"  {count}个标注: {num_imgs}张图片")
    
    # 显示几个示例
    print(f"\n标注示例（前3张）:")
    for i, (img_path, anns) in enumerate(list(data.items())[:3]):
        img_name = Path(img_path).name
        print(f"\n  图片 {i+1}: {img_name}")
        print(f"    标注数: {len(anns)}")
        for ann in anns:
            label = ann.get('label') or ann.get('class')
            if 'x1' in ann:
                x1, y1, x2, y2 = ann['x1'], ann['y1'], ann['x2'], ann['y2']
            else:
                bbox = ann['bbox']
                x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            print(f"      - {label}: ({x1:.0f}, {y1:.0f}) -> ({x2:.0f}, {y2:.0f})")

if __name__ == "__main__":
    print("检查个人页_已登录标注数据:\n")
    check_annotations("training_data/个人页_已登录")
    
    print("\n\n检查首页标注数据:\n")
    check_annotations("training_data/首页")
