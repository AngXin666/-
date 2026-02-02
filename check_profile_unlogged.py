"""
检查个人页_未登录标注数据
"""
import json
from pathlib import Path
from collections import Counter

def check_profile_unlogged():
    """检查个人页_未登录标注数据"""
    
    data_dir = Path("training_data/个人页_未登录")
    annotation_file = data_dir / "annotations.json"
    
    print("=" * 80)
    print("检查个人页_未登录标注数据")
    print("=" * 80)
    
    # 统计图片数量
    images = list(data_dir.glob("*.png"))
    print(f"\n总图片数: {len(images)}")
    
    # 读取标注文件
    if not annotation_file.exists():
        print(f"\n❌ 标注文件不存在: {annotation_file}")
        return
    
    with open(annotation_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    # 统计各类别标注数
    class_counter = Counter()
    annotations_per_image = []
    
    for img_name, img_annotations in annotations.items():
        num_annotations = len(img_annotations)
        annotations_per_image.append(num_annotations)
        
        for ann in img_annotations:
            class_name = ann['class']
            class_counter[class_name] += 1
    
    # 打印类别统计
    print(f"\n各类别标注数:")
    for class_name, count in sorted(class_counter.items()):
        print(f"  {class_name}: {count}")
    
    print(f"\n总标注数: {sum(class_counter.values())}")
    
    # 统计每张图片的标注数量分布
    annotation_dist = Counter(annotations_per_image)
    print(f"\n每张图片的标注数量分布:")
    for num_ann, count in sorted(annotation_dist.items()):
        print(f"  {num_ann}个标注: {count}张图片")
    
    # 显示前3张图片的标注示例
    print(f"\n标注示例（前3张）:")
    for i, (img_name, img_annotations) in enumerate(list(annotations.items())[:3]):
        print(f"\n  图片 {i+1}: {img_name}")
        print(f"    标注数: {len(img_annotations)}")
        for ann in img_annotations:
            x1, y1, x2, y2 = ann['x1'], ann['y1'], ann['x2'], ann['y2']
            print(f"      - {ann['class']}: ({x1}, {y1}) -> ({x2}, {y2})")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    check_profile_unlogged()
