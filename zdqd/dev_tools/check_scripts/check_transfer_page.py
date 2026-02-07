"""
检查转账页标注数据
"""
import json
from pathlib import Path
from collections import Counter

def check_transfer_page():
    """检查转账页标注数据"""
    
    print("=" * 80)
    print("检查转账页标注数据")
    print("=" * 80)
    
    # 数据目录
    data_dir = Path("training_data/转账页")
    annotation_file = data_dir / "annotations.json"
    
    if not annotation_file.exists():
        print(f"❌ 标注文件不存在: {annotation_file}")
        return
    
    # 读取标注
    with open(annotation_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    print(f"\n总图片数: {len(annotations)}")
    
    # 统计标注情况
    annotated_count = 0
    unannotated_count = 0
    class_counter = Counter()
    annotation_counter = Counter()
    
    for img_name, img_annotations in annotations.items():
        if len(img_annotations) > 0:
            annotated_count += 1
            annotation_counter[len(img_annotations)] += 1
            for ann in img_annotations:
                class_counter[ann['class']] += 1
        else:
            unannotated_count += 1
    
    print(f"已标注: {annotated_count} 张")
    print(f"未标注: {unannotated_count} 张")
    
    # 显示类别统计
    print(f"\n类别统计:")
    for class_name, count in class_counter.most_common():
        print(f"  {class_name}: {count} 个")
    
    # 显示每张图片的标注数量分布
    print(f"\n每张图片的标注数量分布:")
    for count, num_images in sorted(annotation_counter.items()):
        print(f"  {count} 个标注: {num_images} 张图片")
    
    # 检查是否有不完整的标注
    print(f"\n检查不完整标注...")
    incomplete_images = []
    for img_name, img_annotations in annotations.items():
        if 0 < len(img_annotations) < 4:  # 应该有4个标注
            incomplete_images.append((img_name, len(img_annotations)))
    
    if incomplete_images:
        print(f"⚠️ 发现 {len(incomplete_images)} 张图片标注不完整:")
        for img_name, count in incomplete_images[:10]:  # 只显示前10个
            print(f"  {Path(img_name).name}: {count} 个标注")
        if len(incomplete_images) > 10:
            print(f"  ... 还有 {len(incomplete_images) - 10} 张")
    else:
        print("✓ 所有已标注图片都有完整标注")
    
    print("\n" + "=" * 80)
    print("检查完成！")
    print("=" * 80)
    
    # 返回统计信息
    return {
        'total': len(annotations),
        'annotated': annotated_count,
        'unannotated': unannotated_count,
        'classes': dict(class_counter),
        'incomplete': len(incomplete_images)
    }

if __name__ == "__main__":
    check_transfer_page()
