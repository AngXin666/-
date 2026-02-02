"""清理余额积分数据，只保留有4个元素的图片"""
import json
from pathlib import Path
import os

def clean_balance_data():
    """删除只有2个元素的图片，只保留有4个元素的"""
    
    data_dir = Path('training_data/个人页_已登录_余额积分')
    ann_file = data_dir / 'annotations.json'
    
    # 读取标注
    with open(ann_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("=" * 60)
    print("清理余额积分数据")
    print("=" * 60)
    
    # 筛选只有4个元素的图片
    filtered = {}
    deleted_images = []
    
    for img_path, anns in data.items():
        if len(anns) == 4:
            # 保留有4个元素的
            filtered[img_path] = anns
        else:
            # 删除图片文件
            img_file = Path(img_path)
            if img_file.exists():
                os.remove(img_file)
                deleted_images.append(img_file.name)
    
    print(f"\n原始图片数: {len(data)}")
    print(f"保留图片数: {len(filtered)}")
    print(f"删除图片数: {len(deleted_images)}")
    
    # 统计类别
    from collections import Counter
    all_classes = []
    for anns in filtered.values():
        for ann in anns:
            all_classes.append(ann.get('label') or ann.get('class'))
    
    counts = Counter(all_classes)
    print("\n保留数据的类别统计:")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    
    # 保存新的标注文件
    with open(ann_file, 'w', encoding='utf-8') as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 清理完成！")
    print(f"保留的图片数: {len(filtered)}")
    print(f"每张图片都有4个元素（余额、积分、抵扣劵、优惠劵）")

if __name__ == "__main__":
    clean_balance_data()
