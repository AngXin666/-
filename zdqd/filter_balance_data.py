"""
筛选余额、积分、抵扣劵、优惠劵标注数据
"""
import json
import shutil
from pathlib import Path
from collections import Counter

def filter_balance_data():
    """筛选只包含余额、积分、抵扣劵、优惠劵的标注数据"""
    
    # 读取原始标注
    source_dir = Path("training_data/个人页_已登录")
    ann_file = source_dir / "annotations.json"
    
    with open(ann_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 目标类别
    target_classes = {'余额数字', '积分数字', '抵扣劵数字', '优惠劵数字'}
    
    # 筛选数据
    filtered = {}
    for img_path, anns in data.items():
        if anns:
            # 获取这张图片的所有类别
            img_classes = {ann.get('label') or ann.get('class') for ann in anns}
            
            # 如果包含目标类别
            if img_classes & target_classes:
                # 只保留目标类别的标注
                new_anns = [ann for ann in anns if (ann.get('label') or ann.get('class')) in target_classes]
                if new_anns:
                    filtered[img_path] = new_anns
    
    print(f"包含目标元素的图片数: {len(filtered)}")
    
    # 统计各类别数量
    classes = []
    for anns in filtered.values():
        for ann in anns:
            classes.append(ann.get('label') or ann.get('class'))
    
    counts = Counter(classes)
    print("\n各类别标注数:")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    
    # 创建新目录
    output_dir = Path("training_data/个人页_已登录_余额积分")
    output_dir.mkdir(exist_ok=True)
    
    # 复制图片
    print(f"\n复制图片到 {output_dir}...")
    for img_path in filtered.keys():
        src = Path(img_path)
        if src.exists():
            dst = output_dir / src.name
            shutil.copy(src, dst)
    
    # 保存标注
    output_ann = output_dir / "annotations.json"
    
    # 更新图片路径
    new_filtered = {}
    for img_path, anns in filtered.items():
        new_path = str(output_dir / Path(img_path).name)
        new_filtered[new_path] = anns
    
    with open(output_ann, 'w', encoding='utf-8') as f:
        json.dump(new_filtered, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 筛选完成！")
    print(f"图片数: {len(filtered)}")
    print(f"保存位置: {output_dir}")
    print(f"标注文件: {output_ann}")

if __name__ == "__main__":
    filter_balance_data()
