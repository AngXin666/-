"""
检查页面分类器数据集数量
"""
from pathlib import Path

def check_dataset():
    """检查数据集"""
    print("=" * 60)
    print("页面分类器数据集统计")
    print("=" * 60)
    
    data_dir = Path('page_classifier_dataset')
    
    if not data_dir.exists():
        print(f"\n❌ 数据集目录不存在: {data_dir}")
        return
    
    all_classes = {}
    
    print("\n各类别图片数量:")
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        
        png_count = len(list(class_dir.glob('*.png')))
        if png_count > 0:
            all_classes[class_dir.name] = png_count
            status = "✓" if png_count >= 20 else "⚠️"
            print(f"  {status} {class_dir.name}: {png_count} 张")
    
    print("\n" + "=" * 60)
    print(f"总计: {sum(all_classes.values())} 张图片，{len(all_classes)} 个类别")
    print("=" * 60)
    
    # 检查数据不足的类别
    insufficient = {cls: count for cls, count in all_classes.items() if count < 20}
    sufficient = {cls: count for cls, count in all_classes.items() if count >= 20}
    
    if sufficient:
        print(f"\n✓ {len(sufficient)} 个类别数据充足（≥20张）")
    
    if insufficient:
        print(f"\n⚠️ {len(insufficient)} 个类别数据不足（<20张）:")
        for cls, count in sorted(insufficient.items(), key=lambda x: x[1]):
            need = 20 - count
            print(f"  - {cls}: {count} 张（还需 {need} 张）")
    else:
        print("\n✓ 所有类别数据都充足！可以开始训练。")
    
    return all_classes

if __name__ == '__main__':
    check_dataset()
