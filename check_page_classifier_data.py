"""
检查页面分类器训练数据
"""
from pathlib import Path

def check_data():
    """检查训练数据"""
    print("=" * 60)
    print("页面分类器训练数据统计")
    print("=" * 60)
    
    # 检查 training_data 目录
    data_dir = Path('training_data')
    completed_dir = Path('training_data_completed')
    
    all_classes = {}
    
    print("\n=== training_data 目录 ===")
    if data_dir.exists():
        for d in sorted(data_dir.iterdir()):
            if d.is_dir():
                png_count = len(list(d.glob('*.png')))
                if png_count > 0:
                    all_classes[d.name] = png_count
                    print(f"  {d.name}: {png_count} 张")
    
    print("\n=== training_data_completed 目录 ===")
    if completed_dir.exists():
        for d in sorted(completed_dir.iterdir()):
            if d.is_dir():
                png_count = len(list(d.glob('*.png')))
                if png_count > 0:
                    # 如果类别已存在，累加数量
                    if d.name in all_classes:
                        all_classes[d.name] += png_count
                        print(f"  {d.name}: {png_count} 张 (累加)")
                    else:
                        all_classes[d.name] = png_count
                        print(f"  {d.name}: {png_count} 张")
    
    print("\n" + "=" * 60)
    print(f"总计: {sum(all_classes.values())} 张图片，{len(all_classes)} 个类别")
    print("=" * 60)
    
    # 检查是否有足够的数据
    insufficient = [cls for cls, count in all_classes.items() if count < 20]
    if insufficient:
        print("\n⚠️ 以下类别图片数量不足20张（建议至少20张）:")
        for cls in insufficient:
            print(f"  - {cls}: {all_classes[cls]} 张")
    
    return all_classes

if __name__ == '__main__':
    check_data()
