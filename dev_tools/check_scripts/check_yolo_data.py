"""检查 YOLO 训练数据准备情况"""
from pathlib import Path

def check_training_data():
    """检查训练数据"""
    training_data = Path("training_data")
    
    if not training_data.exists():
        print("❌ training_data 目录不存在")
        return
    
    total_annotated = 0
    categories = []
    
    for category_dir in sorted(training_data.iterdir()):
        if not category_dir.is_dir():
            continue
        
        # 统计图片和标注
        images = list(category_dir.glob("*.png"))
        labels = list(category_dir.glob("*.txt"))
        
        # 检查非空标注文件
        annotated = [l for l in labels if l.stat().st_size > 0]
        
        if len(annotated) > 0:
            categories.append({
                'name': category_dir.name,
                'images': len(images),
                'annotated': len(annotated)
            })
            total_annotated += len(annotated)
    
    if total_annotated == 0:
        print("❌ 没有找到已标注的数据")
        print("\n请先运行标注工具:")
        print("  python annotation_tool.py")
        return
    
    print("=" * 60)
    print("YOLO 训练数据统计")
    print("=" * 60)
    print(f"\n总计: {total_annotated} 张已标注图片\n")
    
    print("按页面类别统计:")
    for cat in categories:
        print(f"  {cat['name']}: {cat['annotated']} 张")
    
    print("\n" + "=" * 60)
    
    if total_annotated < 20:
        print("⚠️  数据量较少，建议至少标注 20 张图片")
        print("   训练效果可能不理想")
    elif total_annotated < 50:
        print("✓ 数据量基本够用，可以开始训练")
        print("  建议继续标注更多数据以提高准确率")
    else:
        print("✓ 数据量充足，可以开始训练")
    
    return total_annotated

if __name__ == "__main__":
    check_training_data()
