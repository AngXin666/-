"""
检查训练数据集数量和质量
"""
from pathlib import Path
from collections import Counter

def check_dataset():
    """检查数据集"""
    training_data = Path('training_data')
    
    all_images = []
    page_info = []
    
    # 收集所有标注数据
    for category_dir in training_data.iterdir():
        if not category_dir.is_dir():
            continue
        
        page_type = category_dir.name
        
        # 查找所有有标注的图片
        for img_file in category_dir.glob("*.png"):
            label_file = img_file.with_suffix(".txt")
            if label_file.exists():
                all_images.append(img_file)
                page_info.append(page_type)
    
    page_counts = Counter(page_info)
    
    print("=" * 70)
    print("训练数据集统计")
    print("=" * 70)
    print(f"\n总计: {len(all_images)} 张已标注图片\n")
    
    print("各页面标注数量:")
    print("-" * 70)
    
    # 分类统计
    excellent = []  # >= 50
    good = []       # 30-49
    fair = []       # 20-29
    poor = []       # < 20
    
    for page_type, count in sorted(page_counts.items(), key=lambda x: x[1], reverse=True):
        if count >= 50:
            status = "✅ 优秀"
            excellent.append((page_type, count))
        elif count >= 30:
            status = "✅ 良好"
            good.append((page_type, count))
        elif count >= 20:
            status = "⚠️  一般"
            fair.append((page_type, count))
        else:
            status = "❌ 偏少"
            poor.append((page_type, count))
        
        print(f"  {page_type:25s} {count:3d} 张  {status}")
    
    print("\n" + "=" * 70)
    print("数据质量评估")
    print("=" * 70)
    
    print(f"\n✅ 优秀 (≥50张): {len(excellent)} 个页面")
    for page, count in excellent:
        print(f"   - {page}: {count} 张")
    
    print(f"\n✅ 良好 (30-49张): {len(good)} 个页面")
    for page, count in good:
        print(f"   - {page}: {count} 张")
    
    print(f"\n⚠️  一般 (20-29张): {len(fair)} 个页面")
    for page, count in fair:
        print(f"   - {page}: {count} 张")
    
    print(f"\n❌ 偏少 (<20张): {len(poor)} 个页面")
    for page, count in poor:
        print(f"   - {page}: {count} 张")
    
    print("\n" + "=" * 70)
    print("建议")
    print("=" * 70)
    
    total_pages = len(page_counts)
    avg_count = len(all_images) / total_pages if total_pages > 0 else 0
    
    print(f"\n平均每页面: {avg_count:.1f} 张")
    
    if len(poor) > 0:
        print(f"\n⚠️  有 {len(poor)} 个页面标注数量不足 20 张，建议优先增加：")
        for page, count in sorted(poor, key=lambda x: x[1]):
            need = 30 - count
            print(f"   - {page}: 当前 {count} 张，建议再增加 {need} 张")
    
    if len(fair) > 0:
        print(f"\n💡 有 {len(fair)} 个页面标注数量一般，可以考虑增加到 30+ 张")
    
    if len(all_images) < 1000:
        print(f"\n💡 总数据量: {len(all_images)} 张")
        print(f"   建议目标: 1000+ 张（当前 {len(all_images)/1000*100:.1f}%）")
    else:
        print(f"\n✅ 总数据量充足: {len(all_images)} 张")
    
    print("\n" + "=" * 70)
    print("训练建议")
    print("=" * 70)
    
    if len(all_images) >= 1000 and len(poor) == 0:
        print("\n✅ 数据集质量优秀，可以开始训练！")
        print("   预期效果: mAP50 > 50%")
    elif len(all_images) >= 500 and len(poor) <= 3:
        print("\n✅ 数据集质量良好，可以开始训练")
        print("   预期效果: mAP50 30-50%")
    elif len(all_images) >= 300:
        print("\n⚠️  数据集质量一般，可以训练但效果可能不理想")
        print("   预期效果: mAP50 15-30%")
        print("   建议: 增加标注数量后再训练")
    else:
        print("\n❌ 数据集数量不足，建议增加标注后再训练")
        print("   当前预期: mAP50 < 15%")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    check_dataset()
