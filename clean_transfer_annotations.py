"""
清理转账页标注数据 - 只保留完整标注的图片
"""
import json
from pathlib import Path
import shutil

def clean_transfer_annotations():
    """清理转账页标注数据，只保留完整标注"""
    
    print("=" * 80)
    print("清理转账页标注数据")
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
    
    print(f"\n原始数据:")
    print(f"  总图片数: {len(annotations)}")
    
    # 统计标注情况
    complete_annotations = {}
    incomplete_images = []
    empty_images = []
    
    # 定义完整标注的标准（转账页应该有5个元素）
    required_classes = {
        "全部转账按钮",
        "ID输入框", 
        "转账金额输入框",
        "提交按钮",
        "转账明细文本"
    }
    
    for img_name, img_annotations in annotations.items():
        if len(img_annotations) == 0:
            empty_images.append(img_name)
        elif len(img_annotations) >= 4:  # 至少有4个标注（提交按钮可能缺失）
            # 检查是否包含必要的类别
            annotated_classes = {ann['class'] for ann in img_annotations}
            # 至少要有输入框和按钮
            if "ID输入框" in annotated_classes and "转账金额输入框" in annotated_classes:
                complete_annotations[img_name] = img_annotations
            else:
                incomplete_images.append((img_name, len(img_annotations)))
        else:
            incomplete_images.append((img_name, len(img_annotations)))
    
    print(f"\n数据分析:")
    print(f"  完整标注: {len(complete_annotations)} 张")
    print(f"  不完整标注: {len(incomplete_images)} 张")
    print(f"  未标注: {len(empty_images)} 张")
    
    # 显示将被移除的图片
    if incomplete_images or empty_images:
        print(f"\n将移除的图片:")
        
        if incomplete_images:
            print(f"\n  不完整标注 ({len(incomplete_images)} 张):")
            for img_name, count in incomplete_images[:5]:
                print(f"    {Path(img_name).name}: {count} 个标注")
            if len(incomplete_images) > 5:
                print(f"    ... 还有 {len(incomplete_images) - 5} 张")
        
        if empty_images:
            print(f"\n  未标注 ({len(empty_images)} 张):")
            for img_name in empty_images[:5]:
                print(f"    {Path(img_name).name}")
            if len(empty_images) > 5:
                print(f"    ... 还有 {len(empty_images) - 5} 张")
    
    # 自动执行清理
    print(f"\n" + "=" * 80)
    print(f"将保留 {len(complete_annotations)} 张完整标注的图片")
    print(f"将移除 {len(incomplete_images) + len(empty_images)} 张不完整/未标注的图片")
    print("=" * 80)
    print("\n开始清理...")
    
    # 备份原始标注文件
    backup_file = annotation_file.parent / "annotations_backup.json"
    shutil.copy2(annotation_file, backup_file)
    print(f"\n✓ 已备份原始标注文件: {backup_file}")
    
    # 保存清理后的标注
    with open(annotation_file, 'w', encoding='utf-8') as f:
        json.dump(complete_annotations, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 已保存清理后的标注文件: {annotation_file}")
    
    # 移动不完整的图片到单独文件夹
    incomplete_dir = data_dir / "incomplete_annotations"
    incomplete_dir.mkdir(exist_ok=True)
    
    moved_count = 0
    for img_name in incomplete_images + [(img, 0) for img in empty_images]:
        if isinstance(img_name, tuple):
            img_name = img_name[0]
        
        img_path = Path(img_name)
        if img_path.exists():
            dest_path = incomplete_dir / img_path.name
            shutil.move(str(img_path), str(dest_path))
            moved_count += 1
    
    print(f"✓ 已移动 {moved_count} 张不完整图片到: {incomplete_dir}")
    
    # 显示最终统计
    print(f"\n" + "=" * 80)
    print("清理完成！")
    print("=" * 80)
    print(f"\n最终数据:")
    print(f"  保留图片: {len(complete_annotations)} 张")
    print(f"  移除图片: {moved_count} 张")
    print(f"  备份文件: {backup_file}")
    print(f"  不完整图片目录: {incomplete_dir}")
    
    return {
        'complete': len(complete_annotations),
        'removed': moved_count,
        'backup': str(backup_file)
    }

if __name__ == "__main__":
    clean_transfer_annotations()
