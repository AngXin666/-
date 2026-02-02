"""
从YOLO原始标注图准备页面分类器训练数据

用法：
    python prepare_page_classifier_from_yolo.py
"""
import shutil
from pathlib import Path


def prepare_page_classifier_data():
    """从YOLO原始标注图准备页面分类器训练数据（只准备指定的4个类别）"""
    print("=" * 60)
    print("准备页面分类器训练数据（仅4个类别）")
    print("=" * 60)
    
    # 定义源目录和目标目录的映射
    # 格式：(源目录, 目标页面类型名称)
    mappings = [
        ("原始标注图/首页_20260130_030231/images", "首页"),
        ("原始标注图/签到页_20260130_014729/images", "签到页"),
        ("原始标注图/温馨提示_20260130_015651/images", "温馨提示"),
        ("原始标注图/签到成功弹窗_20260130_013633/images", "签到弹窗"),  # 签到成功弹窗 -> 签到弹窗
    ]
    
    # 创建新的目标根目录（只包含这4个类别）
    target_root = Path("page_classifier_dataset_4classes")
    
    # 如果目录已存在，先删除
    if target_root.exists():
        print(f"\n🗑️  删除旧的数据集目录...")
        shutil.rmtree(target_root)
    
    print(f"\n📂 目标目录: {target_root}")
    print(f"\n🔍 检查源目录...")
    
    # 检查所有源目录是否存在
    valid_mappings = []
    for source_dir, page_type in mappings:
        source_path = Path(source_dir)
        if source_path.exists():
            image_count = len(list(source_path.glob("*.png")) + list(source_path.glob("*.jpg")))
            print(f"  ✓ {page_type}: {source_dir} ({image_count}张图片)")
            valid_mappings.append((source_path, page_type, image_count))
        else:
            print(f"  ✗ {page_type}: {source_dir} (不存在)")
    
    if not valid_mappings:
        print(f"\n❌ 没有找到有效的源目录")
        return
    
    print(f"\n📊 找到 {len(valid_mappings)} 个有效的页面类型")
    
    # 询问是否继续
    total_images = sum(count for _, _, count in valid_mappings)
    print(f"\n将复制 {total_images} 张图片到新的页面分类器数据集")
    print(f"目标目录: {target_root}")
    print(f"⚠️  注意：只包含这4个类别，不影响原有的 page_classifier_dataset")
    
    # 复制图片
    print(f"\n📦 开始复制图片...")
    copied_count = 0
    
    for source_path, page_type, image_count in valid_mappings:
        # 创建目标目录
        target_dir = target_root / page_type
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制所有图片
        images = list(source_path.glob("*.png")) + list(source_path.glob("*.jpg"))
        
        for img_path in images:
            target_path = target_dir / img_path.name
            shutil.copy2(img_path, target_path)
            copied_count += 1
        
        print(f"  ✓ {page_type}: 已复制 {len(images)} 张图片")
    
    print(f"\n✅ 复制完成!")
    print(f"  总计: {copied_count} 张图片")
    print(f"  位置: {target_root}")
    
    # 统计每个类别的图片数量
    print(f"\n📊 数据集统计（仅4个类别）:")
    for page_type_dir in sorted(target_root.iterdir()):
        if page_type_dir.is_dir():
            count = len(list(page_type_dir.glob("*.png")) + list(page_type_dir.glob("*.jpg")))
            print(f"  {page_type_dir.name}: {count}张")
    
    print(f"\n🎯 下一步:")
    print(f"  1. 检查数据集: 打开 {target_root} 查看图片")
    print(f"  2. 训练模型: python train_page_classifier_pytorch.py --dataset {target_root}")


if __name__ == "__main__":
    prepare_page_classifier_data()
