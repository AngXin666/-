"""
准备完整的页面分类器数据集
- 保留原有的page_classifier_dataset中的其他类别
- 替换4个类别为新的YOLO标注图（首页、签到页、温馨提示、签到弹窗）

用法：
    python prepare_full_classifier_dataset.py
"""
import shutil
from pathlib import Path


def prepare_full_dataset():
    """准备完整的页面分类器数据集"""
    print("=" * 60)
    print("准备完整页面分类器数据集")
    print("=" * 60)
    
    # 源目录
    old_dataset = Path("page_classifier_dataset")
    new_dataset = Path("page_classifier_dataset_updated")
    
    # 要替换的4个类别及其新数据源
    replace_mappings = {
        "首页": "原始标注图/首页_20260130_030231/images",
        "签到页": "原始标注图/签到页_20260130_014729/images",
        "温馨提示": "原始标注图/温馨提示_20260130_015651/images",
        "签到弹窗": "原始标注图/签到成功弹窗_20260130_013633/images",
    }
    
    # 删除旧的更新数据集
    if new_dataset.exists():
        print(f"\n🗑️  删除旧的数据集...")
        shutil.rmtree(new_dataset)
    
    print(f"\n📂 源目录: {old_dataset}")
    print(f"📂 目标目录: {new_dataset}")
    
    # 复制所有原有类别
    print(f"\n📦 复制原有类别...")
    copied_classes = []
    replaced_classes = []
    
    for class_dir in sorted(old_dataset.iterdir()):
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        target_dir = new_dataset / class_name
        
        # 检查是否需要替换
        if class_name in replace_mappings:
            # 使用新数据
            new_source = Path(replace_mappings[class_name])
            if new_source.exists():
                target_dir.mkdir(parents=True, exist_ok=True)
                images = list(new_source.glob("*.png")) + list(new_source.glob("*.jpg"))
                for img in images:
                    shutil.copy2(img, target_dir / img.name)
                print(f"  ✓ {class_name}: 已替换 ({len(images)}张新图片)")
                replaced_classes.append((class_name, len(images)))
            else:
                print(f"  ✗ {class_name}: 新数据源不存在，跳过")
        else:
            # 保留原有数据
            shutil.copytree(class_dir, target_dir)
            count = len(list(target_dir.glob("*.png")) + list(target_dir.glob("*.jpg")))
            print(f"  ✓ {class_name}: 已保留 ({count}张)")
            copied_classes.append((class_name, count))
    
    print(f"\n✅ 数据集准备完成!")
    print(f"  位置: {new_dataset}")
    
    # 统计
    print(f"\n📊 数据集统计:")
    print(f"  保留的类别: {len(copied_classes)}个")
    print(f"  替换的类别: {len(replaced_classes)}个")
    
    total_images = 0
    for class_dir in sorted(new_dataset.iterdir()):
        if class_dir.is_dir():
            count = len(list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg")))
            total_images += count
            status = "🆕" if class_dir.name in replace_mappings else "  "
            print(f"  {status} {class_dir.name}: {count}张")
    
    print(f"\n  总计: {total_images}张图片")
    
    print(f"\n🎯 下一步:")
    print(f"  训练模型: python train_page_classifier_pytorch.py")


if __name__ == "__main__":
    prepare_full_dataset()
