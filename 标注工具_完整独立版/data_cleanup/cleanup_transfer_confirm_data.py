"""
清理转账确认弹窗训练数据
1. 删除 YOLO 数据集中的增强图片
2. 将原始标注图移动到 原始标注图/ 文件夹
"""
import shutil
from pathlib import Path
from datetime import datetime


def delete_yolo_dataset(dataset_dir):
    """删除 YOLO 训练数据集"""
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"✗ 数据集目录不存在: {dataset_dir}")
        return
    
    print(f"\n删除 YOLO 数据集: {dataset_dir}")
    
    # 统计文件数量
    images_train = list((dataset_path / "images" / "train").glob("*.png")) if (dataset_path / "images" / "train").exists() else []
    images_val = list((dataset_path / "images" / "val").glob("*.png")) if (dataset_path / "images" / "val").exists() else []
    
    print(f"  训练集图片: {len(images_train)} 张")
    print(f"  验证集图片: {len(images_val)} 张")
    print(f"  总计: {len(images_train) + len(images_val)} 张")
    
    # 删除整个数据集目录
    try:
        shutil.rmtree(dataset_path)
        print(f"✓ 已删除数据集目录")
    except Exception as e:
        print(f"✗ 删除失败: {e}")


def move_original_annotations(source_dir, target_base_dir, model_name):
    """移动原始标注图到统一文件夹"""
    source_path = Path(source_dir)
    target_base = Path(target_base_dir)
    
    # 创建目标目录（带时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_dir = target_base / f"{model_name}_{timestamp}"
    
    print(f"\n移动原始标注图:")
    print(f"  从: {source_dir}")
    print(f"  到: {target_dir}")
    
    # 创建目标目录结构
    (target_dir / "images").mkdir(parents=True, exist_ok=True)
    (target_dir / "labels").mkdir(parents=True, exist_ok=True)
    
    # 获取所有图片和标签
    images = list(source_path.glob("*.png"))
    labels = list(source_path.glob("*.txt"))
    
    print(f"\n  找到 {len(images)} 张图片")
    print(f"  找到 {len(labels)} 个标签文件")
    
    # 复制图片
    copied_images = 0
    for img in images:
        target_img = target_dir / "images" / img.name
        shutil.copy2(img, target_img)
        copied_images += 1
    
    print(f"  ✓ 已复制 {copied_images} 张图片")
    
    # 复制标签
    copied_labels = 0
    for label in labels:
        target_label = target_dir / "labels" / label.name
        shutil.copy2(label, target_label)
        copied_labels += 1
    
    print(f"  ✓ 已复制 {copied_labels} 个标签文件")
    
    # 复制 annotations.json
    annotation_file = source_path / "annotations.json"
    if annotation_file.exists():
        shutil.copy2(annotation_file, target_dir / "annotations.json")
        print(f"  ✓ 已复制 annotations.json")
    
    print(f"\n✓ 原始标注图已保存到: {target_dir}")
    
    return target_dir


def main():
    """主函数"""
    print("=" * 60)
    print("清理转账确认弹窗训练数据")
    print("=" * 60)
    
    # 1. 删除 YOLO 数据集
    print("\n[1/2] 删除 YOLO 训练数据集")
    delete_yolo_dataset("yolo_dataset_transfer_confirm")
    
    # 2. 移动原始标注图
    print("\n[2/2] 移动原始标注图到统一文件夹")
    target_dir = move_original_annotations(
        source_dir="training_data/转账确认弹窗",
        target_base_dir="原始标注图",
        model_name="转账确认弹窗"
    )
    
    print("\n" + "=" * 60)
    print("清理完成！")
    print("=" * 60)
    print(f"✓ YOLO 数据集已删除")
    print(f"✓ 原始标注图已保存到: {target_dir}")
    print(f"✓ 原始数据仍保留在: training_data/转账确认弹窗")


if __name__ == "__main__":
    main()
