"""
清理交易流水训练数据
1. 删除YOLO数据集
2. 移动原始标注图到 原始标注图/ 文件夹
"""
import shutil
from pathlib import Path
from datetime import datetime


def cleanup_transaction_history_data():
    """清理交易流水训练数据"""
    print("=" * 60)
    print("清理交易流水训练数据")
    print("=" * 60)
    
    # 1. 移动原始标注图
    print("\n[1/2] 移动原始标注图...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_dir = Path("training_data/交易流水")
    target_dir = Path(f"原始标注图/交易流水_{timestamp}")
    
    if source_dir.exists():
        # 创建目标目录
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "images").mkdir(exist_ok=True)
        (target_dir / "labels").mkdir(exist_ok=True)
        
        # 复制图片
        images = list(source_dir.glob("*.png"))
        for img in images:
            shutil.copy2(img, target_dir / "images" / img.name)
        
        print(f"  ✓ 已复制 {len(images)} 张原始图片")
        
        # 复制标注文件
        annotation_file = source_dir / "annotations.json"
        if annotation_file.exists():
            shutil.copy2(annotation_file, target_dir / "annotations.json")
            print(f"  ✓ 已复制标注文件")
        
        print(f"  ✓ 原始标注图已保存到: {target_dir}")
    else:
        print(f"  ✗ 源目录不存在: {source_dir}")
    
    # 2. 删除YOLO数据集
    print("\n[2/2] 删除YOLO数据集...")
    dataset_dir = Path("yolo_dataset_transaction_history")
    
    if dataset_dir.exists():
        # 统计数据
        train_images = list((dataset_dir / "images" / "train").glob("*.png"))
        val_images = list((dataset_dir / "images" / "val").glob("*.png"))
        
        print(f"  删除训练集: {len(train_images)} 张")
        print(f"  删除验证集: {len(val_images)} 张")
        
        # 删除整个目录
        shutil.rmtree(dataset_dir)
        print(f"  ✓ 已删除YOLO数据集: {dataset_dir}")
    else:
        print(f"  ✗ YOLO数据集不存在: {dataset_dir}")
    
    print("\n" + "=" * 60)
    print("清理完成！")
    print("=" * 60)
    print(f"原始标注图: {target_dir}")
    print(f"模型路径: runs/detect/yolo_runs/transaction_history_detector/exp2/weights/best.pt")
    print("=" * 60)


if __name__ == "__main__":
    cleanup_transaction_history_data()
