"""
移动转账页训练数据到完成文件夹并更新模型注册表
"""
import shutil
import json
from pathlib import Path
from datetime import datetime


def move_transfer_data_to_completed():
    """移动转账页数据到完成文件夹"""
    print("=" * 60)
    print("移动转账页训练数据到完成文件夹")
    print("=" * 60)
    
    # 创建时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 源目录和目标目录
    source_dataset = Path("yolo_dataset_transfer")
    completed_base = Path("training_data_completed")
    target_dir = completed_base / f"transfer_detector_{timestamp}"
    
    # 创建目标目录
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n创建目标目录: {target_dir}")
    
    # 复制数据集
    print("\n复制数据集...")
    
    # 复制图片
    images_src = source_dataset / "images"
    images_dst = target_dir / "images"
    if images_src.exists():
        shutil.copytree(images_src, images_dst)
        print(f"  ✓ 已复制图片: {images_src} -> {images_dst}")
    
    # 复制标签
    labels_src = source_dataset / "labels"
    labels_dst = target_dir / "labels"
    if labels_src.exists():
        shutil.copytree(labels_src, labels_dst)
        print(f"  ✓ 已复制标签: {labels_src} -> {labels_dst}")
    
    # 复制配置文件
    yaml_file = source_dataset / "dataset.yaml"
    if yaml_file.exists():
        shutil.copy2(yaml_file, target_dir / "dataset.yaml")
        print(f"  ✓ 已复制配置: {yaml_file}")
    
    # 复制测试结果
    test_results_src = Path("test_results/transfer_predictions")
    test_results_dst = target_dir / "test_results"
    if test_results_src.exists():
        shutil.copytree(test_results_src, test_results_dst)
        print(f"  ✓ 已复制测试结果: {test_results_src} -> {test_results_dst}")
    
    # 复制模型文件
    model_src = Path("yolo_runs/transfer_detector11/weights/best.pt")
    model_dst = target_dir / "best.pt"
    if model_src.exists():
        shutil.copy2(model_src, model_dst)
        print(f"  ✓ 已复制模型: {model_src} -> {model_dst}")
    
    print(f"\n数据已移动到: {target_dir}")
    
    return target_dir


def update_model_registry():
    """更新模型注册表"""
    print("\n" + "=" * 60)
    print("更新模型注册表")
    print("=" * 60)
    
    registry_file = Path("yolo_model_registry.json")
    
    # 读取现有注册表
    with open(registry_file, 'r', encoding='utf-8') as f:
        registry = json.load(f)
    
    # 添加转账页模型信息
    registry["models"]["transfer"] = {
        "name": "转账页检测模型",
        "page_type": "转账页",
        "model_path": "yolo_runs/transfer_detector11/weights/best.pt",
        "classes": ["全部转账按钮", "ID输入框", "转账金额输入框", "提交按钮", "转账明细文本"],
        "num_classes": 5,
        "performance": {
            "mAP50": 0.991,
            "precision": 0.992,
            "recall": 1.0,
            "mAP50-95": 0.755
        },
        "training_date": "2026-01-27",
        "dataset_size": {
            "original": 55,
            "augmented": 1705,
            "train": 1364,
            "val": 341
        },
        "notes": "使用31倍数据增强，训练50轮，测试100%检测成功率（341/341张图片）"
    }
    
    # 更新最后更新时间
    registry["last_updated"] = datetime.now().strftime("%Y-%m-%d")
    
    # 保存注册表
    with open(registry_file, 'w', encoding='utf-8') as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 已更新模型注册表: {registry_file}")
    print("\n转账页模型信息:")
    print(f"  模型路径: {registry['models']['transfer']['model_path']}")
    print(f"  类别数: {registry['models']['transfer']['num_classes']}")
    print(f"  mAP50: {registry['models']['transfer']['performance']['mAP50']:.1%}")
    print(f"  精确率: {registry['models']['transfer']['performance']['precision']:.1%}")
    print(f"  召回率: {registry['models']['transfer']['performance']['recall']:.1%}")
    print(f"  识别率: 100% (341/341)")


def main():
    """主函数"""
    # 移动数据
    target_dir = move_transfer_data_to_completed()
    
    # 更新注册表
    update_model_registry()
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    print(f"\n数据位置: {target_dir}")
    print(f"模型注册表: yolo_model_registry.json")
    print("\n转账页模型训练和整理完成！")


if __name__ == "__main__":
    main()
