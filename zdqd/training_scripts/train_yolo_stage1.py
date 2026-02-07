"""
YOLO 分阶段训练 - 阶段1：核心按钮
只训练最重要的 4 个类别
"""
from ultralytics import YOLO
from pathlib import Path
import yaml
import shutil


def prepare_stage1_dataset():
    """准备阶段1数据集：核心按钮"""
    print("准备阶段1数据集：核心按钮")
    
    # 阶段1：核心按钮（原类别ID）
    stage1_classes = {
        7: 0,   # 签到按钮 -> 签到按钮
        8: 0,   # 每日签到按钮 -> 签到按钮
        9: 1,   # 转账按钮 -> 转账按钮
        10: 1,  # 转增按钮 -> 转账按钮
        28: 1,  # 全部转账按钮 -> 转账按钮
        0: 2,   # 同意按钮 -> 确认按钮
        2: 2,   # 确认按钮 -> 确认按钮
        11: 2,  # 提交按钮 -> 确认按钮
        5: 3,   # 返回按钮 -> 返回按钮
    }
    
    new_class_names = [
        "签到按钮",    # 0
        "转账按钮",    # 1
        "确认按钮",    # 2
        "返回按钮",    # 3
    ]
    
    # 创建数据集目录
    dataset_root = Path("yolo_dataset_stage1")
    dataset_root.mkdir(exist_ok=True)
    (dataset_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    # 从原数据集复制并转换标签
    print("转换标签文件...")
    
    for split in ['train', 'val']:
        src_img_dir = Path("yolo_dataset") / "images" / split
        src_label_dir = Path("yolo_dataset") / "labels" / split
        dst_img_dir = dataset_root / "images" / split
        dst_label_dir = dataset_root / "labels" / split
        
        img_files = list(src_img_dir.glob("*.png"))
        copied_count = 0
        
        for img_file in img_files:
            label_file = src_label_dir / img_file.with_suffix(".txt").name
            if not label_file.exists():
                continue
            
            # 读取标签
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            # 转换标签
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    old_class = int(parts[0])
                    if old_class in stage1_classes:
                        new_class = stage1_classes[old_class]
                        new_line = f"{new_class} {' '.join(parts[1:])}\n"
                        new_lines.append(new_line)
            
            # 只保存有标注的图片
            if new_lines:
                shutil.copy(img_file, dst_img_dir / img_file.name)
                with open(dst_label_dir / label_file.name, 'w') as f:
                    f.writelines(new_lines)
                copied_count += 1
        
        print(f"{split} 集: {copied_count} 张图片")
    
    print(f"\n✓ 阶段1数据集准备完成")
    print(f"  类别数: {len(new_class_names)}")
    print(f"  类别: {new_class_names}")
    
    return dataset_root, new_class_names


def create_dataset_yaml_stage1(dataset_root, classes):
    """创建阶段1数据集配置"""
    print("\n创建数据集配置...")
    
    config = {
        'path': str(dataset_root.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(classes),
        'names': classes
    }
    
    yaml_path = dataset_root / "dataset.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    
    print(f"✓ 配置文件: {yaml_path}")
    return yaml_path


def train_stage1(dataset_yaml, epochs=100):
    """阶段1训练：核心按钮"""
    import multiprocessing
    
    print("\n" + "=" * 60)
    print("阶段1训练：核心按钮")
    print("=" * 60)
    print(f"训练轮数: {epochs}")
    print(f"Batch Size: 32 (稳定)")
    print(f"验证频率: 每 5 轮")
    print(f"Early Stopping: 关闭")
    print("=" * 60)
    
    cpu_count = multiprocessing.cpu_count()
    workers = min(cpu_count, 16)
    
    # 加载预训练模型
    model = YOLO('yolov8n.pt')
    
    # 训练参数
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=640,
        batch=32,  # 稳定 batch size
        device=0,
        
        # 关闭 Early Stopping
        patience=0,
        
        # 验证频率
        val=True,
        save_period=5,  # 每 5 轮保存一次
        
        # 数据增强（保持原有）
        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.5,
        degrees=10.0,
        translate=0.2,
        scale=0.9,
        shear=5.0,
        perspective=0.001,
        flipud=0.1,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        
        # 其他参数
        save=True,
        project='yolo_runs',
        name='stage1_buttons',
        exist_ok=True,
        verbose=True,
        cache='disk',
        workers=workers,
        amp=True,
        close_mosaic=10,
        rect=True,
    )
    
    print("\n阶段1训练完成！")
    print(f"模型保存: yolo_runs/stage1_buttons/weights/best.pt")
    
    return model


def main():
    """主函数"""
    print("=" * 60)
    print("YOLO 分阶段训练 - 阶段1")
    print("=" * 60)
    print("\n训练策略:")
    print("✓ 只训练 4 个核心按钮类别")
    print("✓ Batch Size: 32 (稳定训练)")
    print("✓ 每 5 轮验证一次 (减少验证时间)")
    print("✓ 100 轮训练")
    print("=" * 60)
    
    # 1. 准备阶段1数据集
    dataset_root, classes = prepare_stage1_dataset()
    
    # 2. 创建配置文件
    dataset_yaml = create_dataset_yaml_stage1(dataset_root, classes)
    
    # 3. 训练
    model = train_stage1(dataset_yaml, epochs=100)
    
    # 4. 验证
    print("\n验证模型...")
    results = model.val(data=str(dataset_yaml))
    print(f"\nmAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    
    print("\n" + "=" * 60)
    print("阶段1训练完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
