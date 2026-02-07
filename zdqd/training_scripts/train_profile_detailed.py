"""
训练个人页详细标注YOLO模型
Train Profile Detailed Annotation YOLO Model
"""

from ultralytics import YOLO
import os

# 数据集配置文件
DATA_YAML = "yolo_dataset/profile_detailed/data.yaml"

# 预训练模型
PRETRAINED_MODEL = "yolo26n.pt"  # 使用YOLOv8n作为基础模型

# 训练参数
EPOCHS = 30  # 训练30轮
BATCH_SIZE = 16
IMG_SIZE = 640

def train_model():
    """训练YOLO模型"""
    
    print("=" * 70)
    print("训练个人页详细标注YOLO模型")
    print("=" * 70)
    
    # 检查数据集配置文件
    if not os.path.exists(DATA_YAML):
        print(f"❌ 数据集配置文件不存在: {DATA_YAML}")
        return
    
    print(f"\n[配置]")
    print(f"  数据集: {DATA_YAML}")
    print(f"  预训练模型: {PRETRAINED_MODEL}")
    print(f"  训练轮数: {EPOCHS}")
    print(f"  批次大小: {BATCH_SIZE}")
    print(f"  图片尺寸: {IMG_SIZE}")
    
    # 加载预训练模型
    print(f"\n[1] 加载预训练模型...")
    model = YOLO(PRETRAINED_MODEL)
    print(f"✓ 模型已加载")
    
    # 开始训练
    print(f"\n[2] 开始训练...")
    print("=" * 70)
    
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        project="runs/detect",
        name="profile_detailed_detector",
        patience=20,  # 早停耐心值
        save=True,
        save_period=10,  # 每10轮保存一次
        cache=False,  # 不缓存图片到内存
        device=0,  # 使用GPU 0
        workers=4,  # 数据加载线程数
        exist_ok=True,  # 允许覆盖已存在的项目
        pretrained=True,
        optimizer='AdamW',
        verbose=True,
        seed=42,
        deterministic=False,
        single_cls=False,
        rect=False,
        cos_lr=True,  # 使用余弦学习率调度
        close_mosaic=10,  # 最后10轮关闭mosaic增强
        amp=True,  # 自动混合精度训练
        fraction=1.0,
        profile=False,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        val=True,
        plots=True
    )
    
    print("\n" + "=" * 70)
    print("✓ 训练完成！")
    print("=" * 70)
    
    # 显示最佳模型路径
    best_model_path = "runs/detect/profile_detailed_detector/weights/best.pt"
    print(f"\n最佳模型: {best_model_path}")
    
    # 显示性能指标
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        print(f"\n[性能指标]")
        if 'metrics/mAP50(B)' in metrics:
            print(f"  mAP50: {metrics['metrics/mAP50(B)']:.3f}")
        if 'metrics/mAP50-95(B)' in metrics:
            print(f"  mAP50-95: {metrics['metrics/mAP50-95(B)']:.3f}")
        if 'metrics/precision(B)' in metrics:
            print(f"  Precision: {metrics['metrics/precision(B)']:.3f}")
        if 'metrics/recall(B)' in metrics:
            print(f"  Recall: {metrics['metrics/recall(B)']:.3f}")
    
    print("\n下一步: 运行测试脚本验证模型效果")
    print("=" * 70)


if __name__ == '__main__':
    train_model()
