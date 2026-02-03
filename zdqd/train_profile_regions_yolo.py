"""
训练个人页区域检测 YOLO 模型
Train Profile Regions YOLO Model

检测两个区域：
1. 确认按钮区域（昵称+ID）
2. 数据区域（余额+积分+抵扣劵+优惠劵）
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO


def train_profile_regions_detector():
    """训练个人页区域检测模型"""
    
    print("=" * 70)
    print("训练个人页区域检测 YOLO 模型")
    print("=" * 70)
    
    # 配置
    config = {
        'model_name': 'profile_regions',
        'base_model': 'yolov8n.pt',  # 使用 nano 模型（最快）
        'data_yaml': 'yolo_dataset/profile_regions/data.yaml',
        'epochs': 100,
        'batch_size': 16,
        'img_size': 640,
        'device': 0,  # GPU
        'project': 'runs/detect',
        'name': 'profile_regions_detector',
    }
    
    # 类别定义
    classes = [
        "确认按钮区域",  # 0 - 昵称+ID
        "数据区域",      # 1 - 余额+积分+抵扣劵+优惠劵
    ]
    
    print(f"\n[配置]")
    print(f"  模型: {config['base_model']}")
    print(f"  类别数: {len(classes)}")
    print(f"  训练轮数: {config['epochs']}")
    print(f"  批次大小: {config['batch_size']}")
    
    print(f"\n[类别列表]")
    for i, cls in enumerate(classes):
        print(f"  {i}: {cls}")
    
    # 检查数据集
    data_yaml_path = Path(config['data_yaml'])
    if not data_yaml_path.exists():
        print(f"\n❌ 数据集配置文件不存在: {data_yaml_path}")
        print(f"\n请先准备数据集：")
        print(f"  1. python prepare_profile_region_data.py")
        print(f"  2. python split_and_augment_dataset.py --dataset profile_regions")
        return
    
    print(f"\n✓ 数据集配置: {data_yaml_path}")
    
    # 检查训练数据
    train_img_dir = Path("yolo_dataset/profile_regions/images/train")
    val_img_dir = Path("yolo_dataset/profile_regions/images/val")
    
    train_count = len(list(train_img_dir.glob("*.png"))) if train_img_dir.exists() else 0
    val_count = len(list(val_img_dir.glob("*.png"))) if val_img_dir.exists() else 0
    
    print(f"\n[数据集统计]")
    print(f"  训练集: {train_count} 张")
    print(f"  验证集: {val_count} 张")
    print(f"  总计: {train_count + val_count} 张")
    
    if train_count == 0 or val_count == 0:
        print(f"\n❌ 数据集为空，请先运行数据准备脚本")
        return
    
    # 加载模型
    print(f"\n[1] 加载基础模型...")
    model = YOLO(config['base_model'])
    print(f"  ✓ 模型已加载")
    
    # 开始训练
    print(f"\n[2] 开始训练...")
    print(f"  保存路径: {config['project']}/{config['name']}")
    
    try:
        results = model.train(
            data=str(data_yaml_path),
            epochs=config['epochs'],
            batch=config['batch_size'],
            imgsz=config['img_size'],
            device=config['device'],
            project=config['project'],
            name=config['name'],
            patience=15,  # 早停
            save=True,
            plots=True,
            verbose=True,
            # 数据增强参数（YOLO内置）
            hsv_h=0.015,  # 色调
            hsv_s=0.7,    # 饱和度
            hsv_v=0.4,    # 明度
            degrees=5.0,  # 旋转
            translate=0.1,  # 平移
            scale=0.5,    # 缩放
            flipud=0.0,   # 上下翻转（不适合文字）
            fliplr=0.0,   # 左右翻转（不适合文字）
            mosaic=1.0,   # 马赛克增强
        )
        
        print(f"\n✅ 训练完成！")
        print(f"  最佳模型: {config['project']}/{config['name']}/weights/best.pt")
        
        # 显示性能指标
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            print(f"\n[性能指标]")
            print(f"  mAP50: {metrics.get('metrics/mAP50(B)', 0):.3f}")
            print(f"  mAP50-95: {metrics.get('metrics/mAP50-95(B)', 0):.3f}")
            print(f"  Precision: {metrics.get('metrics/precision(B)', 0):.3f}")
            print(f"  Recall: {metrics.get('metrics/recall(B)', 0):.3f}")
        
        print(f"\n下一步：")
        print(f"  测试模型: python test_profile_regions.py")
        
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    train_profile_regions_detector()
