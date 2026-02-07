"""
训练个人页数字识别 YOLO 模型
Train Profile Numbers YOLO Model

目标：替代 OCR，直接用 YOLO 识别余额、积分、抵扣券、优惠券数字
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO


def train_profile_numbers_detector():
    """训练个人页数字识别模型"""
    
    print("=" * 70)
    print("训练个人页数字识别 YOLO 模型")
    print("=" * 70)
    
    # 配置
    config = {
        'model_name': 'profile_numbers',
        'base_model': 'yolov8n.pt',  # 使用 nano 模型（最快）
        'data_yaml': 'yolo_dataset/profile_numbers/data.yaml',
        'epochs': 50,
        'batch_size': 16,
        'img_size': 640,
        'device': 0,  # GPU
        'project': 'runs/detect/yolo_runs',
        'name': 'profile_numbers_detector',
    }
    
    # 类别定义
    classes = [
        "余额数字",      # 0
        "积分数字",      # 1
        "抵扣券数字",    # 2
        "优惠券数字",    # 3
        "昵称文本",      # 4
        "用户ID",        # 5
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
        print(f"  1. 创建目录: yolo_dataset/profile_numbers/")
        print(f"  2. 标注数据（标注整个数字区域，不是单个数字）")
        print(f"  3. 创建 data.yaml 配置文件")
        print(f"\n标注建议：")
        print(f"  - 余额数字：标注整个 '19.78' 区域")
        print(f"  - 积分数字：标注整个 '0' 区域")
        print(f"  - 抵扣券数字：标注整个 '5.97' 区域")
        print(f"  - 优惠券数字：标注整个 '0' 区域")
        print(f"  - 昵称文本：标注整个昵称区域")
        print(f"  - 用户ID：标注整个ID区域")
        return
    
    print(f"\n✓ 数据集配置: {data_yaml_path}")
    
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
            patience=10,  # 早停
            save=True,
            plots=True,
            verbose=True
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
        
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()


def create_dataset_template():
    """创建数据集模板"""
    
    print("\n" + "=" * 70)
    print("创建数据集模板")
    print("=" * 70)
    
    dataset_dir = Path("yolo_dataset/profile_numbers")
    
    # 创建目录结构
    dirs = [
        dataset_dir / "images" / "train",
        dataset_dir / "images" / "val",
        dataset_dir / "labels" / "train",
        dataset_dir / "labels" / "val",
    ]
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ 创建目录: {d}")
    
    # 创建 data.yaml
    data_yaml = dataset_dir / "data.yaml"
    yaml_content = f"""# 个人页数字识别数据集配置
# Profile Numbers Detection Dataset

path: {dataset_dir.absolute()}
train: images/train
val: images/val

# 类别定义
nc: 6
names:
  0: 余额数字
  1: 积分数字
  2: 抵扣券数字
  3: 优惠券数字
  4: 昵称文本
  5: 用户ID

# 标注说明：
# - 标注整个数字区域（不是单个数字）
# - 例如：余额 "19.78" 标注为一个框
# - 昵称和用户ID也标注完整区域
"""
    
    with open(data_yaml, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"  ✓ 创建配置: {data_yaml}")
    
    print(f"\n✅ 数据集模板创建完成！")
    print(f"\n下一步：")
    print(f"  1. 将图片放入: {dataset_dir}/images/train/")
    print(f"  2. 使用标注工具标注（推荐 LabelImg 或 Roboflow）")
    print(f"  3. 将标注文件放入: {dataset_dir}/labels/train/")
    print(f"  4. 运行训练: python train_profile_numbers_yolo.py")
    
    print(f"\n标注技巧：")
    print(f"  - 使用现有的 '原始标注图/个人页_已登录_余额积分/' 数据")
    print(f"  - 重新标注，标注整个数字区域（不是单个数字）")
    print(f"  - 确保标注框紧贴数字边缘")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='训练个人页数字识别 YOLO 模型')
    parser.add_argument('--create-template', action='store_true', 
                       help='创建数据集模板')
    
    args = parser.parse_args()
    
    if args.create_template:
        create_dataset_template()
    else:
        train_profile_numbers_detector()
