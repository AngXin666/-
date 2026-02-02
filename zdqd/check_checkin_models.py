"""检查所有签到模型的类别"""
from ultralytics import YOLO
from pathlib import Path

models_to_check = [
    "models/yolo_runs/checkin_detector/weights/best.pt",
    "models/yolo_runs/checkin_detector2/weights/best.pt",
]

print("="*60)
print("检查签到模型类别")
print("="*60)

for model_path in models_to_check:
    print(f"\n模型: {model_path}")
    
    if not Path(model_path).exists():
        print(f"  ❌ 模型文件不存在")
        continue
    
    try:
        model = YOLO(model_path)
        if hasattr(model, 'names'):
            classes = model.names
            print(f"  ✓ 类别: {classes}")
        else:
            print(f"  ⚠ 无法获取类别名称")
    except Exception as e:
        print(f"  ❌ 加载失败: {e}")

print("\n" + "="*60)
