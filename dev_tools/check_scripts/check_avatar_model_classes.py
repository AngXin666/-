"""检查 avatar_homepage 模型的类别"""
import os
from ultralytics import YOLO

os.chdir("zdqd")

model_path = "models/runs/detect/runs/detect/yolo_runs/avatar_homepage_detector/train/weights/best.pt"
model = YOLO(model_path)

print("模型类别:")
print(model.names)
