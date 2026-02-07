"""检查模型信息"""
import torch
from pathlib import Path

model_path = Path("models/page_classifier_pytorch_best.pth")

if model_path.exists():
    checkpoint = torch.load(model_path, map_location='cpu')
    
    print("模型信息:")
    print(f"  - Epoch: {checkpoint.get('epoch', '未知')}")
    print(f"  - 验证准确率: {checkpoint.get('val_acc', '未知')}")
    print(f"  - 验证损失: {checkpoint.get('val_loss', '未知')}")
    print(f"  - 包含的键: {list(checkpoint.keys())}")
else:
    print("模型文件不存在")
