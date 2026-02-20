#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
清理广告页训练数据 - 删除标注错误的图片
"""

import os
import shutil
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import json

# 加载模型和类别
model_path = "models/page_classifier_pytorch_best.pth"
classes_path = "models/page_classes.json"

print("=" * 80)
print("🧹 清理广告页训练数据")
print("=" * 80)

# 加载类别
with open(classes_path, 'r', encoding='utf-8') as f:
    classes = json.load(f)

print(f"\n📦 加载模型: {model_path}")

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(model_path, map_location=device)

# 提取模型
if isinstance(checkpoint, dict):
    if 'model_state_dict' in checkpoint:
        # 需要重新创建模型结构
        from torchvision import models
        model = models.efficientnet_b0(weights=None)
        num_classes = len(classes)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model = checkpoint['model']
else:
    model = checkpoint

model = model.to(device)
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 广告页训练数据目录
ad_dir = Path("标注工具_完整独立版/training_data/广告页")
error_dir = Path("标注工具_完整独立版/training_data/_标注错误_广告页")
error_dir.mkdir(exist_ok=True)

print(f"\n📁 检查目录: {ad_dir}")
print(f"📁 错误图片将移动到: {error_dir}")

# 获取所有图片
image_files = list(ad_dir.glob("*.png")) + list(ad_dir.glob("*.jpg"))
print(f"\n📊 找到 {len(image_files)} 张图片")

# 统计
moved_count = 0
kept_count = 0
error_predictions = {}

print("\n🔍 开始检查...")
print("-" * 80)

for img_path in image_files:
    try:
        # 加载图片
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # 预测
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            predicted_class = classes[predicted.item()]
            conf_percent = confidence.item() * 100
        
        # 如果预测不是"广告页"且置信度高于50%，认为是标注错误
        if predicted_class != "广告页" and conf_percent > 50:
            # 移动到错误目录
            dest_path = error_dir / img_path.name
            shutil.move(str(img_path), str(dest_path))
            moved_count += 1
            
            # 记录错误预测
            if predicted_class not in error_predictions:
                error_predictions[predicted_class] = []
            error_predictions[predicted_class].append((img_path.name, conf_percent))
            
            print(f"  ❌ {img_path.name}: 预测为 '{predicted_class}' ({conf_percent:.1f}%) - 已移动")
        else:
            kept_count += 1
            if kept_count % 10 == 0:
                print(f"  ✓ 已检查 {kept_count} 张正确的图片...")
    
    except Exception as e:
        print(f"  ⚠️ 处理 {img_path.name} 时出错: {e}")

print("-" * 80)
print(f"\n📊 清理结果:")
print(f"  • 保留: {kept_count} 张")
print(f"  • 移动: {moved_count} 张")

if error_predictions:
    print(f"\n📋 错误标注统计:")
    for pred_class, files in sorted(error_predictions.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  • {pred_class}: {len(files)} 张")
        # 显示前3个例子
        for filename, conf in files[:3]:
            print(f"    - {filename} ({conf:.1f}%)")
        if len(files) > 3:
            print(f"    ... 还有 {len(files) - 3} 张")

print(f"\n✅ 清理完成!")
print(f"💡 错误图片已移动到: {error_dir}")
