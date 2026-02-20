#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单清理广告页训练数据 - 使用现有模型结构
"""

import os
import shutil
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json

# 定义模型结构（与训练脚本相同）
class PageClassifier(nn.Module):
    """页面分类器模型 - 使用MobileNetV2"""
    
    def __init__(self, num_classes):
        super(PageClassifier, self).__init__()
        
        # 使用MobileNetV2作为骨干网络
        self.mobilenet = models.mobilenet_v2(weights=None)
        
        # 替换分类器
        in_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.mobilenet(x)


print("=" * 80)
print("🧹 清理广告页训练数据")
print("=" * 80)

# 加载类别
classes_path = "models/page_classes.json"
with open(classes_path, 'r', encoding='utf-8') as f:
    classes = json.load(f)

print(f"\n📦 加载模型...")

# 创建模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PageClassifier(num_classes=len(classes))

# 加载权重
model_path = "models/page_classifier_pytorch_best.pth"
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print(f"✓ 模型加载成功 (设备: {device})")

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

# 获取所有图片（排除增强图）
image_files = [f for f in ad_dir.glob("*.png") if "_aug_" not in f.name]
print(f"\n📊 找到 {len(image_files)} 张原始图片（不含增强图）")

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
    
    except Exception as e:
        print(f"  ⚠️ 处理 {img_path.name} 时出错: {e}")

print("-" * 80)
print(f"\n📊 清理结果:")
print(f"  • 保留: {kept_count} 张原始图片")
print(f"  • 移动: {moved_count} 张错误标注图片")

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
print(f"💡 注意: 增强图片未处理，重新训练时会自动重新生成")
