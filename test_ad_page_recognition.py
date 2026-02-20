"""
测试新模型对广告页的识别准确率
使用训练数据中的广告页图片进行测试
"""
import os
import sys
from pathlib import Path
import json
import time

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


class PageClassifier(nn.Module):
    """页面分类器模型 - 使用MobileNetV2（匹配训练脚本）"""
    
    def __init__(self, num_classes):
        super(PageClassifier, self).__init__()
        
        # 使用MobileNetV2作为骨干网络（与训练脚本完全一致）
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


def test_ad_page_recognition():
    """测试广告页识别准确率"""
    print("\n" + "=" * 80)
    print("🧪 测试新模型对广告页的识别准确率")
    print("=" * 80)
    
    # 配置路径
    model_path = Path("models/page_classifier_pytorch_best.pth")
    classes_path = Path("models/page_classes.json")
    ad_data_dir = Path("标注工具_完整独立版/training_data/广告页")
    
    # 检查文件是否存在
    if not model_path.exists():
        print(f"\n❌ 错误: 模型文件不存在: {model_path}")
        return
    
    if not classes_path.exists():
        print(f"\n❌ 错误: 类别文件不存在: {classes_path}")
        return
    
    if not ad_data_dir.exists():
        print(f"\n❌ 错误: 广告页数据目录不存在: {ad_data_dir}")
        return
    
    print(f"\n📁 模型路径: {model_path}")
    print(f"📁 类别路径: {classes_path}")
    print(f"📁 测试数据: {ad_data_dir}")
    
    # 加载类别列表
    print(f"\n📦 加载类别列表...")
    with open(classes_path, 'r', encoding='utf-8') as f:
        classes = json.load(f)
    
    print(f"  ✓ 已加载 {len(classes)} 个类别")
    
    # 检查是否包含广告页
    if "广告页" not in classes:
        print(f"\n❌ 错误: 类别列表中没有'广告页'")
        print(f"  当前类别: {classes}")
        return
    
    ad_class_idx = classes.index("广告页")
    print(f"  ✓ '广告页'类别索引: {ad_class_idx}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  设备: {device}")
    
    # 加载模型
    print(f"\n🏗️  加载模型...")
    model = PageClassifier(num_classes=len(classes))
    
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  ✓ 模型训练轮数: {checkpoint.get('epoch', 'N/A')}")
        print(f"  ✓ 验证准确率: {checkpoint.get('val_acc', 'N/A'):.2f}%")
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    print(f"  ✓ 模型已加载并设置为评估模式")
    
    # 设置图片预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 获取所有广告页图片
    ad_images = list(ad_data_dir.glob("*.png"))
    total_images = len(ad_images)
    
    if total_images == 0:
        print(f"\n❌ 错误: 广告页目录中没有图片")
        return
    
    print(f"\n📊 测试数据:")
    print(f"  • 广告页图片数量: {total_images}")
    
    # 测试识别
    print(f"\n🚀 开始测试...")
    print("=" * 80)
    
    correct = 0
    wrong = 0
    wrong_predictions = []
    
    start_time = time.time()
    
    for i, img_path in enumerate(ad_images, 1):
        # 加载图片
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"\n  ⚠️  跳过无效图片: {img_path.name} ({e})")
            total_images -= 1
            continue
        
        # 预处理
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # 预测
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            predicted_class = classes[predicted_idx.item()]
            confidence_value = confidence.item()
        
        # 判断是否正确
        if predicted_class == "广告页":
            correct += 1
            status = "✓"
        else:
            wrong += 1
            status = "✗"
            wrong_predictions.append({
                'file': img_path.name,
                'predicted': predicted_class,
                'confidence': confidence_value
            })
        
        # 显示进度
        if i % 50 == 0 or i == total_images:
            accuracy = (correct / i) * 100
            print(f"  进度: {i}/{total_images} ({i/total_images*100:.1f}%) | "
                  f"准确率: {accuracy:.2f}% | "
                  f"正确: {correct} | 错误: {wrong}")
    
    # 计算总耗时
    elapsed_time = time.time() - start_time
    avg_time = elapsed_time / total_images * 1000  # 毫秒
    
    # 显示结果
    print("\n" + "=" * 80)
    print("📊 测试结果")
    print("=" * 80)
    
    accuracy = (correct / total_images) * 100
    
    print(f"\n总体统计:")
    print(f"  • 测试图片数: {total_images}")
    print(f"  • 正确识别: {correct}")
    print(f"  • 错误识别: {wrong}")
    print(f"  • 准确率: {accuracy:.2f}%")
    print(f"  • 总耗时: {elapsed_time:.2f}秒")
    print(f"  • 平均耗时: {avg_time:.2f}毫秒/张")
    
    # 显示错误预测
    if wrong_predictions:
        print(f"\n❌ 错误预测详情 ({len(wrong_predictions)} 个):")
        for i, pred in enumerate(wrong_predictions[:10], 1):  # 只显示前10个
            print(f"  {i}. {pred['file']}")
            print(f"     预测为: {pred['predicted']} (置信度: {pred['confidence']:.2%})")
        
        if len(wrong_predictions) > 10:
            print(f"  ... 还有 {len(wrong_predictions) - 10} 个错误预测")
    
    # 评估结果
    print(f"\n💡 评估:")
    if accuracy >= 95:
        print(f"  ✅ 优秀! 准确率达到 {accuracy:.2f}%，可以使用该模型")
    elif accuracy >= 90:
        print(f"  ⚠️  良好，准确率 {accuracy:.2f}%，建议进一步优化")
    else:
        print(f"  ❌ 不合格，准确率仅 {accuracy:.2f}%，需要重新训练")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    try:
        test_ad_page_recognition()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户取消测试")
    except Exception as e:
        print(f"\n\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
