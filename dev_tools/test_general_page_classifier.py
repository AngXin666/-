"""
测试通用页面分类器模型
验证模型对各个类别的识别准确率
"""
import sys
from pathlib import Path
import json
import random

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


class PageClassifier(nn.Module):
    """页面分类器模型 - 使用MobileNetV3"""
    
    def __init__(self, num_classes):
        super(PageClassifier, self).__init__()
        
        # 使用MobileNetV3-Large作为骨干网络
        self.mobilenet = models.mobilenet_v3_large(weights=None)
        
        # 替换MobileNetV3的分类器
        in_features = self.mobilenet.classifier[0].in_features  # 960
        self.mobilenet.classifier = nn.Sequential(
            nn.Linear(in_features, 1280),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )
    
    def forward(self, x):
        return self.mobilenet(x)


def test_model():
    """测试模型"""
    print("\n" + "=" * 80)
    print("🧪 测试通用页面分类器")
    print("=" * 80)
    
    # 配置
    script_dir = Path(__file__).parent.parent
    model_path = script_dir / "models" / "page_classifier_pytorch_best.pth"
    classes_path = script_dir / "models" / "page_classes.json"
    training_data_dir = script_dir / "标注工具_完整独立版" / "training_data"
    
    # 检查文件
    if not model_path.exists():
        print(f"\n❌ 模型文件不存在: {model_path}")
        return
    
    if not classes_path.exists():
        print(f"\n❌ 类别文件不存在: {classes_path}")
        return
    
    # 加载类别
    with open(classes_path, 'r', encoding='utf-8') as f:
        classes = json.load(f)
    
    print(f"\n📋 类别数: {len(classes)}")
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  设备: {device}")
    
    # 加载模型
    print(f"\n🏗️  加载模型...")
    model = PageClassifier(num_classes=len(classes))
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"  ✓ 模型已加载")
    print(f"  • 训练轮数: {checkpoint['epoch']}")
    print(f"  • 验证准确率: {checkpoint['val_acc']:.2f}%")
    
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 测试每个类别
    print("\n" + "=" * 80)
    print("🧪 测试各类别识别准确率")
    print("=" * 80)
    
    total_correct = 0
    total_samples = 0
    category_results = []
    
    for class_idx, class_name in enumerate(classes):
        class_dir = training_data_dir / class_name
        
        if not class_dir.exists():
            print(f"\n⚠️  {class_name}: 目录不存在，跳过")
            continue
        
        # 获取所有图片
        image_files = list(class_dir.glob("*.png"))
        
        if len(image_files) == 0:
            print(f"\n⚠️  {class_name}: 没有图片，跳过")
            continue
        
        # 随机抽样测试（最多50张）
        test_samples = random.sample(image_files, min(50, len(image_files)))
        
        correct = 0
        predictions = {}
        
        for img_path in test_samples:
            # 加载图片
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            # 预测
            with torch.no_grad():
                output = model(image_tensor)
                _, predicted = output.max(1)
                predicted_class = classes[predicted.item()]
            
            # 统计
            if predicted.item() == class_idx:
                correct += 1
            
            # 记录预测结果
            if predicted_class not in predictions:
                predictions[predicted_class] = 0
            predictions[predicted_class] += 1
        
        accuracy = 100.0 * correct / len(test_samples)
        total_correct += correct
        total_samples += len(test_samples)
        
        # 显示结果
        status = "✓" if accuracy >= 95 else "⚠️" if accuracy >= 80 else "❌"
        print(f"\n{status} {class_name}:")
        print(f"  • 测试样本: {len(test_samples)} 张")
        print(f"  • 正确识别: {correct} 张")
        print(f"  • 准确率: {accuracy:.2f}%")
        
        # 显示错误预测
        if accuracy < 100:
            print(f"  • 预测分布:")
            for pred_class, count in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
                if pred_class == class_name:
                    print(f"    - {pred_class}: {count} 张 ✓")
                else:
                    print(f"    - {pred_class}: {count} 张")
        
        category_results.append({
            'class_name': class_name,
            'accuracy': accuracy,
            'correct': correct,
            'total': len(test_samples)
        })
    
    # 总体统计
    overall_accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0
    
    print("\n" + "=" * 80)
    print("📊 总体统计")
    print("=" * 80)
    print(f"  • 总测试样本: {total_samples} 张")
    print(f"  • 正确识别: {total_correct} 张")
    print(f"  • 总体准确率: {overall_accuracy:.2f}%")
    
    # 按准确率排序
    category_results.sort(key=lambda x: x['accuracy'])
    
    print("\n📉 准确率最低的5个类别:")
    for i, result in enumerate(category_results[:5], 1):
        print(f"  {i}. {result['class_name']}: {result['accuracy']:.2f}% ({result['correct']}/{result['total']})")
    
    print("\n📈 准确率最高的5个类别:")
    for i, result in enumerate(category_results[-5:][::-1], 1):
        print(f"  {i}. {result['class_name']}: {result['accuracy']:.2f}% ({result['correct']}/{result['total']})")
    
    print("\n✅ 测试完成!")


if __name__ == '__main__':
    try:
        test_model()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户取消测试")
    except Exception as e:
        print(f"\n\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
