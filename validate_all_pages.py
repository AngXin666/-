"""
验证所有页面类型的识别准确率
使用训练数据中的图片进行混合验证
"""
import sys
from pathlib import Path
import random

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch
from torchvision import transforms
from PIL import Image
import json

# 导入页面分类器模型
sys.path.insert(0, str(Path(__file__).parent / '标注工具_完整独立版'))
from scripts.train_page_classifier_pytorch import PageClassifier


def load_model(model_path, classes_path, device):
    """加载模型"""
    # 加载类别列表
    with open(classes_path, 'r', encoding='utf-8') as f:
        classes = json.load(f)
    
    # 创建模型
    model = PageClassifier(num_classes=len(classes))
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, classes


def predict_image(model, image_path, transform, device, classes):
    """预测单张图片"""
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(1)
    
    predicted_class = classes[predicted.item()]
    confidence_value = confidence.item()
    
    return predicted_class, confidence_value


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("页面分类器混合验证 - 使用训练数据")
    print("=" * 80)
    
    # 配置
    training_data_dir = Path("标注工具_完整独立版/training_data")
    model_path = Path("models/page_classifier_pytorch_best.pth")
    classes_path = Path("models/page_classes.json")
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  设备: {device}")
    
    # 加载模型
    print(f"\n📦 加载模型...")
    model, classes = load_model(model_path, classes_path, device)
    print(f"  ✓ 已加载模型，共 {len(classes)} 个类别")
    
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 统计结果
    total_correct = 0
    total_images = 0
    class_stats = {}
    
    print(f"\n🔍 开始验证...")
    print(f"{'类别':<20} {'测试数':<8} {'正确数':<8} {'准确率':<10} {'平均置信度':<12}")
    print("-" * 80)
    
    # 遍历所有类别目录
    for class_dir in sorted(training_data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        
        # 获取该类别的所有图片
        image_files = list(class_dir.glob("*.png"))
        
        if len(image_files) == 0:
            continue
        
        # 随机抽取最多10张图片进行测试
        sample_size = min(10, len(image_files))
        sample_images = random.sample(image_files, sample_size)
        
        # 测试每张图片
        correct = 0
        confidences = []
        
        for img_path in sample_images:
            predicted_class, confidence = predict_image(model, img_path, transform, device, classes)
            
            if predicted_class == class_name:
                correct += 1
            
            confidences.append(confidence)
            total_images += 1
        
        total_correct += correct
        accuracy = correct / sample_size * 100
        avg_confidence = sum(confidences) / len(confidences) * 100
        
        # 记录统计
        class_stats[class_name] = {
            'total': sample_size,
            'correct': correct,
            'accuracy': accuracy,
            'avg_confidence': avg_confidence
        }
        
        # 显示结果
        status = "✓" if accuracy == 100 else "✗" if accuracy < 50 else "⚠"
        print(f"{status} {class_name:<18} {sample_size:<8} {correct:<8} {accuracy:>6.1f}%    {avg_confidence:>6.2f}%")
    
    # 总体统计
    overall_accuracy = total_correct / total_images * 100
    
    print("-" * 80)
    print(f"{'总计':<20} {total_images:<8} {total_correct:<8} {overall_accuracy:>6.1f}%")
    
    print("\n" + "=" * 80)
    print("验证完成!")
    print("=" * 80)
    
    # 显示问题类别
    problem_classes = [name for name, stats in class_stats.items() if stats['accuracy'] < 100]
    
    if problem_classes:
        print(f"\n⚠️  识别准确率不足100%的类别：")
        for class_name in problem_classes:
            stats = class_stats[class_name]
            print(f"  • {class_name}: {stats['accuracy']:.1f}% ({stats['correct']}/{stats['total']})")
    else:
        print(f"\n✅ 所有类别识别准确率均为100%！")
    
    print(f"\n📊 总体准确率: {overall_accuracy:.2f}%")
    print()


if __name__ == "__main__":
    main()
