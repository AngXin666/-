"""
测试识别稳定性 - 验证同一张图片多次识别是否会得到不同结果
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
from collections import Counter

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


def test_image_stability(model, image_path, transform, device, classes, num_tests=20):
    """测试单张图片的识别稳定性"""
    results = []
    confidences = []
    
    for i in range(num_tests):
        predicted_class, confidence = predict_image(model, image_path, transform, device, classes)
        results.append(predicted_class)
        confidences.append(confidence)
    
    return results, confidences


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("识别稳定性测试 - 同一图片多次识别")
    print("=" * 80)
    
    # 配置
    training_data_dir = Path("标注工具_完整独立版/training_data")
    model_path = Path("models/page_classifier_pytorch_best.pth")
    classes_path = Path("models/page_classes.json")
    num_tests = 20  # 每张图片测试20次
    
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
    
    print(f"\n🔍 开始测试...")
    print(f"  • 每个类别随机抽取1张图片")
    print(f"  • 每张图片识别{num_tests}次")
    print(f"  • 检查识别结果是否一致")
    print()
    
    # 统计结果
    total_classes = 0
    stable_classes = 0
    unstable_classes = []
    
    print(f"{'类别':<20} {'测试次数':<10} {'识别结果':<15} {'置信度范围':<20} {'状态':<10}")
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
        
        # 随机抽取1张图片
        test_image = random.choice(image_files)
        
        # 测试识别稳定性
        results, confidences = test_image_stability(model, test_image, transform, device, classes, num_tests)
        
        # 统计识别结果
        result_counts = Counter(results)
        unique_results = len(result_counts)
        most_common_result = result_counts.most_common(1)[0][0]
        
        # 置信度范围
        min_conf = min(confidences) * 100
        max_conf = max(confidences) * 100
        avg_conf = sum(confidences) / len(confidences) * 100
        
        # 判断稳定性
        is_stable = unique_results == 1 and most_common_result == class_name
        
        total_classes += 1
        if is_stable:
            stable_classes += 1
            status = "✓ 稳定"
        else:
            unstable_classes.append({
                'class_name': class_name,
                'results': result_counts,
                'confidences': (min_conf, max_conf, avg_conf)
            })
            status = "✗ 不稳定"
        
        # 显示结果
        if unique_results == 1:
            result_str = f"{most_common_result} (100%)"
        else:
            result_str = f"{unique_results}种结果"
        
        conf_range = f"{min_conf:.2f}%-{max_conf:.2f}%"
        
        print(f"{class_name:<20} {num_tests:<10} {result_str:<15} {conf_range:<20} {status:<10}")
    
    # 总体统计
    stability_rate = stable_classes / total_classes * 100
    
    print("-" * 80)
    print(f"{'总计':<20} {total_classes}个类别  稳定: {stable_classes}  不稳定: {len(unstable_classes)}  稳定率: {stability_rate:.1f}%")
    
    print("\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)
    
    # 显示不稳定的类别详情
    if unstable_classes:
        print(f"\n⚠️  发现 {len(unstable_classes)} 个不稳定的类别：")
        print()
        for item in unstable_classes:
            print(f"  类别: {item['class_name']}")
            print(f"  识别结果分布:")
            for result, count in item['results'].most_common():
                percentage = count / num_tests * 100
                print(f"    • {result}: {count}次 ({percentage:.1f}%)")
            min_conf, max_conf, avg_conf = item['confidences']
            print(f"  置信度: {min_conf:.2f}% - {max_conf:.2f}% (平均: {avg_conf:.2f}%)")
            print()
    else:
        print(f"\n✅ 所有类别识别结果完全稳定！")
        print(f"  • 每张图片{num_tests}次识别结果完全一致")
        print(f"  • 没有出现同一图片被识别成不同类型的情况")
    
    print(f"\n📊 总体稳定率: {stability_rate:.2f}%")
    print()


if __name__ == "__main__":
    main()
