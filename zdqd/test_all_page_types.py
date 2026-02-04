"""
测试所有页面类型的识别准确率
"""
import sys
from pathlib import Path
import os
import json
import random

sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("❌ PIL未安装")
    sys.exit(1)

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms, models
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("❌ PyTorch未安装")
    sys.exit(1)


def test_all_page_types():
    """测试所有页面类型"""
    print("=" * 70)
    print("测试所有页面类型的识别准确率")
    print("=" * 70)
    
    # 检查模型文件
    print("\n[1] 加载模型和类别...")
    model_path = 'page_classifier_pytorch_best.pth'
    classes_path = 'page_classes.json'
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    if not os.path.exists(classes_path):
        print(f"❌ 类别文件不存在: {classes_path}")
        return
    
    # 加载类别列表
    with open(classes_path, 'r', encoding='utf-8') as f:
        classes = json.load(f)
    
    print(f"✓ 共有 {len(classes)} 个类别")
    
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ 使用设备: {device}")
    
    # 定义模型架构
    class PageClassifier(nn.Module):
        def __init__(self, num_classes):
            super(PageClassifier, self).__init__()
            self.mobilenet = models.mobilenet_v2(weights=None)
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
    
    # 创建模型
    model = PageClassifier(len(classes))
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    print("✓ 模型加载成功")
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 测试数据集目录
    dataset_dir = Path('page_classifier_dataset_updated')
    
    if not dataset_dir.exists():
        print(f"❌ 数据集目录不存在: {dataset_dir}")
        return
    
    print(f"\n[2] 开始测试所有类型...")
    print("=" * 70)
    
    # 统计信息
    total_correct = 0
    total_tested = 0
    results = {}
    
    # 对每个类别进行测试
    for class_name in classes:
        class_dir = dataset_dir / class_name
        
        if not class_dir.exists():
            print(f"\n⚠️  跳过 {class_name}: 目录不存在")
            continue
        
        # 获取所有图片（只测试原图，不测试增强图）
        all_images = [f for f in os.listdir(class_dir) if f.endswith('.png') and '_aug_' not in f]
        
        if len(all_images) == 0:
            print(f"\n⚠️  跳过 {class_name}: 没有原始图片")
            continue
        
        # 随机选择最多10张图片测试
        test_images = random.sample(all_images, min(10, len(all_images)))
        
        correct = 0
        tested = len(test_images)
        
        print(f"\n测试 {class_name} ({tested}/{len(all_images)} 张图片)")
        print("-" * 70)
        
        for img_name in test_images:
            img_path = class_dir / img_name
            
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
                    confidence_pct = confidence.item() * 100
                
                # 检查是否正确
                if predicted_class == class_name:
                    correct += 1
                    print(f"  ✓ {img_name[:40]:40s} -> {predicted_class:20s} ({confidence_pct:5.2f}%)")
                else:
                    print(f"  ✗ {img_name[:40]:40s} -> {predicted_class:20s} ({confidence_pct:5.2f}%) [应为: {class_name}]")
            
            except Exception as e:
                print(f"  ✗ {img_name[:40]:40s} -> 错误: {e}")
                tested -= 1
        
        # 计算准确率
        if tested > 0:
            accuracy = (correct / tested) * 100
            results[class_name] = {
                'correct': correct,
                'tested': tested,
                'accuracy': accuracy
            }
            
            total_correct += correct
            total_tested += tested
            
            print(f"  准确率: {correct}/{tested} = {accuracy:.2f}%")
    
    # 打印总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    
    # 按准确率排序
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'])
    
    print("\n准确率从低到高:")
    for class_name, stats in sorted_results:
        accuracy = stats['accuracy']
        correct = stats['correct']
        tested = stats['tested']
        
        if accuracy == 100:
            status = "✅"
        elif accuracy >= 90:
            status = "⚠️ "
        else:
            status = "❌"
        
        print(f"  {status} {class_name:25s}: {correct:2d}/{tested:2d} = {accuracy:6.2f}%")
    
    # 总体准确率
    if total_tested > 0:
        overall_accuracy = (total_correct / total_tested) * 100
        print(f"\n总体准确率: {total_correct}/{total_tested} = {overall_accuracy:.2f}%")
        
        if overall_accuracy == 100:
            print("\n🎉 完美！所有类型都能正确识别！")
        elif overall_accuracy >= 95:
            print("\n✅ 优秀！准确率超过95%")
        elif overall_accuracy >= 90:
            print("\n⚠️  良好，但有改进空间")
        else:
            print("\n❌ 需要改进")


if __name__ == '__main__':
    test_all_page_types()
