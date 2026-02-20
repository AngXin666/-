"""测试页面分类器对广告页和启动页服务弹窗的识别准确率"""
import torch
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import sys

# 添加标注工具路径
sys.path.insert(0, '标注工具_完整独立版')

def load_model():
    """加载页面分类器模型"""
    model_path = Path("models/page_classifier_pytorch_best.pth")
    classes_path = Path("models/page_classes.json")
    
    if not model_path.exists():
        print(f"❌ 模型文件不存在: {model_path}")
        return None, None
    
    # 从page_classes.json加载类别列表
    if not classes_path.exists():
        print(f"❌ 类别文件不存在: {classes_path}")
        return None, None
    
    import json
    with open(classes_path, 'r', encoding='utf-8') as f:
        classes = json.load(f)
    
    if not classes:
        print(f"❌ 类别列表为空")
        return None, None
    
    # 加载checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 定义模型架构(与训练时相同)
    from torchvision import models
    import torch.nn as nn
    
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
    num_classes = len(classes)
    model = PageClassifier(num_classes)
    
    # 加载权重
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    print(f"✓ 已加载模型: {model_path}")
    print(f"✓ 类别数量: {len(classes)}")
    
    return model, classes

def test_images(model, classes, category_name, image_dir):
    """测试指定类别的图片"""
    # 图片预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_files = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
    
    if not image_files:
        print(f"  ⚠️ 没有找到图片")
        return
    
    print(f"\n测试类别: {category_name}")
    print(f"图片数量: {len(image_files)}")
    print("-" * 60)
    
    correct = 0
    wrong_predictions = []
    
    for img_path in image_files:
        try:
            # 加载图片
            image = Image.open(img_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0)
            
            # 预测
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                predicted_class = classes[predicted_idx.item()]
                confidence_value = confidence.item()
            
            # 判断是否正确
            is_correct = predicted_class == category_name
            
            if is_correct:
                correct += 1
                status = "✓"
            else:
                status = "✗"
                wrong_predictions.append({
                    'file': img_path.name,
                    'predicted': predicted_class,
                    'confidence': confidence_value
                })
            
            # 只显示错误的预测
            if not is_correct:
                print(f"  {status} {img_path.name}: 预测为 '{predicted_class}' (置信度: {confidence_value:.2%})")
        
        except Exception as e:
            print(f"  ✗ {img_path.name}: 处理失败 - {e}")
    
    # 统计结果
    accuracy = correct / len(image_files) * 100 if image_files else 0
    print(f"\n准确率: {correct}/{len(image_files)} = {accuracy:.2f}%")
    
    if wrong_predictions:
        print(f"\n错误预测详情:")
        for item in wrong_predictions:
            print(f"  - {item['file']}: 预测为 '{item['predicted']}' (置信度: {item['confidence']:.2%})")
    
    return accuracy, len(image_files), correct

def main():
    """主函数"""
    print("=" * 60)
    print("测试页面分类器: 广告页 vs 启动页服务弹窗")
    print("=" * 60)
    
    # 加载模型
    model, classes = load_model()
    if not model or not classes:
        return
    
    # 测试广告页
    ad_dir = Path("标注工具_完整独立版/training_data/广告页")
    if ad_dir.exists():
        ad_accuracy, ad_total, ad_correct = test_images(model, classes, "广告页", ad_dir)
    else:
        print(f"\n⚠️ 广告页目录不存在: {ad_dir}")
        ad_accuracy = 0
        ad_total = 0
        ad_correct = 0
    
    # 测试启动页服务弹窗
    startup_dir = Path("标注工具_完整独立版/training_data/启动页服务弹窗")
    if startup_dir.exists():
        startup_accuracy, startup_total, startup_correct = test_images(model, classes, "启动页服务弹窗", startup_dir)
    else:
        print(f"\n⚠️ 启动页服务弹窗目录不存在: {startup_dir}")
        startup_accuracy = 0
        startup_total = 0
        startup_correct = 0
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"广告页: {ad_correct}/{ad_total} = {ad_accuracy:.2f}%")
    print(f"启动页服务弹窗: {startup_correct}/{startup_total} = {startup_accuracy:.2f}%")
    
    if ad_accuracy < 90 or startup_accuracy < 90:
        print("\n⚠️ 警告: 识别准确率低于90%，建议重新训练模型或增加训练数据")
    else:
        print("\n✓ 识别准确率良好")

if __name__ == "__main__":
    main()
