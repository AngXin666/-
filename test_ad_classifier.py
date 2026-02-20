"""
测试广告页二分类器
"""
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path

def load_model():
    """加载广告页二分类器"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    
    # 加载权重
    model.load_state_dict(torch.load("models/ad_classifier_best.pth", map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, device

def test_ad_classifier():
    print("\n" + "=" * 60)
    print("测试广告页二分类器")
    print("=" * 60)
    
    # 加载模型
    model, device = load_model()
    print(f"✓ 已加载模型: models/ad_classifier_best.pth")
    
    # 数据转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 测试广告页
    ad_dir = Path("标注工具_完整独立版/training_data/广告页")
    ad_images = list(ad_dir.glob("*.png"))
    
    print(f"\n测试类别: 广告页")
    print(f"图片数量: {len(ad_images)}")
    print("-" * 60)
    
    ad_correct = 0
    ad_errors = []
    
    for img_path in ad_images:
        image = Image.open(img_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted = output.argmax(1).item()
            confidence = probabilities[0][predicted].item() * 100
        
        if predicted == 1:  # 1 = 广告页
            ad_correct += 1
        else:
            ad_errors.append((img_path.name, confidence))
    
    ad_acc = (ad_correct / len(ad_images)) * 100
    print(f"\n准确率: {ad_correct}/{len(ad_images)} = {ad_acc:.2f}%")
    
    if ad_errors:
        print(f"\n错误预测详情 (前10个):")
        for name, conf in ad_errors[:10]:
            print(f"  - {name}: 预测为 '非广告页' (置信度: {conf:.2f}%)")
    
    # 测试非广告页（随机抽样）
    print(f"\n测试类别: 非广告页 (随机抽样100张)")
    print("-" * 60)
    
    training_data_dir = Path("标注工具_完整独立版/training_data")
    non_ad_images = []
    
    for category_dir in training_data_dir.iterdir():
        if not category_dir.is_dir() or category_dir.name == "广告页":
            continue
        
        images = list(category_dir.glob("*.png"))
        non_ad_images.extend(images[:5])  # 每个类别取5张
        
        if len(non_ad_images) >= 100:
            break
    
    non_ad_images = non_ad_images[:100]
    
    non_ad_correct = 0
    non_ad_errors = []
    
    for img_path in non_ad_images:
        image = Image.open(img_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted = output.argmax(1).item()
            confidence = probabilities[0][predicted].item() * 100
        
        if predicted == 0:  # 0 = 非广告页
            non_ad_correct += 1
        else:
            non_ad_errors.append((img_path.name, confidence))
    
    non_ad_acc = (non_ad_correct / len(non_ad_images)) * 100
    print(f"\n准确率: {non_ad_correct}/{len(non_ad_images)} = {non_ad_acc:.2f}%")
    
    if non_ad_errors:
        print(f"\n错误预测详情 (前10个):")
        for name, conf in non_ad_errors[:10]:
            print(f"  - {name}: 预测为 '广告页' (置信度: {conf:.2f}%)")
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"广告页: {ad_correct}/{len(ad_images)} = {ad_acc:.2f}%")
    print(f"非广告页: {non_ad_correct}/{len(non_ad_images)} = {non_ad_acc:.2f}%")
    
    if ad_acc < 90 or non_ad_acc < 90:
        print("\n⚠️ 警告: 识别准确率低于90%")
    else:
        print("\n✅ 识别准确率良好")


if __name__ == '__main__':
    try:
        test_ad_classifier()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
