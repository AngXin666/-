"""
测试页面分类器（不含广告页）
"""
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import json

def load_model():
    """加载页面分类器（不含广告页）"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载类别列表
    with open("models/page_classes_without_ad.json", "r", encoding="utf-8") as f:
        classes = json.load(f)
    
    # 创建模型
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
    
    # 加载权重
    model.load_state_dict(torch.load("models/page_classifier_without_ad_best.pth", map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, device, classes

def test_classifier():
    print("\n" + "=" * 60)
    print("测试页面分类器（不含广告页）")
    print("=" * 60)
    
    # 加载模型
    model, device, classes = load_model()
    print(f"✓ 已加载模型: models/page_classifier_without_ad_best.pth")
    print(f"✓ 类别数量: {len(classes)}")
    
    # 数据转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 测试启动页服务弹窗
    startup_popup_dir = Path("标注工具_完整独立版/training_data/启动页服务弹窗")
    if startup_popup_dir.exists():
        startup_images = list(startup_popup_dir.glob("*.png"))[:20]  # 测试20张
        
        print(f"\n测试类别: 启动页服务弹窗")
        print(f"图片数量: {len(startup_images)}")
        print("-" * 60)
        
        startup_correct = 0
        startup_errors = []
        
        for img_path in startup_images:
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(image_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_idx = output.argmax(1).item()
                predicted_class = classes[predicted_idx]
                confidence = probabilities[0][predicted_idx].item() * 100
            
            if predicted_class == "启动页服务弹窗":
                startup_correct += 1
            else:
                startup_errors.append((img_path.name, predicted_class, confidence))
        
        startup_acc = (startup_correct / len(startup_images)) * 100
        print(f"\n准确率: {startup_correct}/{len(startup_images)} = {startup_acc:.2f}%")
        
        if startup_errors:
            print(f"\n错误预测详情:")
            for name, pred, conf in startup_errors[:5]:
                print(f"  - {name}: 预测为 '{pred}' (置信度: {conf:.2f}%)")
    
    # 测试广告页（应该被识别为其他类型，但不能是这3个）
    ad_dir = Path("标注工具_完整独立版/training_data/广告页")
    if ad_dir.exists():
        ad_images = list(ad_dir.glob("*.png"))
        # 只测试原始图片
        ad_images = [f for f in ad_images if not f.stem.endswith('_aug') and '_aug_' not in f.name][:30]
        
        print(f"\n测试类别: 广告页（检查是否被误识别）")
        print(f"图片数量: {len(ad_images)}")
        print("-" * 60)
        
        forbidden_classes = ["启动页服务弹窗", "首页广告弹窗", "首页"]
        misidentified = []
        predictions = {}
        
        for img_path in ad_images:
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(image_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_idx = output.argmax(1).item()
                predicted_class = classes[predicted_idx]
                confidence = probabilities[0][predicted_idx].item() * 100
            
            # 统计预测类别
            if predicted_class not in predictions:
                predictions[predicted_class] = 0
            predictions[predicted_class] += 1
            
            # 检查是否被误识别为禁止的类别
            if predicted_class in forbidden_classes:
                misidentified.append((img_path.name, predicted_class, confidence))
        
        print(f"\n广告页被识别为的类别分布:")
        for pred_class, count in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(ad_images)) * 100
            marker = " ⚠️" if pred_class in forbidden_classes else ""
            print(f"  - {pred_class}: {count}张 ({percentage:.1f}%){marker}")
        
        if misidentified:
            print(f"\n⚠️ 警告: {len(misidentified)}张广告页被误识别为禁止类别:")
            for name, pred, conf in misidentified[:10]:
                print(f"  - {name}: 预测为 '{pred}' (置信度: {conf:.2f}%)")
        else:
            print(f"\n✅ 没有广告页被误识别为禁止类别")
    
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    if startup_popup_dir.exists():
        print(f"启动页服务弹窗识别准确率: {startup_acc:.2f}%")
    if ad_dir.exists():
        print(f"广告页误识别为禁止类别: {len(misidentified)}/{len(ad_images)} = {len(misidentified)/len(ad_images)*100:.2f}%")
        if len(misidentified) == 0:
            print("✅ 广告页不会被误识别为禁止类别，可以使用")
        else:
            print("⚠️ 广告页会被误识别，需要使用广告页二分类器")


if __name__ == '__main__':
    try:
        test_classifier()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
