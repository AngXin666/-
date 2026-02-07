"""测试页面分类器对所有页面类型训练图片的识别准确率"""
import sys
from pathlib import Path
import torch
from PIL import Image
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_all_page_types():
    """测试所有页面类型图片的识别"""
    print("="*80)
    print("测试页面分类器对所有页面类型的识别准确率")
    print("="*80)
    
    # 加载模型
    print("\n[步骤1] 加载页面分类器模型...")
    model_path = project_root / "models" / "page_classifier_pytorch_best.pth"
    page_classes_path = project_root / "config" / "page_classes.json"
    
    if not model_path.exists():
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    if not page_classes_path.exists():
        print(f"❌ 页面类型配置不存在: {page_classes_path}")
        return
    
    # 加载页面类型
    with open(page_classes_path, 'r', encoding='utf-8') as f:
        page_classes = json.load(f)
    
    print(f"✓ 页面类型数量: {len(page_classes)}")
    print(f"✓ 页面类型列表: {', '.join(page_classes)}")
    
    # 加载模型
    try:
        from torchvision import transforms
        import torch.nn as nn
        
        # 定义模型结构（需要与训练时一致）
        class PageClassifier(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                from torchvision.models import mobilenet_v2
                self.mobilenet = mobilenet_v2(weights=None)
                
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
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = PageClassifier(len(page_classes))
        
        # 加载checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"✓ 模型加载成功，使用设备: {device}")
        if 'epoch' in checkpoint:
            print(f"✓ 模型训练轮数: {checkpoint['epoch']}")
        if 'val_acc' in checkpoint:
            print(f"✓ 验证准确率: {checkpoint['val_acc']:.2%}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 定义图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 训练数据目录
    training_data_dir = project_root / "标注工具_完整独立版" / "training_data"
    
    # 测试所有页面类型
    print("\n" + "="*80)
    print("测试所有页面类型")
    print("="*80)
    
    results = {}
    total_images = 0
    total_correct = 0
    
    for class_idx, class_name in enumerate(page_classes):
        class_dir = training_data_dir / class_name
        
        if not class_dir.exists():
            print(f"\n⚠️  跳过 {class_name}: 目录不存在")
            continue
        
        images = list(class_dir.glob("*.png"))
        
        if len(images) == 0:
            print(f"\n⚠️  跳过 {class_name}: 没有图片")
            continue
        
        print(f"\n[{class_idx+1}/{len(page_classes)}] 测试 {class_name} ({len(images)}张图片)...")
        
        correct = 0
        wrong = []
        
        for img_path in images:
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(img_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    predicted_idx = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][predicted_idx].item()
                
                if predicted_idx == class_idx:
                    correct += 1
                else:
                    wrong.append({
                        'file': img_path.name,
                        'predicted': page_classes[predicted_idx],
                        'confidence': confidence
                    })
            except Exception as e:
                print(f"  ❌ 处理图片失败 {img_path.name}: {e}")
        
        accuracy = correct / len(images) * 100 if images else 0
        
        results[class_name] = {
            'total': len(images),
            'correct': correct,
            'accuracy': accuracy,
            'wrong': wrong
        }
        
        total_images += len(images)
        total_correct += correct
        
        # 显示结果
        if accuracy == 100:
            print(f"  ✓ 准确率: {accuracy:.2f}% ({correct}/{len(images)})")
        elif accuracy >= 80:
            print(f"  ⚠️  准确率: {accuracy:.2f}% ({correct}/{len(images)})")
        else:
            print(f"  ❌ 准确率: {accuracy:.2f}% ({correct}/{len(images)})")
        
        # 显示误判样例
        if wrong and len(wrong) <= 5:
            for item in wrong:
                print(f"    - {item['file']}: 识别为 {item['predicted']} (置信度: {item['confidence']:.2%})")
        elif wrong:
            print(f"    误判 {len(wrong)} 张，主要识别为:")
            # 统计误判的类别
            predicted_counts = {}
            for item in wrong:
                pred = item['predicted']
                predicted_counts[pred] = predicted_counts.get(pred, 0) + 1
            for pred, count in sorted(predicted_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"      - {pred}: {count}张")
    
    # 总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    
    total_accuracy = total_correct / total_images * 100 if total_images else 0
    
    print(f"\n总体识别结果:")
    print(f"  - 总图片数: {total_images}")
    print(f"  - 正确识别: {total_correct}")
    print(f"  - 总准确率: {total_accuracy:.2f}%")
    
    print(f"\n各类别准确率:")
    
    # 按准确率排序
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'])
    
    print(f"\n准确率最低的5个类别:")
    for class_name, result in sorted_results[:5]:
        print(f"  - {class_name}: {result['accuracy']:.2f}% ({result['correct']}/{result['total']})")
    
    print(f"\n准确率最高的5个类别:")
    for class_name, result in sorted_results[-5:]:
        print(f"  - {class_name}: {result['accuracy']:.2f}% ({result['correct']}/{result['total']})")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    test_all_page_types()
