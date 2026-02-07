"""随机抽样测试页面分类器 - 每个类别10张图片"""
import sys
from pathlib import Path
import torch
from PIL import Image, ImageDraw, ImageFont
import json
import random
import os
import subprocess

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_random_samples():
    """随机抽样测试每个类别10张图片"""
    print("="*80)
    print("随机抽样测试页面分类器 - 每个类别10张图片")
    print("="*80)
    
    # 加载模型
    print("\n[步骤1] 加载页面分类器模型...")
    model_path = project_root / "标注工具_完整独立版" / "models" / "page_classifier_pytorch_best.pth"
    page_classes_path = project_root / "config" / "page_classes.json"
    training_data_dir = project_root / "标注工具_完整独立版" / "training_data"
    
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
    
    # 加载模型
    try:
        from torchvision import transforms, models
        import torch.nn as nn
        
        # 定义模型结构
        class PageClassifier(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
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
        
        # 加载模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = PageClassifier(num_classes=len(page_classes))
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        print(f"✓ 模型加载成功，使用设备: {device}")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 图片预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建输出目录
    output_dir = project_root / "标注工具_完整独立版" / "test_samples"
    output_dir.mkdir(exist_ok=True)
    
    # 清空输出目录
    for f in output_dir.glob("*.png"):
        f.unlink()
    
    print(f"\n[步骤2] 随机抽样并测试...")
    print(f"输出目录: {output_dir}")
    
    total_correct = 0
    total_tested = 0
    
    # 对每个类别随机抽样10张
    for class_idx, class_name in enumerate(page_classes):
        class_dir = training_data_dir / class_name
        
        if not class_dir.exists():
            print(f"\n[{class_idx+1}/{len(page_classes)}] ⚠️  类别目录不存在: {class_name}")
            continue
        
        # 获取所有图片（只要原始图片，不要增强图片）
        all_images = [f for f in class_dir.glob("*.png") if "_aug_" not in f.name]
        
        if len(all_images) == 0:
            print(f"\n[{class_idx+1}/{len(page_classes)}] ⚠️  没有找到图片: {class_name}")
            continue
        
        # 随机抽样10张（如果不足10张就全部测试）
        sample_count = min(10, len(all_images))
        sampled_images = random.sample(all_images, sample_count)
        
        print(f"\n[{class_idx+1}/{len(page_classes)}] 测试 {class_name} (抽样{sample_count}张)...")
        
        correct = 0
        for img_idx, img_path in enumerate(sampled_images, 1):
            try:
                # 加载并预测
                image = Image.open(img_path).convert('RGB')
                input_tensor = transform(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    _, predicted_idx = output.max(1)
                    predicted_class = page_classes[predicted_idx.item()]
                
                # 判断是否正确
                is_correct = (predicted_class == class_name)
                if is_correct:
                    correct += 1
                
                # 创建结果图片
                result_img = image.copy()
                draw = ImageDraw.Draw(result_img)
                
                # 添加文本（真实类别和预测类别）
                try:
                    font = ImageFont.truetype("msyh.ttc", 20)  # 微软雅黑
                except:
                    font = ImageFont.load_default()
                
                # 背景框
                text_bg_color = (0, 255, 0) if is_correct else (255, 0, 0)
                draw.rectangle([(0, 0), (result_img.width, 60)], fill=text_bg_color)
                
                # 文本
                draw.text((10, 10), f"真实: {class_name}", fill=(255, 255, 255), font=font)
                draw.text((10, 35), f"预测: {predicted_class}", fill=(255, 255, 255), font=font)
                
                # 保存
                output_filename = f"{class_idx+1:02d}_{class_name}_{img_idx:02d}_{'正确' if is_correct else '错误'}.png"
                result_img.save(output_dir / output_filename)
                
            except Exception as e:
                print(f"  ⚠️  处理失败 {img_path.name}: {e}")
        
        accuracy = (correct / sample_count) * 100
        print(f"  准确率: {accuracy:.1f}% ({correct}/{sample_count})")
        
        total_correct += correct
        total_tested += sample_count
    
    # 总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    print(f"总测试图片: {total_tested}张")
    print(f"正确识别: {total_correct}张")
    print(f"总准确率: {(total_correct/total_tested)*100:.2f}%")
    print(f"\n截图已保存到: {output_dir}")
    
    # 自动打开截图文件夹
    try:
        if os.name == 'nt':  # Windows
            os.startfile(output_dir)
        elif os.name == 'posix':  # macOS/Linux
            subprocess.run(['open' if sys.platform == 'darwin' else 'xdg-open', str(output_dir)])
        print("✓ 已自动打开截图文件夹")
    except Exception as e:
        print(f"⚠️  无法自动打开文件夹: {e}")
        print(f"请手动打开: {output_dir}")

if __name__ == '__main__':
    test_random_samples()
