"""
使用原始标注图测试页面分类器
Test Page Classifier with Original Annotated Images
"""

import sys
from pathlib import Path
import os
import json

# 添加src目录到路径
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


def test_page_classifier():
    """测试页面分类器"""
    print("=" * 70)
    print("使用原始标注图测试页面分类器")
    print("=" * 70)
    
    # 检查页面分类器模型文件
    print("\n[1] 检查页面分类器模型文件...")
    model_path = 'page_classifier_pytorch_best.pth'
    classes_path = 'page_classes.json'
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        # 尝试在models目录查找
        alt_model_path = os.path.join('models', model_path)
        if os.path.exists(alt_model_path):
            model_path = alt_model_path
            print(f"✓ 在models目录找到: {model_path}")
        else:
            return
    else:
        print(f"✓ 模型文件存在: {model_path}")
    
    if not os.path.exists(classes_path):
        print(f"❌ 类别文件不存在: {classes_path}")
        # 尝试在models目录查找
        alt_classes_path = os.path.join('models', classes_path)
        if os.path.exists(alt_classes_path):
            classes_path = alt_classes_path
            print(f"✓ 在models目录找到: {classes_path}")
        else:
            return
    else:
        print(f"✓ 类别文件存在: {classes_path}")
    
    # 加载类别列表
    print("\n[2] 加载类别列表...")
    with open(classes_path, 'r', encoding='utf-8') as f:
        classes = json.load(f)
    
    print(f"✓ 共有 {len(classes)} 个类别")
    
    # 检查是否包含"个人页_已登录"
    if '个人页_已登录' in classes:
        target_idx = classes.index('个人页_已登录')
        print(f"✓ 包含目标类别: '个人页_已登录' (索引: {target_idx})")
    else:
        print(f"❌ 不包含目标类别: '个人页_已登录'")
        return
    
    # 加载模型
    print("\n[3] 加载页面分类器模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
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
    
    try:
        num_classes = len(classes)
        model = PageClassifier(num_classes)
        
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ 模型加载成功（从checkpoint）")
        else:
            model.load_state_dict(checkpoint)
            print(f"✓ 模型加载成功（直接加载）")
        
        model = model.to(device)
        model.eval()
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 查找个人页已登录的测试图片
    print("\n[4] 查找个人页已登录的测试图片...")
    test_image_dirs = [
        '原始标注图/个人页_已登录_余额积分/images',
        '原始标注图/个人页_已登录_头像首页/images',
    ]
    
    test_images = []
    for img_dir in test_image_dirs:
        if os.path.exists(img_dir):
            images = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                # 取前3张图片
                for img_name in images[:3]:
                    test_images.append(os.path.join(img_dir, img_name))
    
    if not test_images:
        print(f"❌ 未找到测试图片")
        return
    
    print(f"✓ 找到 {len(test_images)} 张测试图片")
    
    # 预处理配置
    img_size = (224, 224)
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 测试每张图片
    print("\n[5] 开始测试...")
    print("=" * 70)
    
    correct_count = 0
    total_count = len(test_images)
    
    for i, img_path in enumerate(test_images, 1):
        print(f"\n测试图片 {i}/{total_count}: {os.path.basename(img_path)}")
        print("-" * 70)
        
        try:
            # 加载图像
            image = Image.open(img_path)
            
            # 转换为RGB
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            
            # 预处理
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            # 预测
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # 获取top-3预测结果
                top3_prob, top3_idx = torch.topk(probabilities, min(3, len(classes)))
                
                print(f"Top-3 预测结果:")
                for j in range(len(top3_idx[0])):
                    idx = top3_idx[0][j].item()
                    prob = top3_prob[0][j].item()
                    class_name = classes[idx]
                    
                    # 高亮显示目标类别
                    if class_name == '个人页_已登录':
                        print(f"  {j+1}. 【{class_name}】 - 置信度: {prob:.2%} ⭐")
                    else:
                        print(f"  {j+1}. {class_name} - 置信度: {prob:.2%}")
                
                # 检查最高预测是否是目标类别
                predicted_idx = top3_idx[0][0].item()
                predicted_class = classes[predicted_idx]
                predicted_prob = top3_prob[0][0].item()
                
                if predicted_class == '个人页_已登录':
                    print(f"\n✅ 正确识别为'个人页_已登录' (置信度: {predicted_prob:.2%})")
                    correct_count += 1
                else:
                    print(f"\n❌ 错误识别为'{predicted_class}' (置信度: {predicted_prob:.2%})")
                    
                    # 检查是否在top-3中
                    target_in_top3 = False
                    for j in range(len(top3_idx[0])):
                        if classes[top3_idx[0][j].item()] == '个人页_已登录':
                            target_in_top3 = True
                            target_rank = j + 1
                            target_prob = top3_prob[0][j].item()
                            print(f"  '个人页_已登录'在Top-{target_rank}中，置信度: {target_prob:.2%}")
                            break
                    
                    if not target_in_top3:
                        print(f"  '个人页_已登录'不在Top-3预测中")
        
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 总结
    print("\n" + "=" * 70)
    print(f"测试总结:")
    print(f"  - 总测试图片: {total_count}")
    print(f"  - 正确识别: {correct_count}")
    print(f"  - 准确率: {correct_count/total_count*100:.1f}%")
    print("=" * 70)
    
    if correct_count == total_count:
        print("\n✅ 所有测试通过：页面分类器能正确识别'个人页_已登录'")
    elif correct_count > 0:
        print(f"\n⚠️ 部分测试通过：{correct_count}/{total_count} 张图片被正确识别")
    else:
        print(f"\n❌ 所有测试失败：页面分类器无法识别'个人页_已登录'")


if __name__ == '__main__':
    test_page_classifier()
