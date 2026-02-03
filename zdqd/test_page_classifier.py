"""
测试页面分类器 - 检查是否能正确识别个人已登录页
Test Page Classifier - Check if it can correctly identify logged-in profile page
"""

import asyncio
import sys
from pathlib import Path
from io import BytesIO

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("❌ PIL未安装")

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms, models
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("❌ PyTorch未安装")

from src.adb_bridge import ADBBridge
from src.page_detector_integrated import PageDetectorIntegrated
import json
import os


async def test_page_classifier():
    """测试页面分类器"""
    print("=" * 70)
    print("测试页面分类器 - 个人已登录页识别")
    print("=" * 70)
    
    if not HAS_PIL or not HAS_TORCH:
        print("\n❌ 缺少必要的库（PIL或PyTorch）")
        return
    
    # 初始化ADB
    print("\n[1] 初始化ADB...")
    adb = ADBBridge()
    
    # 获取设备列表
    devices = await adb.list_devices()
    if not devices:
        print("❌ 未找到设备")
        return
    
    device_id = devices[0]
    print(f"✓ 使用设备: {device_id}")
    
    # 检查页面分类器模型文件
    print("\n[2] 检查页面分类器模型文件...")
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
    print("\n[3] 加载类别列表...")
    with open(classes_path, 'r', encoding='utf-8') as f:
        classes = json.load(f)
    
    print(f"✓ 共有 {len(classes)} 个类别")
    print(f"类别列表: {classes[:5]}..." if len(classes) > 5 else f"类别列表: {classes}")
    
    # 检查是否包含"个人页_已登录"
    if '个人页_已登录' in classes:
        print(f"✓ 包含目标类别: '个人页_已登录' (索引: {classes.index('个人页_已登录')})")
    else:
        print(f"❌ 不包含目标类别: '个人页_已登录'")
        print(f"可用的个人页相关类别:")
        for cls in classes:
            if '个人' in cls or '登录' in cls:
                print(f"  - {cls}")
    
    # 加载模型
    print("\n[4] 加载页面分类器模型...")
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
    
    # 截图并分类
    print("\n[5] 截图并进行页面分类...")
    screenshot_data = await adb.screencap(device_id)
    if not screenshot_data:
        print("❌ 截图失败")
        return
    
    print(f"✓ 截图成功 ({len(screenshot_data)} 字节)")
    
    # 转换为PIL图像
    image = Image.open(BytesIO(screenshot_data))
    print(f"✓ 图像尺寸: {image.size}, 模式: {image.mode}")
    
    # 转换为RGB
    if image.mode == 'RGBA':
        image = image.convert('RGB')
        print(f"✓ 已转换为RGB模式")
    
    # 预处理
    img_size = (224, 224)
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    print(f"✓ 图像预处理完成，张量形状: {image_tensor.shape}")
    
    # 预测
    print("\n[6] 执行分类预测...")
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # 获取top-5预测结果
        top5_prob, top5_idx = torch.topk(probabilities, min(5, len(classes)))
        
        print(f"\nTop-5 预测结果:")
        print("-" * 70)
        for i in range(len(top5_idx[0])):
            idx = top5_idx[0][i].item()
            prob = top5_prob[0][i].item()
            class_name = classes[idx]
            
            # 高亮显示目标类别
            if class_name == '个人页_已登录':
                print(f"  {i+1}. 【{class_name}】 - 置信度: {prob:.2%} ⭐")
            else:
                print(f"  {i+1}. {class_name} - 置信度: {prob:.2%}")
        
        # 检查最高预测是否是目标类别
        predicted_idx = top5_idx[0][0].item()
        predicted_class = classes[predicted_idx]
        predicted_prob = top5_prob[0][0].item()
        
        print("\n" + "=" * 70)
        print(f"最终预测: {predicted_class} (置信度: {predicted_prob:.2%})")
        print("=" * 70)
        
        if predicted_class == '个人页_已登录':
            print("\n✅ 测试通过：页面分类器正确识别为'个人页_已登录'")
        else:
            print(f"\n⚠️ 测试警告：页面分类器识别为'{predicted_class}'，而非'个人页_已登录'")
            
            # 检查是否在top-5中
            target_in_top5 = False
            target_rank = -1
            for i in range(len(top5_idx[0])):
                if classes[top5_idx[0][i].item()] == '个人页_已登录':
                    target_in_top5 = True
                    target_rank = i + 1
                    target_prob = top5_prob[0][i].item()
                    break
            
            if target_in_top5:
                print(f"  '个人页_已登录'在Top-{target_rank}中，置信度: {target_prob:.2%}")
            else:
                print(f"  '个人页_已登录'不在Top-5预测中")
    
    # 使用整合检测器测试
    print("\n[7] 使用整合检测器测试...")
    detector = PageDetectorIntegrated(adb)
    
    result = await detector.detect_page(device_id, use_cache=False, detect_elements=False)
    
    print(f"\n整合检测器结果:")
    print(f"  - 页面状态: {result.state}")
    print(f"  - 中文名称: {result.state.chinese_name}")
    print(f"  - 置信度: {result.confidence:.2%}")
    print(f"  - 详情: {result.details}")
    
    if result.state.chinese_name == '个人页_已登录' or '个人页_已登录' in result.details:
        print("\n✅ 整合检测器也识别为'个人页_已登录'")
    else:
        print(f"\n⚠️ 整合检测器识别为'{result.state.chinese_name}'")


if __name__ == '__main__':
    asyncio.run(test_page_classifier())
