"""
实时测试登录页识别
从设备截图并使用启动专用模型和YOLO模型进行识别对比
"""
import asyncio
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import json
import sys
import time

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adb_bridge import ADBBridge
from src.page_detector import PageState


# 定义模型架构（必须与训练时一致）
class PageClassifier(nn.Module):
    """页面分类器模型 - 使用MobileNetV3"""
    
    def __init__(self, num_classes):
        super(PageClassifier, self).__init__()
        self.mobilenet = models.mobilenet_v3_large(weights=None)
        in_features = self.mobilenet.classifier[0].in_features
        self.mobilenet.classifier = nn.Sequential(
            nn.Linear(in_features, 1280),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )
    
    def forward(self, x):
        return self.mobilenet(x)


def load_startup_model(model_path, classes_path, device):
    """加载启动专用模型"""
    # 加载类别
    with open(classes_path, 'r', encoding='utf-8') as f:
        classes = json.load(f)
    
    # 创建模型
    model = PageClassifier(num_classes=len(classes))
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model, classes


async def test_realtime():
    """实时测试"""
    print("\n" + "=" * 80)
    print("实时登录页识别测试")
    print("=" * 80)
    
    # 配置
    project_root = Path(__file__).parent.parent
    startup_model_path = project_root / "models" / "page_classifier_login_best.pth"
    startup_classes_path = project_root / "models" / "page_classes_login.json"
    
    # 检查文件
    if not startup_model_path.exists():
        print(f"❌ 登录模型文件不存在: {startup_model_path}")
        return
    if not startup_classes_path.exists():
        print(f"❌ 登录类别文件不存在: {startup_classes_path}")
        return
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  设备: {device}")
    if device.type == 'cuda':
        print(f"  • GPU: {torch.cuda.get_device_name(0)}")
    
    # 加载登录专用模型
    print("\n📦 加载登录专用模型（MobileNetV3）...")
    startup_model, startup_classes = load_startup_model(
        startup_model_path, startup_classes_path, device
    )
    print(f"  • 类别数: {len(startup_classes)}")
    print(f"  • 类别: {', '.join(startup_classes)}")
    
    # 初始化ADB
    print("\n🔌 初始化ADB连接...")
    adb = ADBBridge()
    
    # 使用MuMu模拟器的设备ID（端口16384-16389对应实例0-5）
    devices = [f"127.0.0.1:{16384 + i}" for i in range(6)]
    
    print(f"  • 将测试 {len(devices)} 个设备:")
    for i, device_id in enumerate(devices):
        print(f"    {i+1}. {device_id}")
    
    # 初始化通用分类器
    print("\n📦 初始化通用分类器...")
    from src.page_detector_dl import PageDetectorDL
    general_model_path = project_root / "models" / "page_classifier_pytorch_best.pth"
    general_classes_path = project_root / "models" / "page_classes.json"
    
    if not general_model_path.exists() or not general_classes_path.exists():
        print("❌ 通用分类器文件不存在")
        return
    
    general_detector = PageDetectorDL(
        adb,
        model_path=str(general_model_path),
        classes_path=str(general_classes_path),
        fallback_classifier=None
    )
    print("  • 通用分类器已加载")
    
    # 图片预处理（与训练时一致）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("\n" + "=" * 80)
    print("开始测试所有设备...")
    print("=" * 80)
    
    # 测试每个设备
    for device_idx, device_id in enumerate(devices):
        print(f"\n{'=' * 80}")
        print(f"测试设备 {device_idx + 1}/{len(devices)}: {device_id}")
        print(f"{'=' * 80}")
        
        # 1. 截图
        print("  📸 截取屏幕...")
        screenshot_path = f"temp_screenshot_device_{device_idx + 1}.png"
        success = await adb.screencap_to_file(device_id, screenshot_path)
        
        if not success:
            print("  ❌ 截图失败")
            continue
        
        print(f"  ✓ 截图已保存: {screenshot_path}")
        
        # 2. 使用登录专用模型识别
        print("\n  🔍 登录专用模型识别:")
        image = Image.open(screenshot_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = startup_model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            predicted_class = startup_classes[predicted_idx.item()]
            confidence_value = confidence.item()
        inference_time = (time.time() - start_time) * 1000
        
        print(f"    • 识别结果: {predicted_class}")
        print(f"    • 置信度: {confidence_value:.2%}")
        print(f"    • 推理耗时: {inference_time:.2f}ms")
        
        # 显示所有类别的概率（前3名）
        probs = probabilities[0].cpu().numpy()
        top3_indices = probs.argsort()[-3:][::-1]
        print(f"    • Top 3 预测:")
        for idx in top3_indices:
            print(f"      - {startup_classes[idx]}: {probs[idx]:.2%}")
        
        # 3. 使用通用分类器识别
        print("\n  🔍 通用分类器识别:")
        start_time = time.time()
        general_result = await general_detector.detect_page(device_id, use_cache=False)
        general_time = (time.time() - start_time) * 1000
        
        print(f"    • 识别结果: {general_result.state.value}")
        print(f"    • 置信度: {general_result.confidence:.2%}")
        print(f"    • 推理耗时: {general_time:.2f}ms")
        print(f"    • 是否缓存: {general_result.cached}")
        
        # 4. 对比结果
        print("\n  📊 结果对比:")
        login_is_login = predicted_class == "登录页"
        general_is_login = general_result.state == PageState.LOGIN
        
        if login_is_login and general_is_login:
            print("    ✅ 两个模型都识别为登录页")
        elif login_is_login and not general_is_login:
            print(f"    ⚠️  登录模型识别为登录页，通用分类器识别为 {general_result.state.value}")
        elif not login_is_login and general_is_login:
            print(f"    ⚠️  通用分类器识别为登录页，登录模型识别为 {predicted_class}")
        else:
            print(f"    ℹ️  两个模型都未识别为登录页")
            print(f"       登录模型: {predicted_class}")
            print(f"       通用分类器: {general_result.state.value}")
    
    print("\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)
    
    # 保留所有截图供检查
    print("\n📁 截图已保存:")
    for device_idx in range(len(devices)):
        screenshot_path = f"temp_screenshot_device_{device_idx + 1}.png"
        if Path(screenshot_path).exists():
            print(f"  • {screenshot_path}")


if __name__ == '__main__':
    asyncio.run(test_realtime())
