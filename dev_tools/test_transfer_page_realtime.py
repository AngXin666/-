"""
实时测试转账页面识别
从设备截图并使用转账专用模型进行识别
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
sys.path.insert(0, str(Path(__file__).parent))

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


def load_transfer_model(model_path, classes_path, device):
    """加载转账专用模型"""
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
    print("实时转账页面识别测试")
    print("=" * 80)
    
    # 配置
    project_root = Path(__file__).parent
    transfer_model_path = project_root / "标注工具_完整独立版" / "models" / "page_classifier_transfer_best.pth"
    transfer_classes_path = project_root / "标注工具_完整独立版" / "models" / "page_classes_transfer.json"
    
    # 检查文件
    if not transfer_model_path.exists():
        print(f"❌ 转账模型文件不存在: {transfer_model_path}")
        return
    if not transfer_classes_path.exists():
        print(f"❌ 转账类别文件不存在: {transfer_classes_path}")
        return
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  设备: {device}")
    if device.type == 'cuda':
        print(f"  • GPU: {torch.cuda.get_device_name(0)}")
    
    # 加载转账专用模型
    print("\n📦 加载转账专用模型（MobileNetV3）...")
    transfer_model, transfer_classes = load_transfer_model(
        transfer_model_path, transfer_classes_path, device
    )
    print(f"  • 类别数: {len(transfer_classes)}")
    print(f"  • 类别: {', '.join(transfer_classes)}")
    
    # 初始化ADB
    print("\n🔌 初始化ADB连接...")
    # 使用MuMu模拟器的ADB路径
    adb_path = r"D:\Program Files\Netease\MuMu\nx_device\12.0\shell\adb.exe"
    adb = ADBBridge(adb_path=adb_path)
    
    # 使用默认设备ID（MuMu模拟器）
    device_id = "127.0.0.1:16544"
    print(f"  • 设备ID: {device_id}")
    
    # 连接设备
    print(f"  • 正在连接设备...")
    await adb.connect(device_id)
    print(f"  ✓ 设备已连接")
    
    # 图片预处理（与训练时一致）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("\n" + "=" * 80)
    print("开始测试（连续识别5次，每次间隔1秒）")
    print("=" * 80)
    
    # 测试5次，每次间隔1秒
    for i in range(5):
        print(f"\n第 {i+1} 次测试:")
        print("-" * 80)
        
        # 1. 截图
        print("  📸 截取屏幕...")
        screenshot_path = f"temp_transfer_screenshot_{i+1}.png"
        success = await adb.screencap_to_file(device_id, screenshot_path)
        
        if not success:
            print("  ❌ 截图失败")
            continue
        
        print(f"  ✓ 截图已保存: {screenshot_path}")
        
        # 2. 使用转账专用模型识别
        print("\n  🔍 转账专用模型识别:")
        image = Image.open(screenshot_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = transfer_model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            predicted_class = transfer_classes[predicted_idx.item()]
            confidence_value = confidence.item()
        inference_time = (time.time() - start_time) * 1000
        
        print(f"    • 识别结果: {predicted_class}")
        print(f"    • 置信度: {confidence_value:.2%}")
        print(f"    • 推理耗时: {inference_time:.2f}ms")
        
        # 显示所有类别的概率（前5名）
        probs = probabilities[0].cpu().numpy()
        top5_indices = probs.argsort()[-5:][::-1]
        print(f"    • Top 5 预测:")
        for idx in top5_indices:
            print(f"      - {transfer_classes[idx]}: {probs[idx]:.2%}")
        
        # 等待1秒
        if i < 4:
            await asyncio.sleep(1)
    
    print("\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)
    
    # 保留第一张截图供检查
    print("\n📁 保留第一张截图: temp_transfer_screenshot_1.png")
    print("  • 请检查截图确认当前页面")
    
    # 清理其他临时文件
    print("\n🧹 清理其他临时文件...")
    for i in range(2, 6):
        screenshot_path = Path(f"temp_transfer_screenshot_{i}.png")
        if screenshot_path.exists():
            screenshot_path.unlink()
    print("  ✓ 清理完成")


if __name__ == '__main__':
    asyncio.run(test_realtime())
