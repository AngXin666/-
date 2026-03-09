"""
实时测试通用分类器 - 识别当前页面
"""
import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.adb_bridge import ADBBridge
from src.model_manager import ModelManager
from src.screen_capture import ScreenCapture
import torch
from torchvision import transforms
from PIL import Image
import json


async def main():
    """主函数"""
    print("=" * 60)
    print("实时测试通用分类器")
    print("=" * 60)
    
    # 初始化模拟器控制器和ADB
    from src.emulator_controller import EmulatorController
    controller = EmulatorController()
    adb_path = controller.get_adb_path()
    
    if not adb_path:
        print("❌ 无法找到 ADB 路径")
        return
    
    print(f"✓ ADB 路径: {adb_path}")
    
    adb = ADBBridge(adb_path=adb_path)
    
    # 使用设备ID
    device_id = "127.0.0.1:16448"
    print(f"✓ 使用设备: {device_id}\n")
    
    # 获取通用分类器
    model_manager = ModelManager.get_instance()
    
    # 加载所有模型
    print("正在加载模型...")
    stats = model_manager.initialize_all_models(
        adb_bridge=adb,
        log_callback=print
    )
    print(f"\n✓ 模型加载完成")
    print(f"  已加载模型: {stats['models_loaded']}")
    print(f"  加载时间: {stats['total_time']:.2f}秒\n")
    
    # 获取通用分类器
    general_classifier = model_manager.get_general_classifier()
    
    if not general_classifier:
        print("❌ 通用分类器未加载")
        return
    
    print("✓ 通用分类器已加载\n")
    
    # 截图
    print("正在截图...")
    screen_capture = ScreenCapture(adb)
    screenshot_np = await screen_capture.capture(device_id)
    
    if screenshot_np is None:
        print("❌ 截图失败")
        return
    
    # 转换为PIL图像
    from PIL import Image
    import cv2
    screenshot = Image.fromarray(cv2.cvtColor(screenshot_np, cv2.COLOR_BGR2RGB))
    
    # 保存截图
    screenshot_path = project_root / "test_general_classifier_screenshot.png"
    screenshot.save(screenshot_path)
    print(f"✓ 截图已保存: {screenshot_path}\n")
    
    # 使用通用分类器检测
    print("使用通用分类器检测页面...")
    
    # 加载类别
    classes_path = project_root / "models" / "page_classes.json"
    with open(classes_path, 'r', encoding='utf-8') as f:
        classes = json.load(f)
    
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 预处理图像
    image_tensor = transform(screenshot).unsqueeze(0)
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_tensor = image_tensor.to(device)
    general_classifier._model = general_classifier._model.to(device)
    general_classifier._model.eval()
    
    # 预测
    with torch.no_grad():
        output = general_classifier._model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = probabilities.max(1)
        predicted_class = classes[predicted.item()]
    
    print("\n" + "=" * 60)
    print("通用分类器检测结果:")
    print("=" * 60)
    print(f"页面状态: {predicted_class}")
    print(f"置信度: {confidence.item():.2%}")
    print("=" * 60)
    
    # 显示Top 5预测
    print("\nTop 5 预测:")
    top5_prob, top5_idx = probabilities[0].topk(5)
    for i, (prob, idx) in enumerate(zip(top5_prob, top5_idx), 1):
        print(f"  {i}. {classes[idx.item()]}: {prob.item():.2%}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
