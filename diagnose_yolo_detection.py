"""
诊断YOLO检测失败的原因
"""

import asyncio
import sys
from pathlib import Path

async def diagnose_yolo():
    """诊断YOLO检测器状态"""
    
    print("=" * 60)
    print("YOLO检测器诊断")
    print("=" * 60)
    
    # 1. 检查依赖库
    print("\n[步骤1] 检查依赖库...")
    
    try:
        from PIL import Image
        print("  ✓ PIL已安装")
    except ImportError:
        print("  ✗ PIL未安装")
        return
    
    try:
        from ultralytics import YOLO
        print("  ✓ ultralytics已安装")
    except ImportError:
        print("  ✗ ultralytics未安装")
        print("  请运行: pip install ultralytics")
        return
    
    # 2. 检查模型注册表
    print("\n[步骤2] 检查模型注册表...")
    
    registry_path = Path("yolo_model_registry.json")
    if not registry_path.exists():
        print(f"  ✗ 注册表文件不存在: {registry_path}")
        return
    
    print(f"  ✓ 注册表文件存在: {registry_path}")
    
    import json
    with open(registry_path, 'r', encoding='utf-8') as f:
        registry = json.load(f)
    
    models = registry.get('models', {})
    print(f"  ✓ 注册表中有 {len(models)} 个模型")
    
    # 3. 检查profile_logged和balance模型
    print("\n[步骤3] 检查关键模型...")
    
    for model_name in ['profile_logged', 'balance']:
        if model_name in models:
            model_info = models[model_name]
            model_path = model_info.get('model_path')
            print(f"\n  模型: {model_name}")
            print(f"    路径: {model_path}")
            
            if model_path and Path(model_path).exists():
                print(f"    ✓ 模型文件存在")
                
                # 尝试加载模型
                try:
                    model = YOLO(model_path)
                    print(f"    ✓ 模型加载成功")
                    print(f"    类别: {model.names}")
                except Exception as e:
                    print(f"    ✗ 模型加载失败: {e}")
            else:
                print(f"    ✗ 模型文件不存在")
        else:
            print(f"\n  ✗ 模型 {model_name} 未在注册表中")
    
    # 4. 检查ADB连接
    print("\n[步骤4] 检查ADB连接...")
    
    from src.adb_bridge import ADBBridge
    adb = ADBBridge()
    device_id = "127.0.0.1:5555"
    
    devices = await adb.list_devices()
    print(f"  已连接设备: {devices}")
    
    if device_id not in devices:
        print(f"  ✗ 设备 {device_id} 未连接")
        return
    
    print(f"  ✓ 设备 {device_id} 已连接")
    
    # 5. 测试截图
    print("\n[步骤5] 测试截图...")
    
    screenshot_data = await adb.screencap(device_id)
    if not screenshot_data:
        print("  ✗ 截图失败")
        return
    
    print(f"  ✓ 截图成功，大小: {len(screenshot_data)} 字节")
    
    # 6. 测试YOLO检测
    print("\n[步骤6] 测试YOLO检测...")
    
    from src.yolo_button_detector import YoloDetector
    from io import BytesIO
    
    yolo_detector = YoloDetector(adb)
    
    # 测试profile_logged模型
    print("\n  测试 profile_logged 模型...")
    detections = await yolo_detector.detect(device_id, 'profile_logged', conf_threshold=0.5)
    
    if detections:
        print(f"  ✓ 检测到 {len(detections)} 个目标:")
        for det in detections:
            print(f"    - {det.class_name}: 置信度={det.confidence:.2f}, bbox={det.bbox}")
    else:
        print(f"  ✗ 未检测到任何目标")
        print(f"  可能原因:")
        print(f"    1. 当前页面不是个人页（已登录）")
        print(f"    2. 模型置信度阈值过高（当前0.5）")
        print(f"    3. 模型训练数据与当前页面差异较大")
    
    # 测试balance模型
    print("\n  测试 balance 模型...")
    detections = await yolo_detector.detect(device_id, 'balance', conf_threshold=0.5)
    
    if detections:
        print(f"  ✓ 检测到 {len(detections)} 个目标:")
        for det in detections:
            print(f"    - {det.class_name}: 置信度={det.confidence:.2f}, bbox={det.bbox}")
    else:
        print(f"  ✗ 未检测到任何目标")
        print(f"  可能原因:")
        print(f"    1. 当前页面不是个人页（已登录）")
        print(f"    2. 模型置信度阈值过高（当前0.5）")
        print(f"    3. 模型训练数据与当前页面差异较大")
    
    # 7. 测试降低置信度阈值
    print("\n[步骤7] 测试降低置信度阈值（0.3）...")
    
    print("\n  测试 profile_logged 模型（置信度0.3）...")
    detections = await yolo_detector.detect(device_id, 'profile_logged', conf_threshold=0.3)
    
    if detections:
        print(f"  ✓ 检测到 {len(detections)} 个目标:")
        for det in detections:
            print(f"    - {det.class_name}: 置信度={det.confidence:.2f}, bbox={det.bbox}")
    else:
        print(f"  ✗ 仍未检测到任何目标")
    
    print("\n  测试 balance 模型（置信度0.3）...")
    detections = await yolo_detector.detect(device_id, 'balance', conf_threshold=0.3)
    
    if detections:
        print(f"  ✓ 检测到 {len(detections)} 个目标:")
        for det in detections:
            print(f"    - {det.class_name}: 置信度={det.confidence:.2f}, bbox={det.bbox}")
    else:
        print(f"  ✗ 仍未检测到任何目标")
    
    # 8. 保存当前截图用于分析
    print("\n[步骤8] 保存当前截图...")
    
    from PIL import Image
    image = Image.open(BytesIO(screenshot_data))
    output_path = "debug_screenshot.png"
    image.save(output_path)
    print(f"  ✓ 截图已保存到: {output_path}")
    print(f"  请检查截图，确认是否在个人页（已登录）")
    
    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)

if __name__ == '__main__':
    asyncio.run(diagnose_yolo())
