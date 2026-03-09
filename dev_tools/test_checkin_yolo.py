"""
测试签到页YOLO模型检测
Test Checkin Page YOLO Model Detection
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.adb_bridge import ADBBridge
from src.model_manager import ModelManager


async def test_checkin_yolo():
    """测试签到页YOLO模型检测签到按钮"""
    
    print("=" * 60)
    print("测试签到页YOLO模型检测")
    print("=" * 60)
    
    # 1. 使用默认设备ID（MuMu模拟器）
    device_id = "127.0.0.1:16384"
    print(f"\n[1] 使用设备: {device_id}")
    
    # 2. 初始化ADB（使用MuMu的ADB路径）
    print("\n[2] 初始化ADB...")
    adb_path = r"D:\Program Files\Netease\MuMu\nx_device\12.0\shell\adb.exe"
    adb = ADBBridge(adb_path=adb_path)
    
    # [2026-03-02] 统一术语：获取YOLO识别器
    print("\n[3] 初始化模型管理器...")
    model_manager = ModelManager.get_instance()
    model_manager.initialize_all_models(adb)
    
    detector = model_manager.get_page_detector_integrated()
    
    if not detector:
        print("❌ 无法获取YOLO识别器")
        return
    
    print("✓ YOLO识别器已加载")
    
    # 3. 检测当前页面
    print("\n[3] 检测当前页面...")
    page_result = await detector.detect_page(device_id, use_cache=False, detect_elements=False)
    
    print(f"页面类型: {page_result.state.value}")
    print(f"置信度: {page_result.confidence:.2%}")
    print(f"详情: {page_result.details}")
    
    # 4. 如果不是签到页，提示用户
    if page_result.state.value != "checkin":
        print(f"\n⚠️ 当前不是签到页，是 {page_result.state.value}")
        print("请手动导航到签到页后重新运行此脚本")
        return
    
    print("\n✓ 当前在签到页")
    
    # 5. 使用YOLO检测签到按钮
    print("\n[4] 使用YOLO检测签到按钮...")
    detection_result = await detector.detect_page(device_id, use_cache=False, detect_elements=True)
    
    if not detection_result.elements:
        print("❌ 未检测到任何元素")
        print(f"使用的YOLO模型: {detection_result.yolo_model_used}")
        return
    
    print(f"✓ 检测到 {len(detection_result.elements)} 个元素")
    print(f"使用的YOLO模型: {detection_result.yolo_model_used}")
    
    # 6. 查找签到按钮
    print("\n[5] 查找签到按钮...")
    checkin_button = None
    
    for element in detection_result.elements:
        print(f"  - {element.class_name}: 置信度={element.confidence:.2%}, 中心点={element.center}")
        
        if '签到按钮' in element.class_name or '签到' in element.class_name:
            checkin_button = element
    
    if checkin_button:
        print(f"\n✓ 找到签到按钮:")
        print(f"  类别: {checkin_button.class_name}")
        print(f"  置信度: {checkin_button.confidence:.2%}")
        print(f"  中心点: {checkin_button.center}")
        print(f"  边界框: {checkin_button.bbox}")
    else:
        print("\n❌ 未找到签到按钮")
        print("检测到的元素中没有包含'签到按钮'或'签到'关键词")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_checkin_yolo())
