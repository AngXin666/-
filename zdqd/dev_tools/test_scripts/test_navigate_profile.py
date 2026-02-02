"""
测试导航到个人页面功能
"""

import asyncio
import sys
from pathlib import Path

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.emulator_controller import EmulatorController
from src.adb_bridge import ADBBridge
from src.page_detector_hybrid import PageDetectorHybrid
from src.navigator import Navigator


async def test_navigate_to_profile():
    """测试导航到个人页面"""
    
    print("=" * 60)
    print("测试导航到个人页面功能")
    print("=" * 60)
    
    # 1. 自动检测模拟器
    print("\n1. 检测模拟器...")
    found_emulators = EmulatorController.detect_all_emulators()
    
    if not found_emulators:
        print("❌ 未检测到模拟器")
        return
    
    emulator_type, emulator_path = found_emulators[0]
    print(f"✓ 检测到模拟器: {emulator_path}")
    
    # 2. 初始化
    controller = EmulatorController(emulator_path)
    adb_path = controller.get_adb_path()
    adb_port = await controller.get_adb_port(0)
    device_id = f"127.0.0.1:{adb_port}"
    
    print(f"✓ ADB 路径: {adb_path}")
    print(f"✓ 设备 ID: {device_id}")
    
    adb = ADBBridge(adb_path)
    detector = PageDetectorHybrid(adb)
    navigator = Navigator(adb, detector)
    
    # 3. 连接设备
    print(f"\n2. 连接设备...")
    connected = await adb.connect(device_id)
    if not connected:
        print(f"❌ 连接失败")
        return
    print(f"✓ 连接成功")
    
    # 4. 检测当前页面
    print(f"\n3. 检测当前页面...")
    page_result = await detector.detect_page(device_id, use_ocr=True)
    if page_result and page_result.state:
        print(f"✓ 当前页面: {page_result.state.value}")
        print(f"  详细信息: {page_result.details}")
    else:
        print(f"❌ 无法检测页面")
        return
    
    # 5. 测试导航到个人页面
    print(f"\n4. 测试导航到个人页面...")
    print(f"   请确保溪盟商城应用已打开")
    print(f"   开始导航...\n")
    
    success = await navigator.navigate_to_profile(device_id, max_attempts=3)
    
    print(f"\n" + "=" * 60)
    if success:
        print(f"✅ 导航成功！")
    else:
        print(f"❌ 导航失败")
    print(f"=" * 60)
    
    # 6. 最终确认
    print(f"\n5. 最终确认...")
    page_result = await detector.detect_page(device_id, use_ocr=True)
    if page_result and page_result.state:
        print(f"最终页面: {page_result.state.value}")
        print(f"详细信息: {page_result.details}")


if __name__ == "__main__":
    try:
        asyncio.run(test_navigate_to_profile())
    except KeyboardInterrupt:
        print("\n\n测试被中断")
    except Exception as e:
        print(f"\n❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
