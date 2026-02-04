"""
测试从个人页导航到首页
Test navigating from profile page to home page
"""

import asyncio
import sys
import subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.adb_bridge import ADBBridge
from src.model_manager import ModelManager
from src.navigator import Navigator


async def main():
    """主测试函数"""
    
    # ==================== 初始化设备 ====================
    adb_path = r"D:\Program Files\Netease\MuMu\nx_device\12.0\shell\adb.exe"
    adb = ADBBridge(adb_path)
    
    # 自动获取设备列表
    print("正在获取设备列表...")
    try:
        result = subprocess.run(
            [adb_path, "devices"], 
            capture_output=True, 
            text=True, 
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        lines = result.stdout.strip().split('\n')[1:]
        devices = [line.split('\t')[0] for line in lines if line.strip() and 'device' in line]
        
        if not devices:
            print("❌ 没有找到正在运行的设备")
            return
        
        device_id = devices[0]
        print(f"✓ 找到设备: {device_id}")
    except Exception as e:
        print(f"❌ 获取设备列表失败: {e}")
        return
    
    # ==================== 初始化模型 ====================
    print("\n正在初始化模型...")
    model_manager = ModelManager.get_instance()
    model_manager.initialize_all_models(adb)
    
    # 获取常用模型
    detector = model_manager.get_page_detector_integrated()
    
    print("✓ 模型初始化完成\n")
    
    # ==================== 测试导航 ====================
    print("=" * 60)
    print("测试从个人页导航到首页")
    print("=" * 60)
    
    # 创建导航器
    navigator = Navigator(adb, detector)
    
    # 检测当前页面
    print("\n[1] 检测当前页面...")
    page_result = await detector.detect_page(device_id, use_cache=False)
    if page_result:
        print(f"  当前页面: {page_result.state.value}")
        print(f"  置信度: {page_result.confidence:.2%}")
        print(f"  详情: {page_result.details}")
    else:
        print("  ❌ 无法检测页面")
        return
    
    # 导航到首页
    print("\n[2] 导航到首页...")
    print("  （查看详细日志请检查 logs/debug_20260204.log）")
    success = await navigator.navigate_to_home(device_id, max_attempts=3)
    
    if success:
        print("\n✓ 导航成功！")
    else:
        print("\n❌ 导航失败！")
        
        # 再次检测当前页面
        print("\n[3] 检测当前页面（导航失败后）...")
        page_result = await detector.detect_page(device_id, use_cache=False)
        if page_result:
            print(f"  当前页面: {page_result.state.value}")
            print(f"  置信度: {page_result.confidence:.2%}")
            print(f"  详情: {page_result.details}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
