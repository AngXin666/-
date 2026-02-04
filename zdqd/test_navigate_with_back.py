"""测试导航到个人页（点击"我的"按钮后按2次返回键）
使用方法：直接运行此脚本，会自动检测设备并测试导航功能
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
    
    print("=" * 70)
    print("测试：导航到个人页（点击'我的'按钮后按2次返回键）")
    print("=" * 70)
    
    # ==================== 初始化设备 ====================
    adb_path = r"D:\Program Files\Netease\MuMu\nx_device\12.0\shell\adb.exe"
    adb = ADBBridge(adb_path)
    
    # 自动获取设备列表
    print("\n[1] 正在获取设备列表...")
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
    print("\n[2] 正在初始化模型...")
    model_manager = ModelManager.get_instance()
    model_manager.initialize_all_models(adb)
    
    # 获取整合检测器
    integrated_detector = model_manager.get_page_detector_integrated()
    
    print("✓ 模型初始化完成\n")
    
    # ==================== 初始化Navigator ====================
    print("[3] 初始化Navigator...")
    navigator = Navigator(adb, integrated_detector)
    print("✓ Navigator初始化完成\n")
    
    # ==================== 测试导航到个人页 ====================
    print("=" * 70)
    print("[4] 开始测试导航到个人页...")
    print("=" * 70)
    
    try:
        success = await navigator.navigate_to_profile(device_id, max_attempts=3)
        
        print("\n" + "=" * 70)
        if success:
            print("✓ 测试成功：成功导航到个人页")
        else:
            print("✗ 测试失败：无法导航到个人页")
        print("=" * 70)
        
        # 验证最终页面状态
        print("\n[5] 验证最终页面状态...")
        page_result = await integrated_detector.detect_page(device_id, use_cache=False)
        if page_result:
            print(f"  当前页面: {page_result.state.value}")
            print(f"  置信度: {page_result.confidence:.2%}")
            print(f"  详情: {page_result.details}")
        else:
            print("  ⚠️ 无法检测页面状态")
        
        print("\n✓ 测试完成")
        
    except Exception as e:
        print(f"\n❌ 导航测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    asyncio.run(main())
