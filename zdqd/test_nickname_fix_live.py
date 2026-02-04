"""测试昵称识别修复 - 真实设备测试"""
import asyncio
import sys
import subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.adb_bridge import ADBBridge
from src.model_manager import ModelManager
from src.profile_reader import ProfileReader

async def main():
    print("=" * 70)
    print("测试昵称识别修复 - 真实设备")
    print("=" * 70)
    
    # 初始化ADB
    adb_path = r"D:\Program Files\Netease\MuMu\nx_device\12.0\shell\adb.exe"
    adb = ADBBridge(adb_path)
    
    # 获取设备
    result = subprocess.run(
        [adb_path, "devices"], 
        capture_output=True, 
        text=True, 
        creationflags=subprocess.CREATE_NO_WINDOW
    )
    lines = result.stdout.strip().split('\n')[1:]
    devices = [line.split('\t')[0] for line in lines if line.strip() and 'device' in line]
    
    if not devices:
        print("❌ 没有找到设备")
        return
    
    device_id = devices[0]
    print(f"✓ 设备: {device_id}\n")
    
    # 初始化模型
    print("初始化模型...")
    mm = ModelManager.get_instance()
    mm.initialize_all_models(adb)
    detector = mm.get_page_detector_integrated()
    print("✓ 模型初始化完成\n")
    
    # 创建ProfileReader
    reader = ProfileReader(adb, detector)
    
    # 测试昵称识别
    print("=" * 70)
    print("开始识别昵称...")
    print("=" * 70)
    
    result = await reader.get_identity_only(device_id)
    
    print("\n" + "=" * 70)
    print("识别结果:")
    print("=" * 70)
    print(f"  昵称: {result.get('nickname')}")
    print(f"  用户ID: {result.get('user_id')}")
    print("=" * 70)

if __name__ == '__main__':
    asyncio.run(main())
