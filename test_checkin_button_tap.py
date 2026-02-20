"""测试签到按钮点击"""
import asyncio
import sys
sys.path.insert(0, 'src')

from adb_bridge import ADBBridge

async def test_tap():
    """测试点击签到按钮"""
    # 使用MuMu模拟器的ADB路径
    adb_path = r"D:\Program Files\Netease\MuMu\nx_device\12.0\shell\adb.exe"
    adb = ADBBridge(adb_path=adb_path)
    
    # 使用固定的设备ID
    device_id = "127.0.0.1:16384"
    print(f"使用设备: {device_id}")
    
    # 连接设备
    print("连接设备...")
    connected = await adb.connect(device_id)
    if not connected:
        print("❌ 设备连接失败")
        return
    print("✓ 设备已连接")
    
    # 测试坐标
    CHECKIN_BUTTON = (475, 550)
    
    print(f"\n准备点击坐标: {CHECKIN_BUTTON}")
    print("3秒后点击...")
    await asyncio.sleep(3)
    
    # 点击
    print(f"点击 {CHECKIN_BUTTON}")
    await adb.tap(device_id, *CHECKIN_BUTTON)
    
    print("✓ 点击完成")

if __name__ == "__main__":
    asyncio.run(test_tap())
