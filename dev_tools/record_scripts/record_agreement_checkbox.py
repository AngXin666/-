"""
记录登录页面协议勾选框坐标
使用方法：运行脚本后，在登录页面点击协议勾选框，程序会记录坐标
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.adb_bridge import ADBBridge


async def record_agreement_checkbox(device_id: str = "127.0.0.1:5555"):
    """记录协议勾选框坐标"""
    
    adb = ADBBridge()
    
    # 连接设备
    print(f"连接设备: {device_id}")
    if not await adb.connect(device_id):
        print(f"❌ 无法连接到设备: {device_id}")
        return
    
    print(f"✓ 已连接到设备: {device_id}")
    
    print("\n" + "="*60)
    print("记录协议勾选框坐标")
    print("="*60)
    print("\n请确保：")
    print("1. 应用已启动并在登录页面")
    print("2. 可以看到协议勾选框")
    print("\n准备好后，请在模拟器中点击协议勾选框...")
    print("程序将记录您点击的坐标\n")
    
    input("按 Enter 键开始监听点击事件...")
    
    print("\n开始监听点击事件（持续10秒）...")
    print("请在模拟器中点击协议勾选框\n")
    
    # 监听点击事件
    for i in range(10):
        # 获取最近的触摸事件
        result = await adb.shell(device_id, "getevent -l -c 1")
        
        if "ABS_MT_POSITION_X" in result or "ABS_X" in result:
            print(f"[{i+1}/10] 检测到触摸事件")
            
            # 解析坐标
            lines = result.split('\n')
            x, y = None, None
            
            for line in lines:
                if "ABS_MT_POSITION_X" in line or "ABS_X" in line:
                    try:
                        # 提取十六进制值
                        hex_value = line.split()[-1]
                        x = int(hex_value, 16)
                    except:
                        pass
                elif "ABS_MT_POSITION_Y" in line or "ABS_Y" in line:
                    try:
                        hex_value = line.split()[-1]
                        y = int(hex_value, 16)
                    except:
                        pass
            
            if x is not None and y is not None:
                # 转换为屏幕坐标（假设分辨率 540x960）
                screen_x = int(x * 540 / 32768)
                screen_y = int(y * 960 / 32768)
                
                print(f"✓ 记录到坐标: ({screen_x}, {screen_y})")
                print(f"\n建议更新 src/auto_login.py 中的坐标：")
                print(f"'AGREEMENT_CHECK': ({screen_x}, {screen_y}),")
                break
        
        await asyncio.sleep(1)
    else:
        print("\n⚠️  未检测到点击事件")
        print("请确保：")
        print("1. 模拟器正在运行")
        print("2. 在登录页面")
        print("3. 在监听期间点击了协议勾选框")


if __name__ == "__main__":
    device_id = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:5555"
    
    print("="*60)
    print("协议勾选框坐标记录工具")
    print("="*60)
    print(f"设备ID: {device_id}")
    
    asyncio.run(record_agreement_checkbox(device_id))
