"""
检查 lcdpr.xml 文件内容
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.adb_bridge import ADBBridge
from src.emulator_controller import EmulatorController


async def main():
    """主函数"""
    controller = EmulatorController()
    adb_path = controller.get_adb_path()
    
    if not adb_path:
        print("❌ 未找到ADB路径")
        return
    
    # 检测设备
    import subprocess
    result = subprocess.run([adb_path, "devices"], capture_output=True, text=True)
    
    devices = []
    for line in result.stdout.strip().split('\n')[1:]:
        if line.strip() and '\tdevice' in line:
            device_id = line.split('\t')[0]
            devices.append(device_id)
    
    if not devices:
        print("❌ 未找到ADB设备")
        return
    
    device_id = devices[0]
    print(f"✓ 使用设备: {device_id}")
    
    adb = ADBBridge(adb_path=adb_path)
    package_name = "com.ry.xmsc"
    
    try:
        await adb.connect(device_id)
    except:
        pass
    
    print("\n" + "=" * 60)
    print("查看 lcdpr.xml 文件内容")
    print("=" * 60)
    
    # 读取文件内容
    file_path = f"/data/data/{package_name}/shared_prefs/lcdpr.xml"
    result = await adb.shell(device_id, f"su -c 'cat {file_path}'")
    
    print("\n文件内容:")
    print(result)
    
    print("\n" + "=" * 60)
    print("完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
