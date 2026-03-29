"""
直接检查应用目录下的所有文件
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
    
    print(f"✓ ADB路径: {adb_path}")
    
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
    data_path = f"/data/data/{package_name}"
    
    try:
        await adb.connect(device_id)
    except:
        pass
    
    print("\n" + "=" * 60)
    print("检查应用目录结构")
    print("=" * 60)
    
    # 先检查主目录
    print(f"\n检查目录: {data_path}")
    result = await adb.shell(device_id, f"su -c 'ls -la {data_path}'")
    print(result)
    
    # 检查 shared_prefs 目录
    print(f"\n检查目录: {data_path}/shared_prefs")
    result = await adb.shell(device_id, f"su -c 'ls -la {data_path}/shared_prefs'")
    print(result)
    
    # 检查 databases 目录
    print(f"\n检查目录: {data_path}/databases")
    result = await adb.shell(device_id, f"su -c 'ls -la {data_path}/databases'")
    print(result)
    
    # 检查 files 目录
    print(f"\n检查目录: {data_path}/files")
    result = await adb.shell(device_id, f"su -c 'ls -la {data_path}/files'")
    print(result)
    
    # 检查 cache 目录
    print(f"\n检查目录: {data_path}/cache")
    result = await adb.shell(device_id, f"su -c 'ls -la {data_path}/cache'")
    print(result)
    
    print("\n" + "=" * 60)
    print("完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
