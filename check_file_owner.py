"""
检查文件所有者和权限
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
    
    # 使用实例5的设备
    device_id = "127.0.0.1:16544"
    print(f"✓ 使用设备: {device_id}")
    
    adb = ADBBridge(adb_path=adb_path)
    package_name = "com.ry.xmsc"
    data_path = f"/data/data/{package_name}"
    
    try:
        await adb.connect(device_id)
    except:
        pass
    
    print("\n" + "=" * 60)
    print("检查文件所有者和权限")
    print("=" * 60)
    
    # 检查关键文件的所有者和权限
    files_to_check = [
        "shared_prefs/lcdpr.xml",
        "databases/DCStorage",
        "databases/DCStorage-journal",
    ]
    
    for file_path in files_to_check:
        full_path = f"{data_path}/{file_path}"
        result = await adb.shell(device_id, f"su -c 'ls -l {full_path}'")
        print(f"\n{file_path}:")
        print(f"  {result.strip()}")
    
    # 获取应用的UID
    print("\n" + "=" * 60)
    print("获取应用UID")
    print("=" * 60)
    
    result = await adb.shell(device_id, f"su -c 'stat -c \"%u %g\" {data_path}'")
    print(f"\n应用目录的 UID:GID = {result.strip()}")
    
    # 检查ps中的应用进程
    result = await adb.shell(device_id, f"ps | grep {package_name}")
    print(f"\n应用进程信息:")
    print(result.strip())
    
    print("\n" + "=" * 60)
    print("完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
