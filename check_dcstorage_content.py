"""
检查 DCStorage 数据库内容
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
    print("查看 DCStorage 数据库结构和内容")
    print("=" * 60)
    
    db_path = f"/data/data/{package_name}/databases/DCStorage"
    
    # 查看表结构
    print("\n1. 查看所有表:")
    result = await adb.shell(device_id, f"su -c 'sqlite3 {db_path} \".tables\"'")
    print(result)
    
    # 查看每个表的内容
    tables = result.strip().split()
    for table in tables:
        if table:
            print(f"\n2. 查看表 {table} 的结构:")
            result = await adb.shell(device_id, f"su -c 'sqlite3 {db_path} \".schema {table}\"'")
            print(result)
            
            print(f"\n3. 查看表 {table} 的内容 (前10行):")
            result = await adb.shell(device_id, f"su -c 'sqlite3 {db_path} \"SELECT * FROM {table} LIMIT 10\"'")
            print(result)
    
    print("\n" + "=" * 60)
    print("完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
