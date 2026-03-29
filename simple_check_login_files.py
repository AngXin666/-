"""
简单检查：对比当前应用目录和缓存目录的文件
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
    data_path = f"/data/data/{package_name}"
    
    try:
        await adb.connect(device_id)
    except:
        pass
    
    print("\n" + "=" * 60)
    print("检查当前保存的文件是否足够")
    print("=" * 60)
    
    # 当前保存的文件
    saved_files = [
        "shared_prefs/lcdpr.xml",
        "databases/DCStorage"
    ]
    
    print("\n当前保存的文件:")
    for file_path in saved_files:
        full_path = f"{data_path}/{file_path}"
        result = await adb.shell(device_id, f"su -c 'ls -lh {full_path}'")
        print(f"  {file_path}")
        print(f"    {result.strip()}")
    
    print("\n" + "=" * 60)
    print("可能遗漏的重要文件")
    print("=" * 60)
    
    # 检查其他可能重要的文件
    potential_files = [
        "shared_prefs/com.ry.xmsc_preferences.xml",  # 应用配置
        "shared_prefs/ContextData.xml",  # 上下文数据
        "shared_prefs/virtualImeiAndImsi.xml",  # 虚拟IMEI
        "shared_prefs/vkeyid_settings.xml",  # VKey设置
        "files/.DC4278477faeb9.txt",  # DC文件
    ]
    
    print("\n检查其他可能重要的文件:")
    for file_path in potential_files:
        full_path = f"{data_path}/{file_path}"
        result = await adb.shell(device_id, f"su -c 'ls -lh {full_path} 2>/dev/null || echo 不存在'")
        if "不存在" not in result:
            print(f"  ✓ {file_path}")
            print(f"    {result.strip()}")
        else:
            print(f"  ✗ {file_path} (不存在)")
    
    print("\n" + "=" * 60)
    print("建议")
    print("=" * 60)
    print("\n根据检查结果，建议:")
    print("1. 如果只保存 lcdpr.xml 和 DCStorage 就能恢复登录状态")
    print("   → 说明恢复逻辑有问题，需要检查恢复代码")
    print("\n2. 如果需要保存更多文件才能恢复登录状态")
    print("   → 需要更新 CACHE_FILES 列表，添加遗漏的文件")
    
    print("\n" + "=" * 60)
    print("完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
