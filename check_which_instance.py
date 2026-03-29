"""
检查每个实例的应用状态
找出哪个实例是未登录的
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.adb_bridge import ADBBridge
from src.emulator_controller import EmulatorController


async def check_login_status(adb, device_id, package_name):
    """检查登录状态"""
    data_path = f"/data/data/{package_name}"
    
    # 检查关键文件
    lcdpr_result = await adb.shell(device_id, f"su -c 'stat -c \"%s\" {data_path}/shared_prefs/lcdpr.xml 2>/dev/null'")
    db_result = await adb.shell(device_id, f"su -c 'stat -c \"%s\" {data_path}/databases/DCStorage 2>/dev/null'")
    
    lcdpr_size = 0
    db_size = 0
    
    try:
        lcdpr_size = int(lcdpr_result.strip())
    except:
        pass
    
    try:
        db_size = int(db_result.strip())
    except:
        pass
    
    # 检查应用是否在运行
    ps_result = await adb.shell(device_id, f"ps | grep {package_name}")
    is_running = package_name in ps_result
    
    return {
        'lcdpr_size': lcdpr_size,
        'db_size': db_size,
        'is_running': is_running
    }


async def main():
    """主函数"""
    controller = EmulatorController()
    adb_path = controller.get_adb_path()
    
    if not adb_path:
        print("❌ 未找到ADB路径")
        return
    
    print(f"✓ ADB路径: {adb_path}")
    
    # 检测所有设备
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
    
    print(f"\n找到 {len(devices)} 个设备\n")
    
    adb = ADBBridge(adb_path=adb_path)
    package_name = "com.ry.xmsc"
    
    print("=" * 60)
    print("检查每个实例的状态")
    print("=" * 60)
    
    for i, device_id in enumerate(devices, 1):
        print(f"\n设备 {i}: {device_id}")
        
        try:
            await adb.connect(device_id)
        except:
            pass
        
        status = await check_login_status(adb, device_id, package_name)
        
        print(f"  lcdpr.xml: {status['lcdpr_size']} 字节")
        print(f"  DCStorage: {status['db_size']} 字节")
        print(f"  应用运行: {'是' if status['is_running'] else '否'}")
        
        # 判断登录状态
        if status['lcdpr_size'] > 0 and status['db_size'] > 0:
            print(f"  状态: 可能已登录")
        else:
            print(f"  状态: 未登录或数据异常")
    
    print("\n" + "=" * 60)
    print("完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
