"""
找出实例5对应的ADB设备
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.adb_bridge import ADBBridge
from src.emulator_controller import EmulatorController


async def get_device_name(adb, device_id):
    """获取设备名称"""
    result = await adb.shell(device_id, "getprop ro.product.model")
    return result.strip()


async def main():
    """主函数"""
    controller = EmulatorController()
    adb_path = controller.get_adb_path()
    
    if not adb_path:
        print("❌ 未找到ADB路径")
        return
    
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
    
    print(f"找到 {len(devices)} 个设备\n")
    
    adb = ADBBridge(adb_path=adb_path)
    
    print("=" * 60)
    print("检查每个设备")
    print("=" * 60)
    
    # MuMu模拟器的端口规则：16384 + 实例编号*32
    # 实例0: 16384
    # 实例1: 16416
    # 实例2: 16448
    # 实例3: 16480
    # 实例4: 16512
    # 实例5: 16544
    # 实例6: 16576
    
    port_to_instance = {
        16384: 0,
        16416: 1,
        16448: 2,
        16480: 3,
        16512: 4,
        16544: 5,
        16576: 6,
        16608: 7,
    }
    
    for device_id in devices:
        try:
            await adb.connect(device_id)
        except:
            pass
        
        # 从device_id提取端口号
        port = int(device_id.split(':')[1])
        instance_num = port_to_instance.get(port, '未知')
        
        device_name = await get_device_name(adb, device_id)
        
        print(f"\n设备: {device_id}")
        print(f"  端口: {port}")
        print(f"  实例编号: {instance_num}")
        print(f"  设备名称: {device_name}")
        
        if instance_num == 5:
            print(f"  ✓ 这是实例5 (安卓设备-5)")
    
    print("\n" + "=" * 60)
    print("完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
