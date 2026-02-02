"""
测试个人主页信息获取
Test Profile Information Retrieval

测试内容：
- 余额
- 积分
- 抵扣券
- 优惠券
- 用户ID
"""

import asyncio
import sys
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.adb_bridge import ADBBridge
from src.profile_reader import ProfileReader
from src.emulator_controller import EmulatorController


async def test_profile_info():
    """测试个人信息获取"""
    
    print("=" * 80)
    print("测试个人主页信息获取")
    print("=" * 80)
    
    # 1. 检测模拟器
    print("\n[步骤1] 检测模拟器...")
    emulators = EmulatorController.detect_all_emulators()
    if not emulators:
        print("[失败] 未检测到模拟器")
        return
    
    emulator_type, emulator_path = emulators[0]
    print(f"[成功] 检测到模拟器: {emulator_path}")
    
    controller = EmulatorController(emulator_path)
    
    # 2. 获取ADB端口
    print("\n[步骤2] 获取ADB端口...")
    adb_port = await controller.get_adb_port(0)
    print(f"[成功] ADB端口: {adb_port}")
    
    # 3. 连接ADB
    print("\n[步骤3] 连接ADB...")
    adb_path = controller.get_adb_path()
    adb = ADBBridge(adb_path)
    device_id = f"127.0.0.1:{adb_port}"
    
    connected = await adb.connect(device_id)
    if not connected:
        print(f"[失败] 无法连接到设备: {device_id}")
        return
    
    print(f"[成功] 已连接到设备: {device_id}")
    
    # 4. 创建ProfileReader
    print("\n[步骤4] 创建ProfileReader...")
    profile_reader = ProfileReader(adb)
    print("[成功] ProfileReader已创建")
    
    # 5. 获取个人信息（带重试）
    print("\n[步骤5] 获取个人信息（最多重试3次）...")
    print("-" * 80)
    
    profile_data = await profile_reader.get_full_profile_with_retry(
        device_id=device_id,
        max_retries=3,
        account=None  # 不提供账号信息，纯OCR识别
    )
    
    print("-" * 80)
    
    # 6. 显示结果
    print("\n[步骤6] 识别结果汇总")
    print("=" * 80)
    
    # 定义字段显示（个人页面不显示手机号）
    fields = [
        ('昵称', 'nickname', None),
        ('用户ID', 'user_id', None),
        ('余额', 'balance', '元'),
        ('积分', 'points', None),
        ('抵扣券', 'vouchers', None),
        ('优惠券', 'coupons', '张'),
    ]
    
    success_count = 0
    failed_count = 0
    
    for label, key, unit in fields:
        value = profile_data.get(key)
        if value is not None:
            if unit:
                print(f"[成功] {label}: {value} {unit}")
            else:
                print(f"[成功] {label}: {value}")
            success_count += 1
        else:
            print(f"[失败] {label}: 未获取到")
            failed_count += 1
    
    print("=" * 80)
    print(f"\n统计: 成功 {success_count}/{len(fields)}, 失败 {failed_count}/{len(fields)}")
    
    # 7. 成功率评估
    success_rate = (success_count / len(fields)) * 100
    print(f"成功率: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("[通过] 测试通过（成功率 >= 80%）")
    elif success_rate >= 60:
        print("[部分通过] 测试部分通过（成功率 >= 60%）")
    else:
        print("[失败] 测试失败（成功率 < 60%）")
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_profile_info())
