"""
简单签到测试 - 验证签到功能是否正常
"""
import asyncio
from src.adb_bridge import ADBBridge
from src.emulator_controller import EmulatorController
from src.screen_capture import ScreenCapture
from src.ui_automation import UIAutomation
from src.auto_login import AutoLogin
from src.ximeng_automation import XimengAutomation
from src.models.models import Account


async def test_simple_checkin():
    """测试单个账号的签到功能"""
    
    print("\n" + "=" * 80)
    print("简单签到测试")
    print("=" * 80 + "\n")
    
    # 测试账号
    account = Account(phone="13322736481", password="hye1991120619065355068")
    
    print(f"测试账号: {account.phone}\n")
    
    # 1. 检测模拟器
    print("[步骤1] 检测模拟器...")
    emulators = EmulatorController.detect_all_emulators()
    
    if not emulators:
        print("❌ 未检测到模拟器")
        return
    
    emulator_type, emulator_path = emulators[0]
    print(f"✓ 检测到模拟器: {emulator_path}")
    
    controller = EmulatorController(emulator_path)
    adb_path = controller.get_adb_path()
    
    # 2. 获取运行中的实例
    print("\n[步骤2] 获取运行中的实例...")
    running_instances = await controller.get_running_instances()
    
    if not running_instances:
        print("❌ 没有运行中的实例")
        return
    
    instance_id = running_instances[0]
    print(f"✓ 使用实例: {instance_id}")
    
    # 3. 连接ADB
    print("\n[步骤3] 连接ADB...")
    adb_port = await controller.get_adb_port(instance_id)
    device_id = f"127.0.0.1:{adb_port}"
    print(f"设备ID: {device_id}")
    
    adb = ADBBridge(adb_path)
    connected = await adb.connect(device_id)
    if not connected:
        print("❌ ADB连接失败")
        return
    
    print(f"✓ ADB连接成功")
    
    # 4. 初始化组件
    print("\n[步骤4] 初始化组件...")
    screen_capture = ScreenCapture(adb)
    ui_automation = UIAutomation(adb, screen_capture)
    auto_login = AutoLogin(ui_automation, screen_capture)
    
    ximeng = XimengAutomation(
        ui_automation,
        screen_capture,
        auto_login,
        adb_bridge=adb,
        log_callback=print
    )
    
    print("✓ 组件初始化完成")
    
    # 5. 执行完整工作流
    print("\n[步骤5] 执行完整工作流...")
    print("=" * 80)
    
    result = await ximeng.run_full_workflow(device_id, account)
    
    print("\n" + "=" * 80)
    print("测试结果")
    print("=" * 80)
    print(f"成功: {result.success}")
    print(f"昵称: {result.nickname}")
    print(f"用户ID: {result.user_id}")
    print(f"余额前: {result.balance_before}")
    print(f"余额: {result.balance_after}")
    print(f"签到奖励: {result.checkin_reward}")
    print(f"签到总次数: {result.checkin_total_times}")
    
    if result.error_message:
        print(f"错误: {result.error_message}")
    
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_simple_checkin())
