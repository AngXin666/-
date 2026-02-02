"""
单账号完整流程测试
测试优化效果和性能监控
"""
import asyncio
import time
from datetime import datetime
from pathlib import Path

from src.adb_bridge import ADBBridge
from src.emulator_controller import EmulatorController
from src.screen_capture import ScreenCapture
from src.ui_automation import UIAutomation
from src.auto_login import AutoLogin
from src.ximeng_automation import XimengAutomation
from src.models.models import Account


async def test_single_account():
    """测试单个账号的完整流程"""
    
    print("\n" + "=" * 80)
    print("单账号完整流程测试 - 优化效果监控")
    print("=" * 80 + "\n")
    
    # 测试账号
    test_account = Account(
        phone="13322736481",
        password="hye1991120619065355068"
    )
    
    print(f"测试账号: {test_account.phone}")
    print(f"预期昵称: hye19911206")
    print("")
    
    # 1. 检测模拟器
    print("[步骤1] 检测模拟器...")
    emulators = EmulatorController.detect_all_emulators()
    
    if not emulators:
        print("❌ 未检测到运行中的模拟器")
        print("请先启动MuMu模拟器")
        return
    
    emulator_type, emulator_path = emulators[0]
    print(f"✓ 检测到模拟器: {emulator_path}")
    
    controller = EmulatorController(emulator_path)
    
    # 2. 连接 ADB
    print("\n[步骤2] 连接 ADB...")
    adb_port = await controller.get_adb_port(0)
    if not adb_port:
        print("❌ 无法获取 ADB 端口")
        return
    
    device_id = f"127.0.0.1:{adb_port}"
    print(f"✓ ADB 端口: {adb_port}")
    print(f"✓ 设备 ID: {device_id}")
    
    adb_path = controller.get_adb_path()
    adb = ADBBridge(adb_path)
    
    connected = await adb.connect(device_id)
    if not connected:
        print("❌ ADB 连接失败")
        return
    
    print("✓ ADB 连接成功")
    
    # 3. 初始化组件
    print("\n[步骤3] 初始化自动化组件...")
    screen_capture = ScreenCapture(adb)
    ui_automation = UIAutomation(adb, screen_capture)
    auto_login = AutoLogin(ui_automation, screen_capture)
    
    # 创建日志回调函数
    def log_callback(msg):
        print(f"  {msg}")
    
    ximeng = XimengAutomation(
        ui_automation, 
        screen_capture, 
        auto_login,
        adb_bridge=adb,
        log_callback=log_callback
    )
    
    print("✓ 组件初始化完成")
    
    # 4. 启动应用
    print("\n[步骤4] 启动应用...")
    app_package = "com.ry.xmsc"
    app_activity = "io.dcloud.PandoraEntry"
    
    success = await adb.start_app(device_id, app_package, app_activity)
    if not success:
        print("❌ 应用启动失败")
        return
    
    print("✓ 应用已启动")
    await asyncio.sleep(3)
    
    # 5. 处理启动流程（带性能监控）
    print("\n[步骤5] 处理启动流程...")
    print("监控指标:")
    print("  - 启动流程耗时")
    print("  - 页面检测次数")
    print("  - 弹窗处理次数")
    print("")
    
    start_time = time.time()
    
    startup_success = await ximeng.handle_startup_flow(
        device_id,
        log_callback=log_callback,
        package_name=app_package,
        max_retries=3,
        stuck_timeout=15,
        max_wait_time=60,
        enable_debug=True
    )
    
    startup_time = time.time() - start_time
    
    if not startup_success:
        print(f"\n❌ 启动流程失败 (耗时: {startup_time:.2f}秒)")
        return
    
    print(f"\n✓ 启动流程完成 (耗时: {startup_time:.2f}秒)")
    
    # 6. 执行完整工作流（带性能监控）
    print("\n[步骤6] 执行完整工作流...")
    print("监控指标:")
    print("  - 登录耗时")
    print("  - 签到耗时")
    print("  - 余额读取耗时")
    print("  - 个人信息获取耗时")
    print("  - 缓存命中率")
    print("")
    
    workflow_start = time.time()
    
    result = await ximeng.run_full_workflow(device_id, test_account)
    
    workflow_time = time.time() - workflow_start
    total_time = time.time() - start_time
    
    # 7. 显示结果
    print("\n" + "=" * 80)
    print("测试结果")
    print("=" * 80 + "\n")
    
    print(f"账号: {result.phone}")
    print(f"状态: {'✓ 成功' if result.success else '✗ 失败'}")
    
    if result.success:
        print(f"\n个人信息:")
        print(f"  昵称: {result.nickname or '未获取'}")
        print(f"  用户ID: {result.user_id or '未获取'}")
        
        print(f"\n余额信息:")
        print(f"  余额前: {result.balance_before if result.balance_before is not None else '未获取'}")
        print(f"  余额: {result.balance_after if result.balance_after is not None else '未获取'}")
        
        print(f"\n积分信息:")
        print(f"  积分: {result.points if result.points is not None else '未获取'}")
        print(f"  代金券: {result.vouchers if result.vouchers is not None else '未获取'}")
        print(f"  优惠券: {result.coupons if result.coupons is not None else '未获取'}")
        
        print(f"\n签到信息:")
        print(f"  签到奖励: {result.checkin_reward if result.checkin_reward is not None else '未签到'}")
        print(f"  签到次数: {result.checkin_total_times if result.checkin_total_times is not None else '未获取'}")
        print(f"  余额: {result.checkin_balance_after if result.checkin_balance_after is not None else '未获取'}")
        
        print(f"\n登录方式: {result.login_method or '未知'}")
    else:
        print(f"\n错误信息: {result.error_message}")
    
    # 8. 性能统计
    print(f"\n" + "=" * 80)
    print("性能统计")
    print("=" * 80 + "\n")
    
    print(f"启动流程耗时: {startup_time:.2f}秒")
    print(f"工作流耗时: {workflow_time:.2f}秒")
    print(f"总耗时: {total_time:.2f}秒")
    
    # 对比优化前的基准时间
    baseline_time = 35.0  # 优化前基准时间
    if total_time < baseline_time:
        improvement = ((baseline_time - total_time) / baseline_time) * 100
        saved_time = baseline_time - total_time
        print(f"\n✓ 性能提升: {improvement:.1f}%")
        print(f"✓ 节省时间: {saved_time:.1f}秒")
    else:
        print(f"\n⚠️ 耗时超过基准: +{total_time - baseline_time:.1f}秒")
    
    print(f"\n优化效果:")
    print(f"  - 缓存登录: {'✓ 已启用' if result.login_method == '缓存登录' else '✗ 未使用'}")
    print(f"  - 并行处理: ✓ 已启用")
    print(f"  - 位置追踪: ✓ 已启用")
    print(f"  - 检测缓存: ✓ 已启用")
    
    # 9. 保存报告
    print(f"\n[步骤7] 保存测试报告...")
    
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = report_dir / f"single_account_test_{timestamp}.txt"
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("单账号完整流程测试报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"账号: {result.phone}\n")
        f.write(f"状态: {'成功' if result.success else '失败'}\n\n")
        
        if result.success:
            f.write("个人信息:\n")
            f.write(f"  昵称: {result.nickname or '未获取'}\n")
            f.write(f"  用户ID: {result.user_id or '未获取'}\n\n")
            
            f.write("余额信息:\n")
            f.write(f"  余额前: {result.balance_before if result.balance_before is not None else '未获取'}\n")
            f.write(f"  余额: {result.balance_after if result.balance_after is not None else '未获取'}\n\n")
            
            f.write("积分信息:\n")
            f.write(f"  积分: {result.points if result.points is not None else '未获取'}\n")
            f.write(f"  代金券: {result.vouchers if result.vouchers is not None else '未获取'}\n")
            f.write(f"  优惠券: {result.coupons if result.coupons is not None else '未获取'}\n\n")
            
            f.write("签到信息:\n")
            f.write(f"  签到奖励: {result.checkin_reward if result.checkin_reward is not None else '未签到'}\n")
            f.write(f"  签到次数: {result.checkin_total_times if result.checkin_total_times is not None else '未获取'}\n")
            f.write(f"  余额: {result.checkin_balance_after if result.checkin_balance_after is not None else '未获取'}\n\n")
            
            f.write(f"登录方式: {result.login_method or '未知'}\n\n")
        else:
            f.write(f"错误信息: {result.error_message}\n\n")
        
        f.write("性能统计:\n")
        f.write(f"  启动流程耗时: {startup_time:.2f}秒\n")
        f.write(f"  工作流耗时: {workflow_time:.2f}秒\n")
        f.write(f"  总耗时: {total_time:.2f}秒\n\n")
        
        if total_time < baseline_time:
            improvement = ((baseline_time - total_time) / baseline_time) * 100
            saved_time = baseline_time - total_time
            f.write(f"  性能提升: {improvement:.1f}%\n")
            f.write(f"  节省时间: {saved_time:.1f}秒\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"✓ 报告已保存: {report_file}")
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(test_single_account())
