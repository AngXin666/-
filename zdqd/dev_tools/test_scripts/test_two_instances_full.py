"""
双实例完整流程测试
测试2个模拟器实例从启动到签到的完整流程
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


async def process_single_instance(
    instance_num: int,
    instance_id: int,
    account: Account,
    controller: EmulatorController,
    adb_path: str,
    app_package: str,
    app_activity: str
):
    """处理单个实例的完整流程
    
    Args:
        instance_num: 实例编号（用于显示）
        instance_id: 模拟器实例ID
        account: 账号信息
        controller: 模拟器控制器
        adb_path: ADB路径
        app_package: 应用包名
        app_activity: 应用Activity
        
    Returns:
        dict: 处理结果
    """
    result = {
        'instance_num': instance_num,
        'instance_id': instance_id,
        'account': account.phone,
        'success': False,
        'error': None,
        'timings': {},
        'data': {}
    }
    
    def log(msg):
        """日志输出（带时间戳）"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] [实例{instance_num}] {msg}")
    
    try:
        total_start = time.time()
        
        # 1. 启动模拟器实例
        log("启动模拟器实例...")
        step_start = time.time()
        
        launch_success = await controller.launch_instance(instance_id, timeout=120)
        if not launch_success:
            result['error'] = "模拟器启动失败"
            log(f"❌ {result['error']}")
            return result
        
        result['timings']['launch'] = time.time() - step_start
        log(f"✓ 模拟器启动成功 (耗时: {result['timings']['launch']:.2f}秒)")
        
        # 等待模拟器完全启动
        await asyncio.sleep(5)
        
        # 2. 连接ADB
        log("连接ADB...")
        step_start = time.time()
        
        adb_port = await controller.get_adb_port(instance_id)
        if not adb_port:
            result['error'] = "无法获取ADB端口"
            log(f"❌ {result['error']}")
            return result
        
        device_id = f"127.0.0.1:{adb_port}"
        log(f"设备ID: {device_id}")
        
        adb = ADBBridge(adb_path)
        connected = await adb.connect(device_id)
        if not connected:
            result['error'] = "ADB连接失败"
            log(f"❌ {result['error']}")
            return result
        
        result['timings']['adb_connect'] = time.time() - step_start
        log(f"✓ ADB连接成功 (耗时: {result['timings']['adb_connect']:.2f}秒)")
        
        # 3. 初始化自动化组件
        log("初始化自动化组件...")
        screen_capture = ScreenCapture(adb)
        ui_automation = UIAutomation(adb, screen_capture)
        auto_login = AutoLogin(ui_automation, screen_capture)
        
        ximeng = XimengAutomation(
            ui_automation,
            screen_capture,
            auto_login,
            adb_bridge=adb,
            log_callback=log
        )
        
        log("✓ 组件初始化完成")
        
        # 4. 启动应用
        log("启动应用...")
        step_start = time.time()
        
        app_success = await adb.start_app(device_id, app_package, app_activity)
        if not app_success:
            result['error'] = "应用启动失败"
            log(f"❌ {result['error']}")
            return result
        
        result['timings']['app_start'] = time.time() - step_start
        log(f"✓ 应用启动成功 (耗时: {result['timings']['app_start']:.2f}秒)")
        
        await asyncio.sleep(3)
        
        # 5. 处理启动流程
        log("处理启动流程...")
        step_start = time.time()
        
        startup_success = await ximeng.handle_startup_flow(
            device_id,
            log_callback=log,
            package_name=app_package,
            max_retries=3,
            stuck_timeout=15,
            max_wait_time=60,
            enable_debug=False  # 关闭调试日志，避免干扰
        )
        
        result['timings']['startup'] = time.time() - step_start
        
        if not startup_success:
            result['error'] = "启动流程失败"
            log(f"❌ {result['error']} (耗时: {result['timings']['startup']:.2f}秒)")
            return result
        
        log(f"✓ 启动流程完成 (耗时: {result['timings']['startup']:.2f}秒)")
        
        # 6. 执行完整工作流（使用主项目的方法）
        log("执行完整工作流（登录→获取资料→签到→获取最终余额→退出）...")
        step_start = time.time()
        
        workflow_result = await ximeng.run_full_workflow(device_id, account)
        
        result['timings']['workflow'] = time.time() - step_start
        result['timings']['total'] = time.time() - total_start
        
        if not workflow_result.success:
            result['error'] = workflow_result.error_message
            log(f"❌ 工作流失败: {result['error']} (耗时: {result['timings']['workflow']:.2f}秒)")
            return result
        
        log(f"✓ 工作流完成 (耗时: {result['timings']['workflow']:.2f}秒)")
        
        # 7. 收集结果数据
        result['success'] = True
        result['data'] = {
            'nickname': workflow_result.nickname,
            'user_id': workflow_result.user_id,
            'balance_before': workflow_result.balance_before,
            'balance_after': workflow_result.balance_after,
            'points': workflow_result.points,
            'vouchers': workflow_result.vouchers,
            'coupons': workflow_result.coupons,
            'checkin_reward': workflow_result.checkin_reward,
            'checkin_total_times': workflow_result.checkin_total_times,
            'login_method': workflow_result.login_method
        }
        
        log(f"✓ 全部完成 (总耗时: {result['timings']['total']:.2f}秒)")
        log(f"  昵称: {result['data']['nickname']}")
        log(f"  用户ID: {result['data']['user_id']}")
        log(f"  余额变化: {result['data']['balance_before']:.2f} → {result['data']['balance_after']:.2f}")
        log(f"  签到奖励: {result['data']['checkin_reward']:.2f}")
        
        return result
        
    except Exception as e:
        result['error'] = str(e)
        result['timings']['total'] = time.time() - total_start
        log(f"❌ 异常: {e}")
        import traceback
        traceback.print_exc()
        return result


async def test_two_instances():
    """测试3个实例的完整流程"""
    
    print("\n" + "=" * 80)
    print("三实例完整流程测试")
    print("=" * 80 + "\n")
    
    # 测试账号
    accounts = [
        Account(phone="17573358250", password="hye19911206"),
        Account(phone="13247351660", password="hye19911206"),
        Account(phone="13927308879", password="hye19911206"),
    ]
    
    print(f"测试账号:")
    for i, acc in enumerate(accounts, 1):
        print(f"  {i}. {acc.phone}")
    print("")
    
    # 1. 检测模拟器
    print("[步骤1] 检测模拟器...")
    emulators = EmulatorController.detect_all_emulators()
    
    if not emulators:
        print("❌ 未检测到模拟器")
        print("请先安装并启动MuMu模拟器")
        return
    
    emulator_type, emulator_path = emulators[0]
    print(f"检测到模拟器: {emulator_path}")
    
    controller = EmulatorController(emulator_path)
    adb_path = controller.get_adb_path()
    
    # 2. 检查运行中的实例
    print("\n[步骤2] 检查运行中的实例...")
    running_instances = await controller.get_running_instances()
    
    print(f"运行中的实例: {len(running_instances)} 个")
    
    if len(running_instances) >= 3:
        print("已有3个或更多实例运行")
        use_existing = "y"  # 自动使用现有实例
        
        if use_existing == 'y':
            instance_ids = running_instances[:3]
            print(f"使用实例: {instance_ids}")
        else:
            print("将关闭现有实例并重新启动...")
            for inst_id in running_instances:
                await controller.quit_instance(inst_id)
            await asyncio.sleep(2)
            instance_ids = [0, 1, 2]
    else:
        print(f"需要启动新实例...")
        instance_ids = [0, 1, 2]
    
    # 3. 应用配置
    app_package = "com.ry.xmsc"
    app_activity = "io.dcloud.PandoraEntry"
    
    print(f"\n应用配置:")
    print(f"  包名: {app_package}")
    print(f"  Activity: {app_activity}")
    
    # 4. 并发执行3个实例
    print("\n" + "=" * 80)
    print("开始并发执行3个实例")
    print("=" * 80 + "\n")
    
    start_time = time.time()
    
    # 创建任务
    tasks = []
    for i, (instance_id, account) in enumerate(zip(instance_ids, accounts), 1):
        task = process_single_instance(
            instance_num=i,
            instance_id=instance_id,
            account=account,
            controller=controller,
            adb_path=adb_path,
            app_package=app_package,
            app_activity=app_activity
        )
        tasks.append(task)
    
    # 并发执行
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    total_time = time.time() - start_time
    
    # 5. 显示结果
    print("\n" + "=" * 80)
    print("测试结果汇总")
    print("=" * 80 + "\n")
    
    print(f"总耗时: {total_time:.2f}秒\n")
    
    success_count = 0
    
    for i, result in enumerate(results, 1):
        if isinstance(result, Exception):
            print(f"实例{i}: ❌ 异常 - {result}")
            continue
        
        print(f"实例{i} ({result['account']}):")
        print(f"  状态: {'✓ 成功' if result['success'] else '✗ 失败'}")
        
        if result['success']:
            success_count += 1
            print(f"  昵称: {result['data']['nickname']}")
            print(f"  用户ID: {result['data']['user_id']}")
            print(f"  余额: {result['data']['balance_after']}")
            print(f"  签到奖励: {result['data']['checkin_reward']}")
            print(f"  登录方式: {result['data']['login_method']}")
            print(f"  耗时:")
            for key, value in result['timings'].items():
                print(f"    {key}: {value:.2f}秒")
        else:
            print(f"  错误: {result['error']}")
            if result['timings']:
                print(f"  已完成步骤耗时:")
                for key, value in result['timings'].items():
                    print(f"    {key}: {value:.2f}秒")
        
        print("")
    
    # 6. 统计
    print("=" * 80)
    print(f"成功: {success_count}/3")
    print(f"失败: {3 - success_count}/3")
    print(f"成功率: {success_count/3*100:.1f}%")
    print("=" * 80)
    
    # 7. 保存报告
    print("\n保存测试报告...")
    
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = report_dir / f"three_instances_test_{timestamp}.txt"
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("三实例完整流程测试报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总耗时: {total_time:.2f}秒\n\n")
        
        for i, result in enumerate(results, 1):
            if isinstance(result, Exception):
                f.write(f"实例{i}: 异常 - {result}\n\n")
                continue
            
            f.write(f"实例{i} ({result['account']}):\n")
            f.write(f"  状态: {'成功' if result['success'] else '失败'}\n")
            
            if result['success']:
                f.write(f"  昵称: {result['data']['nickname']}\n")
                f.write(f"  用户ID: {result['data']['user_id']}\n")
                f.write(f"  余额: {result['data']['balance_after']}\n")
                f.write(f"  签到奖励: {result['data']['checkin_reward']}\n")
                f.write(f"  登录方式: {result['data']['login_method']}\n")
                f.write(f"  耗时:\n")
                for key, value in result['timings'].items():
                    f.write(f"    {key}: {value:.2f}秒\n")
            else:
                f.write(f"  错误: {result['error']}\n")
            
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write(f"成功: {success_count}/3\n")
        f.write(f"失败: {3 - success_count}/3\n")
        f.write(f"成功率: {success_count/3*100:.1f}%\n")
        f.write("=" * 80 + "\n")
    
    print(f"报告已保存: {report_file}")
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    print("三实例完整流程测试")
    print("=" * 80)
    print("此测试将:")
    print("1. 启动3个模拟器实例")
    print("2. 在每个实例上启动应用")
    print("3. 处理启动流程（弹窗、广告等）")
    print("4. 登录账号")
    print("5. 执行签到")
    print("6. 收集数据并生成报告")
    print("=" * 80)
    print("")
    
    confirm = "y"  # 自动确认
    if confirm == 'y':
        asyncio.run(test_two_instances())
    else:
        print("已取消测试")
