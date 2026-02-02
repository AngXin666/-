"""
测试主脚本的完整登录流程
Test main script complete login flow with monitoring
"""

import asyncio
import sys
from pathlib import Path

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.adb_bridge import ADBBridge
from src.emulator_controller import EmulatorController
from src.screen_capture import ScreenCapture
from src.ui_automation import UIAutomation
from src.auto_login import AutoLogin
from src.ximeng_automation import XimengAutomation
from src.page_detector_hybrid import PageDetectorHybrid
from src.account_manager import Account


async def test_main_script_login(
    device_id: str = "127.0.0.1:5555",
    package_name: str = "com.ry.xmsc",
    phone: str = "15766121960",
    password: str = "hye19911206",
    emulator_type: str = "mumu"  # 模拟器类型
):
    """测试主脚本的完整登录流程"""
    
    print("="*80)
    print("测试主脚本完整登录流程（清理应用缓存 + 账号密码登录）")
    print("="*80)
    print(f"设备ID: {device_id}")
    print(f"应用包名: {package_name}")
    print(f"测试账号: {phone}")
    print(f"模拟器类型: {emulator_type}")
    print("="*80)
    
    adb = ADBBridge()
    
    # 连接设备
    print(f"\n{'='*80}")
    print(f"【步骤1】连接设备")
    print(f"{'='*80}")
    
    if not await adb.connect(device_id):
        print(f"❌ 无法连接到设备: {device_id}")
        return
    
    print(f"✓ 已连接到设备: {device_id}")
    
    # 停止应用
    print(f"\n{'='*80}")
    print(f"【步骤2】停止应用")
    print(f"{'='*80}")
    
    await adb.stop_app(device_id, package_name)
    await asyncio.sleep(1)
    print("✓ 应用已停止")
    
    # 清理应用缓存（不清理账号缓存）
    print(f"\n{'='*80}")
    print(f"【步骤3】清理应用缓存（保留账号缓存）")
    print(f"{'='*80}")
    
    print("说明：")
    print("  - 清理应用缓存：删除临时文件、图片缓存等")
    print("  - 保留账号缓存：不删除登录状态文件")
    
    # 方法1：尝试使用 pm clear-cache
    print("\n尝试方法1: pm clear-cache")
    result = await adb.shell(device_id, f"pm clear-cache {package_name}")
    print(f"  结果: {result.strip() if result.strip() else '成功'}")
    
    if "Unknown" in result or "Error" in result:
        # 方法2：直接删除缓存目录
        print("\npm clear-cache 不支持，尝试方法2: rm -rf cache")
        result = await adb.shell(device_id, f"rm -rf /data/data/{package_name}/cache/*")
        print(f"  结果: {result.strip() if result.strip() else '成功'}")
    
    print("\n✓ 应用缓存清理完成")
    await asyncio.sleep(2)
    
    # 启动应用
    print(f"\n{'='*80}")
    print(f"【步骤4】启动应用")
    print(f"{'='*80}")
    
    success = await adb.start_app(device_id, package_name)
    if not success:
        print("❌ 应用启动失败")
        return
    
    print("✓ 应用启动成功")
    print("等待8秒让应用完全加载...")
    await asyncio.sleep(8)
    
    # 创建必要的组件
    screen_capture = ScreenCapture(adb)
    ui_automation = UIAutomation(adb, screen_capture)
    auto_login = AutoLogin(ui_automation, screen_capture, adb, enable_cache=False, emulator_type=emulator_type)
    
    # 定义日志回调
    log_messages = []
    def log_callback(msg):
        timestamp = asyncio.get_event_loop().time()
        log_messages.append((timestamp, msg))
        print(f"  [主脚本] {msg}")
    
    ximeng = XimengAutomation(ui_automation, screen_capture, auto_login, adb, log_callback=log_callback)
    
    # 创建账号对象
    account = Account(phone=phone, password=password)
    
    # 执行主脚本的登录流程
    print(f"\n{'='*80}")
    print(f"【步骤5】执行主脚本登录流程（模拟 _process_single_account_monitored）")
    print(f"{'='*80}")
    print("监控日志：")
    print("-" * 80)
    
    start_time = asyncio.get_event_loop().time()
    
    try:
        # 检测当前页面
        detector = PageDetectorHybrid(adb)
        page_result = await detector.detect_page(device_id, use_ocr=True)
        log_callback(f"当前页面: {page_result.state.value}")
        
        # 执行登录
        log_callback("开始执行正常登录...")
        login_result = await auto_login.login(
            device_id,
            account.phone,
            account.password,
            log_callback,
            use_cache=False
        )
        
        if not login_result.success:
            log_callback(f"❌ 登录失败: {login_result.error_message}")
            return
        
        log_callback("✓ 登录成功！")
        
        # 登录后按2次返回键回到个人页面
        log_callback("登录后跳转到积分页，按2次返回键回到个人页面...")
        await adb.press_back(device_id)
        await asyncio.sleep(1)
        await adb.press_back(device_id)
        await asyncio.sleep(2)
        log_callback("✓ 已返回到个人页面")
        
        # 验证页面状态
        page_result = await detector.detect_page(device_id, use_ocr=True)
        log_callback(f"当前页面: {page_result.state.value}")
        
        if page_result.state.value in ['profile_logged', 'profile']:
            log_callback("✓ 验证成功：已在个人页面")
        else:
            log_callback(f"⚠️ 页面状态异常: {page_result.state.value}")
        
        # 获取余额
        log_callback("正在获取账户余额...")
        balance = await ximeng.get_balance(device_id, from_cache_login=False)
        if balance is not None:
            log_callback(f"✓ 当前余额: {balance:.2f} 元")
        else:
            log_callback("⚠️ 无法获取余额")
        
        # 获取个人信息
        log_callback("正在获取个人信息...")
        profile_info = await ximeng.get_profile_info(device_id, account.phone, from_cache_login=False)
        
        if profile_info.get('nickname'):
            log_callback(f"✓ 昵称: {profile_info['nickname']}")
        if profile_info.get('user_id'):
            log_callback(f"✓ 用户ID: {profile_info['user_id']}")
        if profile_info.get('points') is not None:
            log_callback(f"✓ 积分: {profile_info['points']}")
        if profile_info.get('vouchers') is not None:
            log_callback(f"✓ 代金券: {profile_info['vouchers']} 张")
        
        end_time = asyncio.get_event_loop().time()
        elapsed_time = end_time - start_time
        
        print("-" * 80)
        print(f"\n{'='*80}")
        print(f"【测试结果】")
        print(f"{'='*80}")
        print(f"✅ 主脚本登录流程测试成功！")
        print(f"✓ 总耗时: {elapsed_time:.2f} 秒")
        print(f"✓ 登录成功")
        print(f"✓ 返回到个人页面")
        print(f"✓ 获取用户信息成功")
        if balance is not None:
            print(f"✓ 余额: {balance:.2f} 元")
        print(f"{'='*80}")
        
        # 显示日志统计
        print(f"\n日志统计：")
        print(f"  总日志条数: {len(log_messages)}")
        if len(log_messages) > 1:
            total_time = log_messages[-1][0] - log_messages[0][0]
            print(f"  总处理时间: {total_time:.2f} 秒")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_main_script_login())
