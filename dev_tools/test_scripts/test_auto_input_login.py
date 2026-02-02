"""
测试自动输入账号密码登录功能（不使用缓存）
验证账号密码自动登录流程和模板匹配优先级
"""

import asyncio
import sys
from pathlib import Path

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.adb_bridge import ADBBridge
from src.auto_login import AutoLogin
from src.page_detector_hybrid import PageDetectorHybrid


async def test_auto_input_login(
    device_id: str = "127.0.0.1:5555",
    package_name: str = "com.ry.xmsc",
    phone: str = None,
    password: str = None
):
    """测试自动输入账号密码登录功能"""
    
    if not phone or not password:
        print("❌ 请提供手机号和密码")
        print("使用方法: python test_auto_input_login.py [设备ID] [包名] [手机号] [密码]")
        return
    
    adb = ADBBridge()
    
    # 连接设备
    print(f"连接设备: {device_id}")
    if not await adb.connect(device_id):
        print(f"❌ 无法连接到设备: {device_id}")
        return
    
    print(f"✓ 已连接到设备: {device_id}")
    
    # 创建必要的组件
    from src.screen_capture import ScreenCapture
    from src.ui_automation import UIAutomation
    
    screen_capture = ScreenCapture(adb)
    ui_automation = UIAutomation(adb, screen_capture)
    
    # 创建登录对象（禁用缓存）
    auto_login = AutoLogin(ui_automation, screen_capture, adb, enable_cache=False)
    
    # 定义日志回调
    def log_callback(msg):
        print(f"[登录] {msg}")
    
    print("\n" + "="*60)
    print(f"测试自动输入账号密码登录: {phone}")
    print("="*60)
    
    # 停止应用
    print("\n【步骤1】停止应用")
    await adb.stop_app(device_id, package_name)
    await asyncio.sleep(1)
    print("✓ 应用已停止")
    
    # 启动应用
    print("\n【步骤2】启动应用")
    success = await adb.start_app(device_id, package_name)
    if not success:
        print("❌ 应用启动失败")
        return
    
    print("✓ 应用启动成功，等待8秒...")
    await asyncio.sleep(8)
    
    # 检测启动后的页面状态
    print("\n【步骤2.5】检测应用启动后的页面")
    detector = PageDetectorHybrid(adb)
    page_result = await detector.detect_page(device_id, use_ocr=True)
    print(f"当前页面: {page_result.state.value}")
    print(f"详情: {page_result.details}")
    
    # 执行自动登录（程序自动输入账号密码）
    print("\n【步骤3】执行自动输入账号密码登录")
    print(f"手机号: {phone}")
    print(f"密码: {'*' * len(password)}")
    print("注意：程序将自动导航到登录页面，自动输入账号密码，自动点击登录按钮")
    
    login_result = await auto_login.login(
        device_id,
        phone,
        password,
        log_callback,
        use_cache=False  # 不使用缓存，强制执行账号密码登录
    )
    
    # 检查登录结果
    print("\n" + "="*60)
    print("登录结果")
    print("="*60)
    
    if login_result.success:
        print("✅ 自动登录成功！")
        if hasattr(login_result, 'elapsed_time'):
            print(f"   耗时: {login_result.elapsed_time:.2f} 秒")
        
        # 验证登录状态
        print("\n【步骤4】验证登录状态")
        detector = PageDetectorHybrid(adb)
        
        # 登录后会跳转到积分页，需要按2次返回键回到个人页面
        print("登录后跳转到积分页，按2次返回键回到个人页面...")
        await adb.press_back(device_id)
        await asyncio.sleep(1)
        await adb.press_back(device_id)
        await asyncio.sleep(2)
        
        # 检测页面状态
        page_result = await detector.detect_page(device_id, use_ocr=True)
        print(f"当前页面: {page_result.state.value}")
        
        if page_result.state.value in ['profile_logged', 'profile']:
            print("✅ 验证成功：已登录到个人页面")
            
            # 显示登录信息
            if page_result.details:
                print(f"   页面详情: {page_result.details}")
        else:
            print(f"⚠️  页面状态异常: {page_result.state.value}")
    else:
        print("❌ 自动登录失败！")
        print(f"   错误信息: {login_result.error_message}")
        if hasattr(login_result, 'elapsed_time'):
            print(f"   耗时: {login_result.elapsed_time:.2f} 秒")


if __name__ == "__main__":
    # 从命令行参数获取
    device_id = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:5555"
    package_name = sys.argv[2] if len(sys.argv) > 2 else "com.ry.xmsc"
    phone = sys.argv[3] if len(sys.argv) > 3 else None
    password = sys.argv[4] if len(sys.argv) > 4 else None
    
    print("="*60)
    print("自动输入账号密码登录测试")
    print("="*60)
    print(f"设备ID: {device_id}")
    print(f"应用包名: {package_name}")
    
    if not phone or not password:
        print("\n使用方法:")
        print("python test_auto_input_login.py [设备ID] [包名] [手机号] [密码]")
        print("\n示例:")
        print("python test_auto_input_login.py 127.0.0.1:5555 com.ry.xmsc 13800138000 password123")
        print("\n说明:")
        print("- 程序会自动导航到登录页面")
        print("- 程序会自动输入账号和密码")
        print("- 程序会自动点击登录按钮")
        print("- 不使用缓存登录，每次都输入账号密码")
    else:
        asyncio.run(test_auto_input_login(device_id, package_name, phone, password))
