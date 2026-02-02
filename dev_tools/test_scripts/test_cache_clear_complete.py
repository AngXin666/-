"""
完整测试：清理账号缓存 + 输入账号密码登录
Complete Test: Clear account cache + Auto input login
"""

import asyncio
import sys
from pathlib import Path

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.adb_bridge import ADBBridge
from src.auto_login import AutoLogin
from src.page_detector_hybrid import PageDetectorHybrid


async def test_cache_clear_and_login(
    device_id: str = "127.0.0.1:5555",
    package_name: str = "com.ry.xmsc",
    phone: str = None,
    password: str = None
):
    """完整测试：清理账号缓存 + 自动输入账号密码登录"""
    
    if not phone or not password:
        print("❌ 请提供手机号和密码")
        print("使用方法: python test_cache_clear_complete.py [设备ID] [包名] [手机号] [密码]")
        return
    
    adb = ADBBridge()
    
    # 连接设备
    print(f"\n{'='*80}")
    print(f"【步骤1】连接设备")
    print(f"{'='*80}")
    print(f"设备ID: {device_id}")
    
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
        print(f"  [登录] {msg}")
    
    # ========================================
    # 步骤2：停止应用
    # ========================================
    print(f"\n{'='*80}")
    print(f"【步骤2】停止应用")
    print(f"{'='*80}")
    
    await adb.stop_app(device_id, package_name)
    await asyncio.sleep(1)
    print("✓ 应用已停止")
    
    # ========================================
    # 步骤3：清理账号缓存（让应用回到未登录状态）
    # ========================================
    print(f"\n{'='*80}")
    print(f"【步骤3】清理账号缓存（让应用回到未登录状态）")
    print(f"{'='*80}")
    
    print("说明：")
    print("  - 清理账号缓存：删除登录状态文件")
    print("  - 效果：应用启动后需要重新登录")
    
    # 清理账号缓存文件（登录状态）
    print("\n清理账号缓存文件...")
    cache_files = [
        f"/data/data/{package_name}/shared_prefs/userInfo.xml",
        f"/data/data/{package_name}/shared_prefs/loginInfo.xml",
        f"/data/data/{package_name}/shared_prefs/accountInfo.xml",
        f"/data/data/{package_name}/files/user_cache",
        f"/data/data/{package_name}/files/login_cache"
    ]
    
    for cache_file in cache_files:
        result = await adb.shell(device_id, f"rm -f {cache_file}")
        print(f"  删除 {cache_file.split('/')[-1]}: {result.strip() if result.strip() else '成功'}")
    
    print("\n✓ 账号缓存清理完成")
    await asyncio.sleep(2)
    
    # ========================================
    # 步骤4：启动应用
    # ========================================
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
    
    # ========================================
    # 步骤5：检测应用启动后的页面状态
    # ========================================
    print(f"\n{'='*80}")
    print(f"【步骤5】检测应用启动后的页面状态")
    print(f"{'='*80}")
    
    detector = PageDetectorHybrid(adb)
    page_result = await detector.detect_page(device_id, use_ocr=True)
    print(f"当前页面: {page_result.state.value}")
    if page_result.details:
        print(f"详情: {page_result.details}")
    
    # ========================================
    # 步骤6：执行自动输入账号密码登录
    # ========================================
    print(f"\n{'='*80}")
    print(f"【步骤6】执行自动输入账号密码登录")
    print(f"{'='*80}")
    print(f"手机号: {phone}")
    print(f"密码: {'*' * len(password)}")
    print("\n登录流程监控：")
    print("-" * 80)
    
    # 记录开始时间
    import time
    start_time = time.time()
    
    # 执行登录
    login_result = await auto_login.login(
        device_id,
        phone,
        password,
        log_callback,
        use_cache=False  # 不使用缓存，强制执行账号密码登录
    )
    
    # 记录结束时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("-" * 80)
    
    # ========================================
    # 步骤7：检查登录结果
    # ========================================
    print(f"\n{'='*80}")
    print(f"【步骤7】登录结果")
    print(f"{'='*80}")
    
    if login_result.success:
        print("✅ 自动登录成功！")
        print(f"   耗时: {elapsed_time:.2f} 秒")
        
        # ========================================
        # 步骤8：验证登录状态
        # ========================================
        print(f"\n{'='*80}")
        print(f"【步骤8】验证登录状态")
        print(f"{'='*80}")
        
        # 等待页面稳定
        print("等待3秒让页面稳定...")
        await asyncio.sleep(3)
        
        # 导航到个人页面
        print("导航到个人页面...")
        await adb.tap(device_id, 450, 920)  # 点击"我的"按钮
        await asyncio.sleep(3)
        
        # 检测页面状态
        page_result = await detector.detect_page(device_id, use_ocr=True)
        print(f"当前页面: {page_result.state.value}")
        
        if page_result.state.value in ['profile_logged', 'profile']:
            print("✅ 验证成功：已登录到个人页面")
            
            # 显示登录信息
            if page_result.details:
                print(f"   页面详情: {page_result.details}")
            
            # ========================================
            # 步骤9：获取用户信息
            # ========================================
            print(f"\n{'='*80}")
            print(f"【步骤9】获取用户信息")
            print(f"{'='*80}")
            
            # 使用OCR获取用户信息
            from src.ximeng_automation import XimengAutomation
            ximeng = XimengAutomation(adb, screen_capture, ui_automation)
            
            # 获取余额
            print("正在获取余额...")
            balance = await ximeng.get_balance(device_id, from_cache_login=False)
            if balance is not None:
                print(f"✓ 余额: {balance:.2f} 元")
            else:
                print("⚠️ 无法获取余额")
            
            # 获取个人信息
            print("\n正在获取个人信息...")
            profile_info = await ximeng.get_profile_info(device_id, phone, from_cache_login=False)
            
            if profile_info.get('nickname'):
                print(f"✓ 昵称: {profile_info['nickname']}")
            if profile_info.get('user_id'):
                print(f"✓ 用户ID: {profile_info['user_id']}")
            if profile_info.get('points') is not None:
                print(f"✓ 积分: {profile_info['points']}")
            if profile_info.get('vouchers') is not None:
                print(f"✓ 代金券: {profile_info['vouchers']} 张")
            
        else:
            print(f"⚠️  页面状态异常: {page_result.state.value}")
    else:
        print("❌ 自动登录失败！")
        print(f"   错误信息: {login_result.error_message}")
        print(f"   耗时: {elapsed_time:.2f} 秒")
    
    # ========================================
    # 测试总结
    # ========================================
    print(f"\n{'='*80}")
    print(f"【测试总结】")
    print(f"{'='*80}")
    print(f"✓ 清理应用缓存（保留账号缓存）")
    print(f"✓ 应用重新启动")
    print(f"✓ 自动输入账号密码登录")
    print(f"✓ 登录{'成功' if login_result.success else '失败'}")
    print(f"✓ 总耗时: {elapsed_time:.2f} 秒")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # 从命令行参数获取
    device_id = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:5555"
    package_name = sys.argv[2] if len(sys.argv) > 2 else "com.ry.xmsc"
    phone = sys.argv[3] if len(sys.argv) > 3 else "15766121960"
    password = sys.argv[4] if len(sys.argv) > 4 else "hye19911206"
    
    print("="*80)
    print("完整测试：清理账号缓存 + 自动输入账号密码登录")
    print("="*80)
    print(f"设备ID: {device_id}")
    print(f"应用包名: {package_name}")
    print(f"测试账号: {phone}")
    print("="*80)
    
    asyncio.run(test_cache_clear_and_login(device_id, package_name, phone, password))
