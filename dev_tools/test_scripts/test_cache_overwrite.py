"""
测试缓存覆盖策略
验证直接覆盖缓存文件是否能正确切换账号
"""

import asyncio
import sys
from pathlib import Path

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.adb_bridge import ADBBridge
from src.login_cache_manager import LoginCacheManager
from src.page_detector_hybrid import PageDetectorHybrid, PageState
from src.profile_reader import ProfileReader


async def get_current_user_info(adb: ADBBridge, device_id: str):
    """获取当前登录的用户信息"""
    print("\n获取当前用户信息...")
    
    profile_reader = ProfileReader(adb)
    profile_data = await profile_reader.get_full_profile(device_id)
    
    if profile_data:
        user_id = profile_data.get('user_id', '未知')
        nickname = profile_data.get('nickname', '未知')
        phone = profile_data.get('phone', '未知')
        print(f"✓ 用户ID: {user_id}")
        print(f"✓ 昵称: {nickname}")
        print(f"✓ 手机号: {phone}")
        return profile_data
    else:
        print("⚠️  无法获取用户信息")
        return None


async def test_cache_overwrite(
    device_id: str = "127.0.0.1:5555",
    package_name: str = "com.ry.xmsc",
    account_a: str = "13800138000",
    account_b: str = "13900139000"
):
    """测试缓存覆盖策略"""
    
    adb = ADBBridge()
    cache_manager = LoginCacheManager(adb)
    
    # 连接设备
    print(f"连接设备: {device_id}")
    if not await adb.connect(device_id):
        print(f"❌ 无法连接到设备: {device_id}")
        return
    
    print(f"✓ 已连接到设备: {device_id}")
    
    # 检查两个账号的缓存
    print("\n" + "="*60)
    print("检查账号缓存")
    print("="*60)
    
    cache_a_exists = cache_manager.has_cache(account_a)
    cache_b_exists = cache_manager.has_cache(account_b)
    
    print(f"账号A ({account_a}) 缓存: {'存在' if cache_a_exists else '不存在'}")
    print(f"账号B ({account_b}) 缓存: {'存在' if cache_b_exists else '不存在'}")
    
    if cache_a_exists:
        cache_info_a = cache_manager.get_cache_info(account_a)
        if cache_info_a and 'user_id' in cache_info_a:
            print(f"  账号A 用户ID: {cache_info_a['user_id']}")
    
    if cache_b_exists:
        cache_info_b = cache_manager.get_cache_info(account_b)
        if cache_info_b and 'user_id' in cache_info_b:
            print(f"  账号B 用户ID: {cache_info_b['user_id']}")
    
    if not cache_a_exists or not cache_b_exists:
        print("\n⚠️  需要两个账号都有缓存才能测试")
        print("请先运行主程序保存账号缓存")
        return
    
    # 测试1：切换到账号A
    print("\n" + "="*60)
    print(f"测试1：切换到账号A ({account_a})")
    print("="*60)
    
    # 停止应用
    print("\n【步骤1】停止应用")
    await adb.stop_app(device_id, package_name)
    await asyncio.sleep(1)
    print("✓ 应用已停止")
    
    # 直接恢复账号A的缓存（覆盖旧文件）
    print(f"\n【步骤2】恢复账号A ({account_a}) 的缓存（覆盖旧文件）")
    if await cache_manager.restore_login_cache(device_id, account_a, package_name):
        print(f"✓ 账号A的缓存恢复成功")
        
        # 验证文件是否真的被恢复了
        print("\n验证缓存文件...")
        result = await adb.shell(device_id, f"su -c 'ls -lh /data/data/{package_name}/databases/DCStorage'")
        print(f"DCStorage 文件信息: {result}")
        
        # 检查文件大小
        result = await adb.shell(device_id, f"su -c 'stat -c %s /data/data/{package_name}/databases/DCStorage'")
        file_size = result.strip()
        print(f"DCStorage 文件大小: {file_size} 字节")
        
        if file_size == "0":
            print("⚠️  警告：文件大小为0，可能恢复失败")
    else:
        print(f"❌ 账号A的缓存恢复失败")
        return
    
    # 启动应用
    print("\n【步骤3】启动应用")
    success = await adb.start_app(device_id, package_name)
    if not success:
        print("❌ 应用启动失败")
        return
    
    print("✓ 应用启动成功，等待5秒...")
    await asyncio.sleep(5)
    
    # 检测页面状态
    print("\n【步骤4】检测页面状态")
    detector = PageDetectorHybrid(adb)
    
    for i in range(10):
        result = await detector.detect_page(device_id, use_ocr=True)
        print(f"[{i+1}/10] 页面状态: {result.state.value}")
        
        if result.state in [PageState.HOME, PageState.PROFILE, PageState.PROFILE_LOGGED]:
            print(f"✓ 到达 {result.state.value}")
            break
        
        await asyncio.sleep(1)
    
    # 获取当前用户信息
    print("\n【步骤5】导航到个人页面并验证用户信息")
    
    # 先导航到个人页面
    print("导航到个人页面...")
    await adb.tap(device_id, 446, 949)  # 点击"我的"按钮
    await asyncio.sleep(3)
    
    # 检测是否到达个人页面
    for i in range(5):
        result = await detector.detect_page(device_id, use_ocr=True)
        print(f"[{i+1}/5] 页面状态: {result.state.value}")
        
        if result.state in [PageState.PROFILE, PageState.PROFILE_LOGGED]:
            print(f"✓ 到达个人页面")
            break
        
        await asyncio.sleep(1)
    
    current_info_a = await get_current_user_info(adb, device_id)
    
    cache_info_a = cache_manager.get_cache_info(account_a)
    expected_user_id_a = cache_info_a.get('user_id') if cache_info_a else None
    
    if current_info_a and expected_user_id_a:
        current_user_id = current_info_a.get('user_id')
        if current_user_id == expected_user_id_a:
            print(f"\n✅ 账号A验证成功！")
            print(f"   当前用户ID: {current_user_id}")
            print(f"   期望用户ID: {expected_user_id_a}")
        else:
            print(f"\n❌ 账号A验证失败！")
            print(f"   当前用户ID: {current_user_id}")
            print(f"   期望用户ID: {expected_user_id_a}")
            return
    
    # 测试2：切换到账号B
    print("\n" + "="*60)
    print(f"测试2：切换到账号B ({account_b})")
    print("="*60)
    
    # 停止应用
    print("\n【步骤1】停止应用")
    await adb.stop_app(device_id, package_name)
    await asyncio.sleep(1)
    print("✓ 应用已停止")
    
    # 直接恢复账号B的缓存（覆盖账号A的文件）
    print(f"\n【步骤2】恢复账号B ({account_b}) 的缓存（覆盖账号A的文件）")
    if await cache_manager.restore_login_cache(device_id, account_b, package_name):
        print(f"✓ 账号B的缓存恢复成功")
    else:
        print(f"❌ 账号B的缓存恢复失败")
        return
    
    # 启动应用
    print("\n【步骤3】启动应用")
    success = await adb.start_app(device_id, package_name)
    if not success:
        print("❌ 应用启动失败")
        return
    
    print("✓ 应用启动成功，等待5秒...")
    await asyncio.sleep(5)
    
    # 检测页面状态
    print("\n【步骤4】检测页面状态")
    
    for i in range(10):
        result = await detector.detect_page(device_id, use_ocr=True)
        print(f"[{i+1}/10] 页面状态: {result.state.value}")
        
        if result.state in [PageState.HOME, PageState.PROFILE, PageState.PROFILE_LOGGED]:
            print(f"✓ 到达 {result.state.value}")
            break
        
        await asyncio.sleep(1)
    
    # 获取当前用户信息
    print("\n【步骤5】导航到个人页面并验证用户信息")
    
    # 先导航到个人页面
    print("导航到个人页面...")
    await adb.tap(device_id, 446, 949)  # 点击"我的"按钮
    await asyncio.sleep(3)
    
    # 检测是否到达个人页面
    for i in range(5):
        result = await detector.detect_page(device_id, use_ocr=True)
        print(f"[{i+1}/5] 页面状态: {result.state.value}")
        
        if result.state in [PageState.PROFILE, PageState.PROFILE_LOGGED]:
            print(f"✓ 到达个人页面")
            break
        
        await asyncio.sleep(1)
    
    current_info_b = await get_current_user_info(adb, device_id)
    
    cache_info_b = cache_manager.get_cache_info(account_b)
    expected_user_id_b = cache_info_b.get('user_id') if cache_info_b else None
    
    if current_info_b and expected_user_id_b:
        current_user_id = current_info_b.get('user_id')
        if current_user_id == expected_user_id_b:
            print(f"\n✅ 账号B验证成功！")
            print(f"   当前用户ID: {current_user_id}")
            print(f"   期望用户ID: {expected_user_id_b}")
        else:
            print(f"\n❌ 账号B验证失败！")
            print(f"   当前用户ID: {current_user_id}")
            print(f"   期望用户ID: {expected_user_id_b}")
            return
    
    # 最终验证：确认两个账号的用户ID不同
    print("\n" + "="*60)
    print("最终验证")
    print("="*60)
    
    if current_info_a and current_info_b:
        user_id_a = current_info_a.get('user_id')
        user_id_b = current_info_b.get('user_id')
        
        if user_id_a != user_id_b:
            print(f"✅ 测试通过！两个账号的用户ID不同")
            print(f"   账号A用户ID: {user_id_a}")
            print(f"   账号B用户ID: {user_id_b}")
            print(f"\n✅ 缓存覆盖策略有效！可以正确区分不同账号")
        else:
            print(f"❌ 测试失败！两个账号的用户ID相同")
            print(f"   用户ID: {user_id_a}")
            print(f"\n❌ 缓存覆盖策略无效！无法区分不同账号")


if __name__ == "__main__":
    # 从命令行参数获取设备ID和包名
    device_id = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:5555"
    package_name = sys.argv[2] if len(sys.argv) > 2 else "com.ry.xmsc"
    account_a = sys.argv[3] if len(sys.argv) > 3 else "13800138000"
    account_b = sys.argv[4] if len(sys.argv) > 4 else "13900139000"
    
    print("="*60)
    print("缓存覆盖策略测试")
    print("="*60)
    print(f"设备ID: {device_id}")
    print(f"应用包名: {package_name}")
    print(f"账号A: {account_a}")
    print(f"账号B: {account_b}")
    
    asyncio.run(test_cache_overwrite(device_id, package_name, account_a, account_b))
