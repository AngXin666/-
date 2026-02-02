"""
测试账号缓存匹配
验证切换账号时，恢复的缓存是否匹配正确的账号
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


async def clear_account_cache(adb: ADBBridge, device_id: str, package_name: str):
    """清理应用内的账号缓存文件"""
    print("\n清理账号缓存文件...")
    
    cache_files = [
        f"/data/data/{package_name}/shared_prefs/lcdpr.xml",
        f"/data/data/{package_name}/databases/DCStorage",
        f"/data/data/{package_name}/databases/DCStorage-shm",
        f"/data/data/{package_name}/databases/DCStorage-wal"
    ]
    
    for file_path in cache_files:
        result = await adb.shell(device_id, f"su -c 'rm -f {file_path}'")
        # 检查文件是否还存在
        check = await adb.shell(device_id, f"su -c 'test -f {file_path} && echo EXISTS || echo NOT_EXISTS'")
        if "NOT_EXISTS" in check:
            print(f"  ✓ 已删除: {file_path}")
        else:
            print(f"  ⚠️  删除失败: {file_path}")
    
    print("✓ 账号缓存文件清理完成")


async def get_current_user_id(adb: ADBBridge, device_id: str, package_name: str):
    """获取当前登录的用户ID"""
    print("\n获取当前登录的用户ID...")
    
    # 读取 DCStorage 数据库中的用户ID
    # 这里简化处理，实际可能需要解析数据库
    result = await adb.shell(device_id, f"su -c 'cat /data/data/{package_name}/databases/DCStorage' 2>/dev/null")
    
    # 或者通过 ProfileReader 获取
    profile_reader = ProfileReader(adb)
    profile_data = await profile_reader.get_full_profile(device_id)
    
    if profile_data and profile_data.get('user_id'):
        user_id = profile_data['user_id']
        print(f"✓ 当前用户ID: {user_id}")
        return user_id
    else:
        print("⚠️  无法获取用户ID")
        return None


async def test_account_switch_with_cache_match(
    device_id: str = "127.0.0.1:5555",
    package_name: str = "com.ry.xmsc",
    account_a: str = "13800138000",
    account_b: str = "13900139000"
):
    """测试账号切换时的缓存匹配"""
    
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
    
    if not cache_a_exists and not cache_b_exists:
        print("\n⚠️  两个账号都没有缓存，无法测试缓存匹配")
        print("请先运行主程序保存账号缓存")
        return
    
    # 测试1：切换到账号A
    if cache_a_exists:
        print("\n" + "="*60)
        print(f"测试1：切换到账号A ({account_a})")
        print("="*60)
        
        # 停止应用
        print("\n【步骤1】停止应用")
        await adb.stop_app(device_id, package_name)
        await asyncio.sleep(1)
        print("✓ 应用已停止")
        
        # 清理账号缓存
        print("\n【步骤2】清理账号缓存文件")
        await clear_account_cache(adb, device_id, package_name)
        
        # 恢复账号A的缓存
        print(f"\n【步骤3】恢复账号A ({account_a}) 的缓存")
        if await cache_manager.restore_login_cache(device_id, account_a, package_name):
            print(f"✓ 账号A的缓存恢复成功")
        else:
            print(f"❌ 账号A的缓存恢复失败")
            return
        
        # 启动应用
        print("\n【步骤4】启动应用")
        success = await adb.start_app(device_id, package_name)
        if not success:
            print("❌ 应用启动失败")
            return
        
        print("✓ 应用启动成功，等待5秒...")
        await asyncio.sleep(5)
        
        # 检测页面状态
        print("\n【步骤5】检测页面状态")
        detector = PageDetectorHybrid(adb)
        
        for i in range(10):
            result = await detector.detect_page(device_id, use_ocr=True)
            print(f"[{i+1}/10] 页面状态: {result.state.value}")
            
            if result.state in [PageState.HOME, PageState.PROFILE, PageState.PROFILE_LOGGED]:
                print(f"✓ 到达 {result.state.value}")
                break
            
            await asyncio.sleep(1)
        
        # 验证用户ID
        print("\n【步骤6】验证用户ID是否匹配")
        current_user_id = await get_current_user_id(adb, device_id, package_name)
        
        cache_info_a = cache_manager.get_cache_info(account_a)
        expected_user_id = cache_info_a.get('user_id') if cache_info_a else None
        
        if current_user_id and expected_user_id:
            if current_user_id == expected_user_id:
                print(f"✅ 用户ID匹配！")
                print(f"   当前: {current_user_id}")
                print(f"   期望: {expected_user_id}")
            else:
                print(f"❌ 用户ID不匹配！")
                print(f"   当前: {current_user_id}")
                print(f"   期望: {expected_user_id}")
        else:
            print("⚠️  无法验证用户ID（缓存中没有保存或无法读取）")
    
    # 测试2：切换到账号B
    if cache_b_exists:
        print("\n" + "="*60)
        print(f"测试2：切换到账号B ({account_b})")
        print("="*60)
        
        # 停止应用
        print("\n【步骤1】停止应用")
        await adb.stop_app(device_id, package_name)
        await asyncio.sleep(1)
        print("✓ 应用已停止")
        
        # 清理账号缓存
        print("\n【步骤2】清理账号缓存文件")
        await clear_account_cache(adb, device_id, package_name)
        
        # 恢复账号B的缓存
        print(f"\n【步骤3】恢复账号B ({account_b}) 的缓存")
        if await cache_manager.restore_login_cache(device_id, account_b, package_name):
            print(f"✓ 账号B的缓存恢复成功")
        else:
            print(f"❌ 账号B的缓存恢复失败")
            return
        
        # 启动应用
        print("\n【步骤4】启动应用")
        success = await adb.start_app(device_id, package_name)
        if not success:
            print("❌ 应用启动失败")
            return
        
        print("✓ 应用启动成功，等待5秒...")
        await asyncio.sleep(5)
        
        # 检测页面状态
        print("\n【步骤5】检测页面状态")
        detector = PageDetectorHybrid(adb)
        
        for i in range(10):
            result = await detector.detect_page(device_id, use_ocr=True)
            print(f"[{i+1}/10] 页面状态: {result.state.value}")
            
            if result.state in [PageState.HOME, PageState.PROFILE, PageState.PROFILE_LOGGED]:
                print(f"✓ 到达 {result.state.value}")
                break
            
            await asyncio.sleep(1)
        
        # 验证用户ID
        print("\n【步骤6】验证用户ID是否匹配")
        current_user_id = await get_current_user_id(adb, device_id, package_name)
        
        cache_info_b = cache_manager.get_cache_info(account_b)
        expected_user_id = cache_info_b.get('user_id') if cache_info_b else None
        
        if current_user_id and expected_user_id:
            if current_user_id == expected_user_id:
                print(f"✅ 用户ID匹配！")
                print(f"   当前: {current_user_id}")
                print(f"   期望: {expected_user_id}")
            else:
                print(f"❌ 用户ID不匹配！")
                print(f"   当前: {current_user_id}")
                print(f"   期望: {expected_user_id}")
        else:
            print("⚠️  无法验证用户ID（缓存中没有保存或无法读取）")
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)


if __name__ == "__main__":
    # 从命令行参数获取设备ID和包名
    device_id = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:5555"
    package_name = sys.argv[2] if len(sys.argv) > 2 else "com.ry.xmsc"
    account_a = sys.argv[3] if len(sys.argv) > 3 else "13800138000"
    account_b = sys.argv[4] if len(sys.argv) > 4 else "13900139000"
    
    print("="*60)
    print("账号缓存匹配测试")
    print("="*60)
    print(f"设备ID: {device_id}")
    print(f"应用包名: {package_name}")
    print(f"账号A: {account_a}")
    print(f"账号B: {account_b}")
    
    asyncio.run(test_account_switch_with_cache_match(device_id, package_name, account_a, account_b))
