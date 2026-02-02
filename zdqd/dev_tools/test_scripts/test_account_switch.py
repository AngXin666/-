"""
测试切换账号功能
模拟真实的账号切换流程：
1. 账号A登录并使用
2. 切换到账号B（清理缓存 + 恢复B的缓存）
3. 验证B的登录状态
4. 检查是否出现服务协议弹窗
"""

import asyncio
import sys
from pathlib import Path

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.adb_bridge import ADBBridge
from src.page_detector_hybrid import PageDetectorHybrid, PageState
from src.login_cache_manager import LoginCacheManager


async def simulate_account_usage(adb: ADBBridge, device_id: str, package_name: str, account_name: str):
    """模拟账号使用（创建一些缓存文件）"""
    print(f"\n模拟 {account_name} 使用应用...")
    
    # 启动应用
    success = await adb.start_app(device_id, package_name)
    if not success:
        print(f"❌ 应用启动失败")
        return False
    
    print(f"✓ 应用启动成功")
    await asyncio.sleep(3)
    
    # 检测页面状态
    detector = PageDetectorHybrid(adb)
    result = await detector.detect_page(device_id, use_ocr=True)
    print(f"当前页面: {result.state.value} - {result.details}")
    
    # 停止应用
    await adb.stop_app(device_id, package_name)
    await asyncio.sleep(1)
    
    return True


async def check_login_files(adb: ADBBridge, device_id: str, package_name: str, label: str):
    """检查登录文件状态"""
    print(f"\n【{label}】检查登录文件")
    
    login_files = [
        f"/data/data/{package_name}/databases/DCStorage",
        f"/data/data/{package_name}/shared_prefs/lcdpr.xml",
    ]
    
    files_status = {}
    for file_path in login_files:
        result = await adb.shell(device_id, f"su -c 'test -f {file_path} && echo EXISTS || echo NOT_EXISTS'")
        exists = "EXISTS" in result
        files_status[file_path] = exists
        status = "✓ 存在" if exists else "❌ 不存在"
        print(f"{status}: {file_path}")
    
    return files_status


async def switch_account(adb: ADBBridge, device_id: str, package_name: str, 
                        from_account: str, to_account: str):
    """切换账号（模拟GUI中的流程）"""
    print(f"\n{'='*60}")
    print(f"切换账号: {from_account} → {to_account}")
    print(f"{'='*60}")
    
    # 1. 停止应用
    print("\n【步骤1】停止应用")
    await adb.stop_app(device_id, package_name)
    await asyncio.sleep(1)
    print("✓ 应用已停止")
    
    # 2. 检查切换前的登录文件
    files_before = await check_login_files(adb, device_id, package_name, "步骤2：切换前")
    
    # 3. 清理缓存（使用新方法 - 只清理缓存，不清理数据）
    print("\n【步骤3】清理应用缓存（保留登录数据）")
    # 方法1：尝试使用 pm clear-cache
    result = await adb.shell(device_id, f"pm clear-cache {package_name}")
    print(f"pm clear-cache 结果: {result.strip() if result.strip() else '(无输出)'}")
    
    if "Unknown" in result or "Error" in result:
        # 方法2：如果不支持，使用 rm 命令
        print("pm clear-cache 不支持，使用 rm 命令...")
        result = await adb.shell(device_id, f"rm -rf /data/data/{package_name}/cache/*")
        print(f"rm 命令结果: {result.strip() if result.strip() else '成功（无输出）'}")
    
    await asyncio.sleep(2)
    
    # 4. 检查清理后的登录文件
    files_after_clear = await check_login_files(adb, device_id, package_name, "步骤4：清理后")
    
    # 5. 恢复新账号的缓存（如果有）
    print(f"\n【步骤5】恢复 {to_account} 的登录缓存")
    cache_manager = LoginCacheManager(adb)
    
    if cache_manager.has_cache(to_account):
        print(f"✓ 找到 {to_account} 的缓存")
        cache_info = cache_manager.get_cache_info(to_account)
        if cache_info:
            print(f"缓存保存于: {cache_info.get('saved_at', '未知')}")
        
        # 恢复缓存
        if await cache_manager.restore_login_cache(device_id, to_account, package_name):
            print(f"✓ {to_account} 的缓存恢复成功")
        else:
            print(f"❌ {to_account} 的缓存恢复失败")
    else:
        print(f"⚠️  未找到 {to_account} 的缓存")
    
    # 6. 检查恢复后的登录文件
    files_after_restore = await check_login_files(adb, device_id, package_name, "步骤6：恢复后")
    
    # 7. 启动应用
    print("\n【步骤7】启动应用")
    success = await adb.start_app(device_id, package_name)
    if not success:
        print("❌ 应用启动失败")
        return False
    
    print("✓ 应用启动成功，等待5秒...")
    await asyncio.sleep(5)
    
    # 8. 检测页面状态
    print("\n【步骤8】检测页面状态")
    detector = PageDetectorHybrid(adb)
    
    has_service_popup = False
    final_state = None
    
    for i in range(10):
        result = await detector.detect_page(device_id, use_ocr=True)
        print(f"[{i+1}/10] 页面状态: {result.state.value} - {result.details}")
        
        # 检查是否出现服务协议弹窗
        if result.state == PageState.POPUP and ("服务" in result.details or "协议" in result.details):
            print("\n❌ 检测到服务协议弹窗！")
            print("说明：切换账号时的清理方法有问题，清除了登录数据")
            has_service_popup = True
            break
        
        # 如果已经到达首页或个人页面，说明没有弹窗
        if result.state in [PageState.HOME, PageState.PROFILE, PageState.PROFILE_LOGGED]:
            print(f"\n✓ 成功到达 {result.state.value}，没有服务协议弹窗")
            final_state = result.state
            break
        
        await asyncio.sleep(1)
    
    # 9. 对比文件状态
    print("\n【步骤9】文件状态对比")
    print("\n登录文件变化：")
    for file_path in files_before.keys():
        before = "存在" if files_before[file_path] else "不存在"
        after_clear = "存在" if files_after_clear[file_path] else "不存在"
        after_restore = "存在" if files_after_restore[file_path] else "不存在"
        
        print(f"\n{file_path}")
        print(f"  切换前:   {before}")
        print(f"  清理后:   {after_clear}")
        print(f"  恢复后:   {after_restore}")
        
        # 验证清理后文件是否保留
        if files_before[file_path] and not files_after_clear[file_path]:
            print(f"  ❌ 清理时丢失了文件！")
        elif files_before[file_path] and files_after_clear[file_path]:
            print(f"  ✓ 清理后文件保留")
    
    # 10. 总结
    print("\n" + "="*60)
    print("切换账号测试总结")
    print("="*60)
    
    success = True
    
    # 检查是否出现服务协议弹窗
    if has_service_popup:
        print("❌ 出现服务协议弹窗 - 清理方法有问题")
        success = False
    else:
        print("✓ 没有服务协议弹窗")
    
    # 检查登录文件是否在清理后保留
    all_preserved = all(
        files_before[f] == files_after_clear[f] 
        for f in files_before.keys() 
        if files_before[f]  # 只检查原本存在的文件
    )
    
    if all_preserved:
        print("✓ 清理后所有登录文件都保留")
    else:
        print("❌ 清理后部分登录文件丢失")
        success = False
    
    # 检查最终状态
    if final_state in [PageState.HOME, PageState.PROFILE, PageState.PROFILE_LOGGED]:
        print(f"✓ 成功到达 {final_state.value}")
    else:
        print(f"⚠️  最终状态: {final_state.value if final_state else '未知'}")
    
    return success


async def test_account_switch(device_id: str = "127.0.0.1:5555", 
                              package_name: str = "com.ry.xmsc"):
    """测试账号切换"""
    
    adb = ADBBridge()
    
    # 连接设备
    print(f"连接设备: {device_id}")
    if not await adb.connect(device_id):
        print(f"❌ 无法连接到设备: {device_id}")
        return
    
    print(f"✓ 已连接到设备: {device_id}")
    
    # 模拟两个账号
    account_a = "13800138000"
    account_b = "13900139000"
    
    # 测试1：从账号A切换到账号B
    print("\n" + "="*60)
    print("测试场景1：从账号A切换到账号B")
    print("="*60)
    
    success1 = await switch_account(adb, device_id, package_name, account_a, account_b)
    
    # 等待一下
    await asyncio.sleep(2)
    
    # 测试2：从账号B切换回账号A
    print("\n" + "="*60)
    print("测试场景2：从账号B切换回账号A")
    print("="*60)
    
    success2 = await switch_account(adb, device_id, package_name, account_b, account_a)
    
    # 最终总结
    print("\n" + "="*60)
    print("最终测试结果")
    print("="*60)
    
    if success1 and success2:
        print("✅ 所有测试通过！")
        print("✓ 切换账号时不会出现服务协议弹窗")
        print("✓ 登录文件在清理后正确保留")
    else:
        print("❌ 测试失败！")
        if not success1:
            print("  - 场景1失败：A → B")
        if not success2:
            print("  - 场景2失败：B → A")


if __name__ == "__main__":
    # 从命令行参数获取设备ID和包名
    device_id = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:5555"
    package_name = sys.argv[2] if len(sys.argv) > 2 else "com.ry.xmsc"
    
    print("="*60)
    print("账号切换测试")
    print("="*60)
    print(f"设备ID: {device_id}")
    print(f"应用包名: {package_name}")
    
    asyncio.run(test_account_switch(device_id, package_name))
