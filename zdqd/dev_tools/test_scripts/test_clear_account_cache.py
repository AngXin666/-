"""
测试清理账号缓存功能
验证只清理账号缓存文件，不影响其他应用数据
"""

import asyncio
import sys
from pathlib import Path

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.adb_bridge import ADBBridge


async def check_files(adb: ADBBridge, device_id: str, package_name: str, label: str):
    """检查文件状态"""
    print(f"\n【{label}】文件状态")
    print("="*60)
    
    # 账号缓存文件
    account_cache_files = [
        f"/data/data/{package_name}/shared_prefs/lcdpr.xml",
        f"/data/data/{package_name}/databases/DCStorage",
        f"/data/data/{package_name}/databases/DCStorage-shm",
        f"/data/data/{package_name}/databases/DCStorage-wal"
    ]
    
    # 其他应用文件
    other_files = [
        f"/data/data/{package_name}/shared_prefs/",
        f"/data/data/{package_name}/databases/",
        f"/data/data/{package_name}/cache/",
        f"/data/data/{package_name}/files/",
    ]
    
    print("\n账号缓存文件：")
    account_status = {}
    for file_path in account_cache_files:
        result = await adb.shell(device_id, f"su -c 'test -f {file_path} && echo EXISTS || echo NOT_EXISTS'")
        exists = "EXISTS" in result
        account_status[file_path] = exists
        status = "✓ 存在" if exists else "❌ 不存在"
        print(f"  {status}: {file_path}")
    
    print("\n其他应用目录：")
    other_status = {}
    for dir_path in other_files:
        result = await adb.shell(device_id, f"su -c 'test -d {dir_path} && echo EXISTS || echo NOT_EXISTS'")
        exists = "EXISTS" in result
        other_status[dir_path] = exists
        status = "✓ 存在" if exists else "❌ 不存在"
        
        # 如果目录存在，显示文件数量
        if exists:
            count_result = await adb.shell(device_id, f"su -c 'ls -1 {dir_path} 2>/dev/null | wc -l'")
            try:
                count = int(count_result.strip())
                print(f"  {status}: {dir_path} (文件数: {count})")
            except:
                print(f"  {status}: {dir_path}")
        else:
            print(f"  {status}: {dir_path}")
    
    return account_status, other_status


async def clear_account_cache(adb: ADBBridge, device_id: str, package_name: str):
    """清理账号缓存文件"""
    print("\n清理账号缓存文件...")
    
    cache_files = [
        f"/data/data/{package_name}/shared_prefs/lcdpr.xml",
        f"/data/data/{package_name}/databases/DCStorage",
        f"/data/data/{package_name}/databases/DCStorage-shm",
        f"/data/data/{package_name}/databases/DCStorage-wal"
    ]
    
    # 方法1：逐个删除并验证
    print("\n【方法1】逐个删除并验证")
    for file_path in cache_files:
        print(f"\n尝试删除: {file_path}")
        result = await adb.shell(device_id, f"su -c 'rm -f {file_path}'")
        print(f"  删除命令输出: {result}")
        
        # 立即验证
        check = await adb.shell(device_id, f"su -c 'test -f {file_path} && echo EXISTS || echo NOT_EXISTS'")
        print(f"  验证结果: {check}")
    
    # 方法2：在同一个 shell 会话中删除并验证
    print("\n【方法2】在同一个 shell 会话中删除并验证")
    for file_path in cache_files:
        print(f"\n尝试删除: {file_path}")
        # 在同一个命令中删除并验证
        result = await adb.shell(device_id, f"su -c 'rm -f {file_path} && test -f {file_path} && echo STILL_EXISTS || echo DELETED'")
        print(f"  结果: {result}")
    
    print("\n✓ 账号缓存文件清理完成")


async def test_clear_account_cache(device_id: str = "127.0.0.1:5555", 
                                   package_name: str = "com.ry.xmsc"):
    """测试清理账号缓存"""
    
    adb = ADBBridge()
    
    # 连接设备
    print(f"连接设备: {device_id}")
    if not await adb.connect(device_id):
        print(f"❌ 无法连接到设备: {device_id}")
        return
    
    print(f"✓ 已连接到设备: {device_id}")
    
    # 1. 停止应用
    print("\n【步骤1】停止应用")
    await adb.stop_app(device_id, package_name)
    await asyncio.sleep(1)
    print("✓ 应用已停止")
    
    # 验证应用是否真的停止
    print("\n验证应用进程...")
    ps_result = await adb.shell(device_id, f"ps | grep {package_name}")
    if ps_result.strip():
        print(f"⚠️  应用进程仍在运行:")
        print(ps_result)
        
        # 强制杀死所有相关进程
        print("\n强制杀死所有进程...")
        await adb.shell(device_id, f"su -c 'killall -9 {package_name}'")
        await asyncio.sleep(1)
        
        # 再次验证
        ps_result2 = await adb.shell(device_id, f"ps | grep {package_name}")
        if ps_result2.strip():
            print(f"⚠️  进程仍然存在:")
            print(ps_result2)
        else:
            print("✓ 所有进程已停止")
    else:
        print("✓ 应用进程已停止")
    
    # 2. 检查清理前的文件状态
    account_before, other_before = await check_files(adb, device_id, package_name, "步骤2：清理前")
    
    # 3. 清理账号缓存
    print("\n【步骤3】清理账号缓存")
    await clear_account_cache(adb, device_id, package_name)
    
    # 立即检查（不等待）
    print("\n【步骤3.5】清理后立即检查")
    print("="*60)
    
    # 先检查应用进程
    print("\n检查应用进程...")
    ps_result = await adb.shell(device_id, f"ps | grep {package_name}")
    if ps_result.strip():
        print(f"⚠️  应用进程正在运行:")
        print(ps_result)
    else:
        print("✓ 应用进程未运行")
    
    print("\n账号缓存文件（立即检查）：")
    account_immediate = {}
    for file_path in [
        f"/data/data/{package_name}/shared_prefs/lcdpr.xml",
        f"/data/data/{package_name}/databases/DCStorage",
        f"/data/data/{package_name}/databases/DCStorage-shm",
        f"/data/data/{package_name}/databases/DCStorage-wal"
    ]:
        result = await adb.shell(device_id, f"su -c 'test -f {file_path} && echo EXISTS || echo NOT_EXISTS'")
        exists = "EXISTS" in result
        account_immediate[file_path] = exists
        status = "✓ 存在" if exists else "❌ 不存在"
        print(f"  {status}: {file_path}")
    
    print("\n等待2秒后检查...")
    await asyncio.sleep(2)
    
    # 4. 检查清理后的文件状态
    account_after, other_after = await check_files(adb, device_id, package_name, "步骤4：等待2秒后")
    
    # 5. 对比分析
    print("\n【步骤5】对比分析")
    print("="*60)
    
    print("\n账号缓存文件变化：")
    all_cleared = True
    for file_path in account_before.keys():
        before = "存在" if account_before[file_path] else "不存在"
        after = "存在" if account_after[file_path] else "不存在"
        
        print(f"\n{file_path}")
        print(f"  清理前: {before}")
        print(f"  清理后: {after}")
        
        if account_before[file_path] and not account_after[file_path]:
            print(f"  ✓ 成功清理")
        elif account_before[file_path] and account_after[file_path]:
            print(f"  ❌ 清理失败")
            all_cleared = False
        elif not account_before[file_path]:
            print(f"  - 本来就不存在")
    
    print("\n其他应用目录变化：")
    all_preserved = True
    for dir_path in other_before.keys():
        before = "存在" if other_before[dir_path] else "不存在"
        after = "存在" if other_after[dir_path] else "不存在"
        
        if other_before[dir_path] != other_after[dir_path]:
            print(f"\n{dir_path}")
            print(f"  清理前: {before}")
            print(f"  清理后: {after}")
            print(f"  ⚠️  状态改变")
            all_preserved = False
    
    if all_preserved:
        print("✓ 所有其他应用目录都保留")
    
    # 6. 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    if all_cleared:
        print("✅ 账号缓存文件全部清理成功")
    else:
        print("⚠️  部分账号缓存文件未清理")
    
    if all_preserved:
        print("✅ 其他应用数据全部保留")
    else:
        print("⚠️  部分应用数据被影响")
    
    if all_cleared and all_preserved:
        print("\n✅ 测试通过！只清理了账号缓存，其他数据保留")
    else:
        print("\n❌ 测试失败")


if __name__ == "__main__":
    # 从命令行参数获取设备ID和包名
    device_id = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:5555"
    package_name = sys.argv[2] if len(sys.argv) > 2 else "com.ry.xmsc"
    
    print("="*60)
    print("清理账号缓存测试")
    print("="*60)
    print(f"设备ID: {device_id}")
    print(f"应用包名: {package_name}")
    
    asyncio.run(test_clear_account_cache(device_id, package_name))
