"""
详细诊断缓存恢复问题
检查文件权限、所有者、SELinux上下文等
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.adb_bridge import ADBBridge
from src.login_cache_manager import LoginCacheManager


async def diagnose_cache_restore():
    """详细诊断缓存恢复问题"""
    
    # 初始化ADB
    adb = ADBBridge()
    
    # 使用默认的MuMu模拟器设备ID（实例0）
    device_id = "127.0.0.1:16384"
    print(f"✓ 使用设备: {device_id}")
    
    # 尝试连接设备
    try:
        await adb.connect(device_id)
    except Exception as e:
        print(f"⚠️ 连接设备失败: {e}")
        print(f"提示：请确保MuMu模拟器实例0正在运行")
    
    # 初始化缓存管理器
    cache_manager = LoginCacheManager(adb)
    
    # 输入手机号
    phone = input("请输入手机号: ").strip()
    
    # 获取预期的user_id
    expected_user_id = cache_manager._get_expected_user_id(phone)
    print(f"✓ 预期user_id: {expected_user_id}")
    
    # 检查是否有缓存
    if not cache_manager.has_cache(phone, expected_user_id):
        print(f"❌ 未找到 {phone} 的缓存")
        return
    
    print(f"✓ 找到缓存")
    
    # 获取缓存信息
    cache_info = cache_manager.get_cache_info(phone, expected_user_id)
    print(f"\n缓存信息:")
    for key, value in cache_info.items():
        print(f"  {key}: {value}")
    
    # 应用包名
    package_name = "com.ry.xmsc"
    data_path = f"/data/data/{package_name}"
    
    print(f"\n开始诊断...")
    print("=" * 60)
    
    # 1. 检查应用是否运行
    print("\n1. 检查应用状态")
    result = await adb.shell(device_id, f"pidof {package_name}")
    if result.strip():
        print(f"  ⚠️ 应用正在运行 (PID: {result.strip()})")
        print(f"  建议：先停止应用再恢复缓存")
    else:
        print(f"  ✓ 应用未运行")
    
    # 2. 检查应用数据目录权限
    print("\n2. 检查应用数据目录")
    result = await adb.shell(device_id, f"su -c 'ls -ld {data_path}'")
    print(f"  {result.strip()}")
    
    # 获取应用的UID和GID
    uid_result = await adb.shell(device_id, f"su -c 'stat -c %u {data_path}'")
    uid = uid_result.strip()
    gid_result = await adb.shell(device_id, f"su -c 'stat -c %g {data_path}'")
    gid = gid_result.strip()
    print(f"  应用 UID:GID = {uid}:{gid}")
    
    # 3. 检查缓存文件是否存在
    print("\n3. 检查缓存文件")
    cache_files = [
        "shared_prefs/lcdpr.xml",
        "databases/DCStorage",
        "databases/DCStorage-shm",
        "databases/DCStorage-wal"
    ]
    
    for file_path in cache_files:
        full_path = f"{data_path}/{file_path}"
        result = await adb.shell(device_id, f"su -c 'test -f {full_path} && echo EXISTS || echo NOT_EXISTS'")
        
        if "EXISTS" in result:
            # 检查文件详细信息
            stat_result = await adb.shell(device_id, f"su -c 'ls -l {full_path}'")
            print(f"  ✓ {file_path}")
            print(f"    {stat_result.strip()}")
            
            # 检查SELinux上下文
            selinux_result = await adb.shell(device_id, f"su -c 'ls -Z {full_path}'")
            if selinux_result.strip():
                print(f"    SELinux: {selinux_result.strip()}")
        else:
            print(f"  ✗ {file_path} (不存在)")
    
    # 4. 检查文件内容（部分）
    print("\n4. 检查关键文件内容")
    
    # 检查 lcdpr.xml
    lcdpr_path = f"{data_path}/shared_prefs/lcdpr.xml"
    result = await adb.shell(device_id, f"su -c 'test -f {lcdpr_path} && cat {lcdpr_path} | head -20'")
    if result.strip():
        print(f"  lcdpr.xml (前20行):")
        for line in result.strip().split('\n')[:20]:
            print(f"    {line}")
    
    # 检查数据库文件大小
    db_path = f"{data_path}/databases/DCStorage"
    result = await adb.shell(device_id, f"su -c 'test -f {db_path} && stat -c %s {db_path}'")
    if result.strip():
        size = int(result.strip())
        print(f"\n  DCStorage 大小: {size} 字节 ({size/1024:.2f} KB)")
    
    # 5. 对比正确的文件权限
    print("\n5. 对比参考权限")
    print("  正确的权限应该是:")
    print(f"    所有者: {uid}:{gid}")
    print(f"    权限: -rw-rw---- (660)")
    print(f"    SELinux: u:object_r:app_data_file:s0:c512,c768")
    
    # 6. 测试恢复缓存
    print("\n6. 测试恢复缓存")
    confirm = input("是否测试恢复缓存？(y/n): ").strip().lower()
    
    if confirm == 'y':
        # 停止应用
        print("  停止应用...")
        await adb.stop_app(device_id, package_name)
        await asyncio.sleep(1)
        
        # 恢复缓存
        print("  恢复缓存...")
        success = await cache_manager.restore_login_cache(device_id, phone, user_id=expected_user_id)
        
        if success:
            print("  ✓ 缓存恢复成功")
            
            # 再次检查文件
            print("\n  恢复后的文件状态:")
            for file_path in cache_files:
                full_path = f"{data_path}/{file_path}"
                result = await adb.shell(device_id, f"su -c 'test -f {full_path} && ls -l {full_path}'")
                if result.strip():
                    print(f"    {file_path}:")
                    print(f"      {result.strip()}")
        else:
            print("  ✗ 缓存恢复失败")
    
    print("\n" + "=" * 60)
    print("诊断完成")


if __name__ == "__main__":
    asyncio.run(diagnose_cache_restore())
