"""诊断缓存恢复问题"""
import asyncio
from src.adb_bridge import ADBBridge
from src.login_cache_manager import LoginCacheManager
from pathlib import Path

async def diagnose_cache(phone: str, device_id: str = "127.0.0.1:16416"):
    """诊断指定账号的缓存问题
    
    Args:
        phone: 手机号
        device_id: 设备ID
    """
    print(f"=" * 60)
    print(f"诊断账号 {phone} 的缓存问题")
    print(f"=" * 60)
    
    # 初始化
    adb = ADBBridge()
    
    # 初始化ADB连接
    devices = adb.list_devices()
    if not devices:
        print("❌ 没有检测到设备")
        return
    
    print(f"检测到设备: {devices}")
    
    cache_manager = LoginCacheManager(adb)
    
    # 1. 检查本地缓存文件
    print("\n1. 检查本地缓存文件:")
    print("-" * 60)
    
    # 获取预期的user_id
    expected_user_id = cache_manager._get_expected_user_id(phone)
    print(f"预期用户ID: {expected_user_id}")
    
    # 检查缓存目录
    if expected_user_id:
        cache_dir = cache_manager._get_account_cache_dir(phone, expected_user_id)
    else:
        cache_dir = cache_manager._get_account_cache_dir(phone)
    
    print(f"缓存目录: {cache_dir}")
    
    if not cache_dir.exists():
        print("❌ 缓存目录不存在")
        return
    
    print("✓ 缓存目录存在")
    
    # 检查每个缓存文件
    print("\n缓存文件列表:")
    for file_path in cache_manager.CACHE_FILES:
        cache_file_name = file_path.replace('/', '_')
        encrypted_file = cache_dir / f"{cache_file_name}.enc"
        
        is_required = file_path in cache_manager.REQUIRED_FILES
        status = "必需" if is_required else "可选"
        
        if encrypted_file.exists():
            size = encrypted_file.stat().st_size
            print(f"  ✓ {file_path} ({status}) - {size} 字节")
        else:
            if is_required:
                print(f"  ❌ {file_path} ({status}) - 不存在")
            else:
                print(f"  - {file_path} ({status}) - 不存在（正常）")
    
    # 2. 检查设备上的应用数据
    print("\n2. 检查设备上的应用数据:")
    print("-" * 60)
    
    package_name = "com.ry.xmsc"
    data_path = f"/data/data/{package_name}"
    
    # 检查应用数据目录是否存在
    result = await adb.shell(device_id, f"su -c 'test -d {data_path} && echo EXISTS || echo NOT_EXISTS'")
    if "EXISTS" not in result:
        print(f"❌ 应用数据目录不存在: {data_path}")
        return
    
    print(f"✓ 应用数据目录存在: {data_path}")
    
    # 检查每个缓存文件在设备上是否存在
    print("\n设备上的文件:")
    for file_path in cache_manager.CACHE_FILES:
        source_path = f"{data_path}/{file_path}"
        result = await adb.shell(device_id, f"su -c 'test -f {source_path} && echo EXISTS || echo NOT_EXISTS'")
        
        is_required = file_path in cache_manager.REQUIRED_FILES
        status = "必需" if is_required else "可选"
        
        if "EXISTS" in result:
            # 获取文件大小
            size_result = await adb.shell(device_id, f"su -c 'stat -c %s {source_path}'")
            size = size_result.strip()
            
            # 获取文件权限
            perm_result = await adb.shell(device_id, f"su -c 'stat -c %a {source_path}'")
            perm = perm_result.strip()
            
            # 获取文件所有者
            owner_result = await adb.shell(device_id, f"su -c 'stat -c %U:%G {source_path}'")
            owner = owner_result.strip()
            
            print(f"  ✓ {file_path} ({status})")
            print(f"    大小: {size} 字节, 权限: {perm}, 所有者: {owner}")
        else:
            if is_required:
                print(f"  ❌ {file_path} ({status}) - 不存在")
            else:
                print(f"  - {file_path} ({status}) - 不存在（正常）")
    
    # 3. 测试缓存恢复
    print("\n3. 测试缓存恢复:")
    print("-" * 60)
    
    if not cache_manager.has_cache(phone, expected_user_id):
        print("❌ 没有可用的缓存")
        return
    
    print("✓ 检测到缓存，开始恢复测试...")
    
    # 停止应用
    print("停止应用...")
    await adb.stop_app(device_id, package_name)
    await asyncio.sleep(1)
    
    # 恢复缓存
    print("恢复缓存...")
    success = await cache_manager.restore_login_cache(device_id, phone, user_id=expected_user_id)
    
    if success:
        print("✓ 缓存恢复成功")
    else:
        print("❌ 缓存恢复失败")
        return
    
    # 验证恢复后的文件
    print("\n恢复后验证:")
    for file_path in cache_manager.REQUIRED_FILES:
        source_path = f"{data_path}/{file_path}"
        result = await adb.shell(device_id, f"su -c 'test -f {source_path} && echo EXISTS || echo NOT_EXISTS'")
        
        if "EXISTS" in result:
            # 获取文件大小
            size_result = await adb.shell(device_id, f"su -c 'stat -c %s {source_path}'")
            size = size_result.strip()
            print(f"  ✓ {file_path} - {size} 字节")
        else:
            print(f"  ❌ {file_path} - 不存在")
    
    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python diagnose_cache_restore.py <手机号> [设备ID]")
        print("示例: python diagnose_cache_restore.py 18825724627")
        sys.exit(1)
    
    phone = sys.argv[1]
    device_id = sys.argv[2] if len(sys.argv) > 2 else "127.0.0.1:16416"
    
    asyncio.run(diagnose_cache(phone, device_id))
