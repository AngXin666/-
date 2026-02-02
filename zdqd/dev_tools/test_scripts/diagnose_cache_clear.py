"""
诊断缓存清理问题 - 检查清理前后的文件状态
"""

import asyncio
import sys
from pathlib import Path

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.adb_bridge import ADBBridge


async def check_files(adb: ADBBridge, device_id: str, package_name: str):
    """检查应用数据目录中的文件"""
    print("\n" + "="*60)
    print("检查应用数据目录文件")
    print("="*60)
    
    # 检查的目录和文件
    paths_to_check = [
        f"/data/data/{package_name}/",
        f"/data/data/{package_name}/cache/",
        f"/data/data/{package_name}/databases/",
        f"/data/data/{package_name}/shared_prefs/",
        f"/data/data/{package_name}/databases/DCStorage",
        f"/data/data/{package_name}/shared_prefs/lcdpr.xml",
    ]
    
    for path in paths_to_check:
        # 检查是否存在
        result = await adb.shell(device_id, f"su -c 'ls -la {path} 2>&1'")
        
        if "No such file" in result:
            print(f"❌ 不存在: {path}")
        elif "Permission denied" in result:
            print(f"⚠️  权限拒绝: {path}")
        else:
            print(f"✓ 存在: {path}")
            # 如果是目录，显示文件数量
            if path.endswith('/'):
                file_count = len([line for line in result.split('\n') if line.strip() and not line.startswith('total')])
                print(f"  文件数量: {file_count}")


async def test_cache_clear_methods(device_id: str = "127.0.0.1:5555", package_name: str = "com.ry.xmsc"):
    """测试不同的缓存清理方法"""
    
    adb = ADBBridge()
    
    # 连接设备
    print(f"连接设备: {device_id}")
    if not await adb.connect(device_id):
        print(f"❌ 无法连接到设备: {device_id}")
        return
    
    print(f"✓ 已连接到设备: {device_id}")
    
    # 1. 检查清理前的状态
    print("\n【步骤1】清理前的文件状态")
    await check_files(adb, device_id, package_name)
    
    # 2. 测试 pm clear-cache
    print("\n【步骤2】测试 pm clear-cache 命令")
    result = await adb.shell(device_id, f"pm clear-cache {package_name}")
    print(f"执行结果: {result}")
    
    if "Unknown" in result or "Error" in result:
        print("⚠️  pm clear-cache 不支持，尝试 rm 命令")
        result = await adb.shell(device_id, f"rm -rf /data/data/{package_name}/cache/*")
        print(f"rm 命令结果: {result.strip() if result.strip() else '成功（无输出）'}")
    
    # 3. 检查清理后的状态
    print("\n【步骤3】清理后的文件状态")
    await check_files(adb, device_id, package_name)
    
    # 4. 检查关键登录文件是否还在
    print("\n【步骤4】验证登录文件是否保留")
    login_files = [
        f"/data/data/{package_name}/databases/DCStorage",
        f"/data/data/{package_name}/shared_prefs/lcdpr.xml",
    ]
    
    all_exist = True
    for file_path in login_files:
        result = await adb.shell(device_id, f"su -c 'test -f {file_path} && echo EXISTS || echo NOT_EXISTS'")
        if "EXISTS" in result:
            print(f"✓ 登录文件保留: {file_path}")
        else:
            print(f"❌ 登录文件丢失: {file_path}")
            all_exist = False
    
    # 5. 总结
    print("\n" + "="*60)
    print("诊断总结")
    print("="*60)
    if all_exist:
        print("✓ 所有登录文件都保留，清理方法正确")
    else:
        print("❌ 登录文件丢失，清理方法有问题")
        print("\n建议：")
        print("1. 检查是否使用了 pm clear 而不是 pm clear-cache")
        print("2. 检查 rm 命令是否误删了 databases 或 shared_prefs 目录")
        print("3. 确认应用包名是否正确")


if __name__ == "__main__":
    # 从命令行参数获取设备ID和包名
    device_id = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:5555"
    package_name = sys.argv[2] if len(sys.argv) > 2 else "com.ry.xmsc"
    
    print("="*60)
    print("缓存清理诊断工具")
    print("="*60)
    print(f"设备ID: {device_id}")
    print(f"应用包名: {package_name}")
    
    asyncio.run(test_cache_clear_methods(device_id, package_name))
