"""
找出应用真正使用的所有登录相关文件
对比登录前后的文件变化
"""

import asyncio
import sys
from pathlib import Path
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from src.adb_bridge import ADBBridge
from src.emulator_controller import EmulatorController


async def get_all_app_files(adb, device_id, package_name):
    """获取应用目录下的所有文件及其修改时间和大小"""
    data_path = f"/data/data/{package_name}"
    
    # 使用find命令来获取所有文件信息
    result = await adb.shell(device_id, f"su -c 'find {data_path} -type f -exec ls -l {{}} \\;'")
    
    print(f"\n[调试] 命令输出前100个字符: {result[:100]}")
    print(f"[调试] 输出总长度: {len(result)}")
    
    files = {}
    
    for line in result.strip().split('\n'):
        line = line.strip()
        
        # 跳过空行
        if not line:
            continue
        
        # 解析文件信息行: -rw------- 1 u0_a123 u0_a123 1234 2024-01-01 12:00 /data/data/com.ry.xmsc/file.txt
        parts = line.split()
        if len(parts) >= 9 and parts[0].startswith('-'):
            try:
                size = int(parts[4])
                # 文件路径是最后一个部分
                file_path = parts[-1]
                
                files[file_path] = {
                    'size': size,
                    'mtime': 0
                }
            except Exception as e:
                print(f"[调试] 解析失败: {line[:50]}... 错误: {e}")
                pass
    
    return files


async def main():
    """主函数"""
    # 初始化模拟器控制器
    from src.emulator_controller import EmulatorController
    controller = EmulatorController()
    adb_path = controller.get_adb_path()
    
    if not adb_path:
        print("❌ 未找到ADB路径")
        return
    
    print(f"✓ ADB路径: {adb_path}")
    
    # 直接用ADB检测设备
    print("\n检测ADB设备...")
    import subprocess
    result = subprocess.run([adb_path, "devices"], capture_output=True, text=True)
    
    devices = []
    for line in result.stdout.strip().split('\n')[1:]:
        if line.strip() and '\tdevice' in line:
            device_id = line.split('\t')[0]
            devices.append(device_id)
    
    if not devices:
        print("❌ 未找到ADB设备")
        print("ADB输出:")
        print(result.stdout)
        return
    
    print(f"✓ 找到 {len(devices)} 个设备:")
    for device_id in devices:
        print(f"  {device_id}")
    
    # 使用第一个设备
    device_id = devices[0]
    print(f"\n使用设备: {device_id}")
    
    adb = ADBBridge(adb_path=adb_path)
    package_name = "com.ry.xmsc"
    
    print("=" * 60)
    print("找出应用真正使用的所有登录相关文件")
    print("=" * 60)
    
    try:
        await adb.connect(device_id)
    except Exception as e:
        print(f"⚠️ 连接设备失败: {e}")
    
    print("\n请按照以下步骤操作：")
    print("1. 确保应用当前是未登录状态")
    print("2. 按回车键开始记录登录前的文件状态")
    input("按回车继续...")
    
    print("\n正在记录登录前的文件状态...")
    files_before = await get_all_app_files(adb, device_id, package_name)
    print(f"✓ 记录了 {len(files_before)} 个文件")
    
    print("\n请执行以下操作：")
    print("1. 手动登录应用")
    print("2. 登录成功后，按回车键记录登录后的文件状态")
    input("按回车继续...")
    
    print("\n正在记录登录后的文件状态...")
    files_after = await get_all_app_files(adb, device_id, package_name)
    print(f"✓ 记录了 {len(files_after)} 个文件")
    
    print("\n" + "=" * 60)
    print("分析文件变化")
    print("=" * 60)
    
    # 找出新增的文件
    new_files = set(files_after.keys()) - set(files_before.keys())
    if new_files:
        print(f"\n新增的文件 ({len(new_files)} 个):")
        for file_path in sorted(new_files):
            size = files_after[file_path]['size']
            print(f"  + {file_path} ({size} 字节)")
    
    # 找出修改的文件
    modified_files = []
    for file_path in files_before.keys():
        if file_path in files_after:
            before = files_before[file_path]
            after = files_after[file_path]
            if before['size'] != after['size'] or before['mtime'] != after['mtime']:
                modified_files.append({
                    'path': file_path,
                    'size_before': before['size'],
                    'size_after': after['size'],
                    'mtime_before': before['mtime'],
                    'mtime_after': after['mtime']
                })
    
    if modified_files:
        print(f"\n修改的文件 ({len(modified_files)} 个):")
        for file_info in sorted(modified_files, key=lambda x: x['path']):
            path = file_info['path']
            size_before = file_info['size_before']
            size_after = file_info['size_after']
            print(f"  * {path}")
            print(f"    大小: {size_before} -> {size_after} 字节")
    
    # 找出删除的文件
    deleted_files = set(files_before.keys()) - set(files_after.keys())
    if deleted_files:
        print(f"\n删除的文件 ({len(deleted_files)} 个):")
        for file_path in sorted(deleted_files):
            print(f"  - {file_path}")
    
    # 保存结果到文件
    result = {
        'timestamp': datetime.now().isoformat(),
        'device_id': device_id,
        'package_name': package_name,
        'new_files': list(new_files),
        'modified_files': modified_files,
        'deleted_files': list(deleted_files)
    }
    
    result_file = Path('login_files_analysis.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 分析结果已保存到: {result_file}")
    
    # 生成建议的缓存文件列表
    print("\n" + "=" * 60)
    print("建议的缓存文件列表")
    print("=" * 60)
    
    all_changed_files = list(new_files) + [f['path'] for f in modified_files]
    
    # 转换为相对路径
    data_path = f"/data/data/{package_name}"
    relative_paths = []
    for file_path in all_changed_files:
        if file_path.startswith(data_path + '/'):
            rel_path = file_path[len(data_path) + 1:]
            relative_paths.append(rel_path)
    
    print("\nCACHE_FILES = [")
    for rel_path in sorted(relative_paths):
        print(f'    "{rel_path}",')
    print("]")
    
    print("\n" + "=" * 60)
    print("完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
