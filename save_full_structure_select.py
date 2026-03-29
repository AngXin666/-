"""
保存完整的应用目录文件结构（可选择设备）
用于对比登录前后的所有文件变化
"""

import asyncio
import sys
from pathlib import Path
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from src.adb_bridge import ADBBridge
from src.emulator_controller import EmulatorController


async def get_all_files(adb, device_id, package_name):
    """获取应用目录下的所有文件信息"""
    data_path = f"/data/data/{package_name}"
    
    # 使用find命令获取所有文件
    result = await adb.shell(device_id, f"su -c 'find {data_path} -type f -exec stat -c \"%n|%s|%Y\" {{}} \\;'")
    
    files = {}
    for line in result.strip().split('\n'):
        if '|' in line:
            parts = line.split('|')
            if len(parts) >= 3:
                file_path = parts[0]
                try:
                    size = int(parts[1])
                    mtime = int(parts[2])
                    
                    # 转换为相对路径
                    rel_path = file_path.replace(data_path + '/', '')
                    
                    files[rel_path] = {
                        'size': size,
                        'mtime': mtime
                    }
                except:
                    pass
    
    return files


async def main():
    """主函数"""
    controller = EmulatorController()
    adb_path = controller.get_adb_path()
    
    if not adb_path:
        print("❌ 未找到ADB路径")
        return
    
    print(f"✓ ADB路径: {adb_path}")
    
    # 检测所有设备
    import subprocess
    result = subprocess.run([adb_path, "devices"], capture_output=True, text=True)
    
    devices = []
    for line in result.stdout.strip().split('\n')[1:]:
        if line.strip() and '\tdevice' in line:
            device_id = line.split('\t')[0]
            devices.append(device_id)
    
    if not devices:
        print("❌ 未找到ADB设备")
        return
    
    print(f"\n找到 {len(devices)} 个设备:")
    for i, device_id in enumerate(devices, 1):
        print(f"  {i}. {device_id}")
    
    # 让用户选择设备
    if len(devices) > 1:
        choice = input(f"\n请选择设备 (1-{len(devices)}): ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(devices):
                device_id = devices[idx]
            else:
                print("❌ 无效选择")
                return
        except:
            print("❌ 无效输入")
            return
    else:
        device_id = devices[0]
    
    print(f"\n✓ 使用设备: {device_id}")
    
    adb = ADBBridge(adb_path=adb_path)
    package_name = "com.ry.xmsc"
    
    try:
        await adb.connect(device_id)
    except:
        pass
    
    print("\n" + "=" * 60)
    print("保存完整文件结构")
    print("=" * 60)
    
    # 询问当前状态
    state = input("\n当前状态 (1=未登录, 2=已登录): ").strip()
    
    if state not in ['1', '2']:
        print("❌ 无效输入")
        return
    
    state_name = "before" if state == '1' else "after"
    
    print(f"\n正在扫描所有文件...")
    
    files = await get_all_files(adb, device_id, package_name)
    
    print(f"✓ 找到 {len(files)} 个文件")
    
    # 保存结果
    result_data = {
        'timestamp': datetime.now().isoformat(),
        'device_id': device_id,
        'package_name': package_name,
        'state': state_name,
        'file_count': len(files),
        'files': files
    }
    
    result_file = Path(f'full_structure_{state_name}.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 文件结构已保存到: {result_file}")
    
    # 如果两个状态都记录了，进行对比
    before_file = Path('full_structure_before.json')
    after_file = Path('full_structure_after.json')
    
    if before_file.exists() and after_file.exists():
        print("\n" + "=" * 60)
        print("对比登录前后的文件变化")
        print("=" * 60)
        
        with open(before_file, 'r', encoding='utf-8') as f:
            before_data = json.load(f)
        
        with open(after_file, 'r', encoding='utf-8') as f:
            after_data = json.load(f)
        
        before_files = before_data['files']
        after_files = after_data['files']
        
        # 找出新增的文件
        new_files = []
        for file_path, after_info in after_files.items():
            if file_path not in before_files:
                new_files.append((file_path, after_info))
        
        if new_files:
            print(f"\n新增的文件 ({len(new_files)} 个):")
            for file_path, info in sorted(new_files)[:20]:
                print(f"  + {file_path} ({info['size']} 字节)")
            if len(new_files) > 20:
                print(f"  ... 还有 {len(new_files) - 20} 个文件")
        
        # 找出修改的文件
        modified_files = []
        for file_path, after_info in after_files.items():
            if file_path in before_files:
                before_info = before_files[file_path]
                if after_info['size'] != before_info['size'] or after_info['mtime'] != before_info['mtime']:
                    modified_files.append({
                        'path': file_path,
                        'size_before': before_info['size'],
                        'size_after': after_info['size']
                    })
        
        if modified_files:
            print(f"\n修改的文件 ({len(modified_files)} 个):")
            for file_info in sorted(modified_files, key=lambda x: x['path'])[:20]:
                print(f"  * {file_info['path']}")
                print(f"    大小: {file_info['size_before']} -> {file_info['size_after']} 字节")
            if len(modified_files) > 20:
                print(f"  ... 还有 {len(modified_files) - 20} 个文件")
        
        # 找出删除的文件
        deleted_files = []
        for file_path in before_files:
            if file_path not in after_files:
                deleted_files.append(file_path)
        
        if deleted_files:
            print(f"\n删除的文件 ({len(deleted_files)} 个):")
            for file_path in sorted(deleted_files)[:20]:
                print(f"  - {file_path}")
            if len(deleted_files) > 20:
                print(f"  ... 还有 {len(deleted_files) - 20} 个文件")
        
        # 生成建议的CACHE_FILES
        all_changed = [f[0] for f in new_files] + [f['path'] for f in modified_files]
        
        # 过滤掉临时文件和缓存文件
        important_files = []
        for file_path in all_changed:
            # 排除临时文件、日志文件、缓存文件
            if any(x in file_path for x in ['cache/', 'temp', '.tmp', '.log', 'code_cache/']):
                continue
            # 只保留 shared_prefs, databases, files 目录下的文件
            if any(file_path.startswith(x) for x in ['shared_prefs/', 'databases/', 'files/']):
                important_files.append(file_path)
        
        if important_files:
            print("\n" + "=" * 60)
            print("建议的 CACHE_FILES 列表（已过滤临时文件）")
            print("=" * 60)
            print("\nCACHE_FILES = [")
            for file_path in sorted(important_files):
                print(f'    "{file_path}",')
            print("]")
    
    print("\n" + "=" * 60)
    print("完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
