"""
记录登录前后的文件状态
用于对比找出真正的登录相关文件
"""

import asyncio
import sys
from pathlib import Path
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from src.adb_bridge import ADBBridge
from src.emulator_controller import EmulatorController


async def get_file_info(adb, device_id, file_path):
    """获取文件信息（大小和修改时间）"""
    result = await adb.shell(device_id, f"su -c 'stat -c \"%s %Y\" {file_path} 2>/dev/null'")
    if result.strip() and not result.startswith('stat:'):
        parts = result.strip().split()
        if len(parts) >= 2:
            return {
                'size': int(parts[0]),
                'mtime': int(parts[1]),
                'exists': True
            }
    return {'exists': False}


async def main():
    """主函数"""
    controller = EmulatorController()
    adb_path = controller.get_adb_path()
    
    if not adb_path:
        print("❌ 未找到ADB路径")
        return
    
    # 检测设备
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
    
    device_id = devices[0]
    print(f"✓ 使用设备: {device_id}")
    
    adb = ADBBridge(adb_path=adb_path)
    package_name = "com.ry.xmsc"
    data_path = f"/data/data/{package_name}"
    
    try:
        await adb.connect(device_id)
    except:
        pass
    
    # 要检查的文件列表
    files_to_check = [
        # 当前保存的文件
        "shared_prefs/lcdpr.xml",
        "databases/DCStorage",
        "databases/DCStorage-shm",
        "databases/DCStorage-wal",
        # 可能重要的文件
        "shared_prefs/com.ry.xmsc_preferences.xml",
        "shared_prefs/ContextData.xml",
        "shared_prefs/virtualImeiAndImsi.xml",
        "shared_prefs/vkeyid_settings.xml",
        "shared_prefs/vkeyid_profiles_v3.xml",
        "shared_prefs/vkeyid_profiles_v4.xml",
        "shared_prefs/Alvin2.xml",
        "shared_prefs/alipay_tid_storage.xml",
        "shared_prefs/alipay_vkey_random.xml",
        "files/.DC4278477faeb9.txt",
    ]
    
    print("\n" + "=" * 60)
    print("记录文件状态")
    print("=" * 60)
    
    # 询问当前状态
    state = input("\n当前状态 (1=未登录, 2=已登录): ").strip()
    
    if state not in ['1', '2']:
        print("❌ 无效输入")
        return
    
    state_name = "before_login" if state == '1' else "after_login"
    
    print(f"\n正在记录{state_name}状态...")
    
    file_states = {}
    for file_path in files_to_check:
        full_path = f"{data_path}/{file_path}"
        info = await get_file_info(adb, device_id, full_path)
        file_states[file_path] = info
        
        if info['exists']:
            print(f"  ✓ {file_path} ({info['size']} 字节)")
        else:
            print(f"  ✗ {file_path} (不存在)")
    
    # 保存结果
    result_data = {
        'timestamp': datetime.now().isoformat(),
        'device_id': device_id,
        'package_name': package_name,
        'state': state_name,
        'files': file_states
    }
    
    result_file = Path(f'login_state_{state_name}.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 状态已保存到: {result_file}")
    
    # 如果两个状态都记录了，进行对比
    before_file = Path('login_state_before_login.json')
    after_file = Path('login_state_after_login.json')
    
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
            before_info = before_files.get(file_path, {'exists': False})
            if after_info['exists'] and not before_info['exists']:
                new_files.append(file_path)
        
        if new_files:
            print(f"\n新增的文件 ({len(new_files)} 个):")
            for file_path in new_files:
                info = after_files[file_path]
                print(f"  + {file_path} ({info['size']} 字节)")
        
        # 找出修改的文件
        modified_files = []
        for file_path, after_info in after_files.items():
            before_info = before_files.get(file_path, {'exists': False})
            if after_info['exists'] and before_info['exists']:
                if after_info['size'] != before_info['size'] or after_info['mtime'] != before_info['mtime']:
                    modified_files.append({
                        'path': file_path,
                        'size_before': before_info['size'],
                        'size_after': after_info['size']
                    })
        
        if modified_files:
            print(f"\n修改的文件 ({len(modified_files)} 个):")
            for file_info in modified_files:
                print(f"  * {file_info['path']}")
                print(f"    大小: {file_info['size_before']} -> {file_info['size_after']} 字节")
        
        # 生成建议的CACHE_FILES
        all_changed = new_files + [f['path'] for f in modified_files]
        if all_changed:
            print("\n" + "=" * 60)
            print("建议的 CACHE_FILES 列表")
            print("=" * 60)
            print("\nCACHE_FILES = [")
            for file_path in sorted(all_changed):
                print(f'    "{file_path}",')
            print("]")
    
    print("\n" + "=" * 60)
    print("完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
