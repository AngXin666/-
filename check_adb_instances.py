"""检查ADB检测到的实例数量"""
import subprocess
import sys

# 查找ADB路径
def find_adb():
    """查找ADB路径"""
    import os
    
    # 常见的ADB路径
    possible_paths = [
        r"C:\Program Files\Netease\MuMuPlayer-12.0\shell\adb.exe",
        r"D:\Program Files\Netease\MuMuPlayer-12.0\shell\adb.exe",
        r"E:\Program Files\Netease\MuMuPlayer-12.0\shell\adb.exe",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None

adb_path = find_adb()

if not adb_path:
    print("❌ 未找到ADB路径")
    sys.exit(1)

print(f"✅ 找到ADB: {adb_path}\n")

# 执行 adb devices 命令
print("=" * 60)
print("执行: adb devices")
print("=" * 60)

result = subprocess.run(
    [adb_path, "devices"],
    capture_output=True,
    text=True,
    encoding='utf-8',
    errors='ignore'
)

print(result.stdout)

# 解析结果
print("=" * 60)
print("解析结果:")
print("=" * 60)

adb_instances = []
for line in result.stdout.split('\n'):
    if '127.0.0.1:' in line and 'device' in line:
        parts = line.split()
        if len(parts) >= 2 and parts[1] == 'device':
            device_id = parts[0]
            try:
                port = int(device_id.split(':')[1])
                # MuMu端口规则：16384 + instance_id * 32
                if port >= 16384 and (port - 16384) % 32 == 0:
                    instance_id = (port - 16384) // 32
                    adb_instances.append(instance_id)
                    print(f"  实例 {instance_id}: {device_id}")
            except (ValueError, IndexError):
                continue

print(f"\n✅ ADB检测到 {len(adb_instances)} 个实例: {sorted(adb_instances)}")
