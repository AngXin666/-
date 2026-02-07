"""GPU诊断脚本 - 详细检查TensorFlow GPU配置"""
import os
import sys

print("=" * 60)
print("GPU诊断报告")
print("=" * 60)

# 1. 检查Python版本
print(f"\n1. Python版本: {sys.version}")

# 2. 检查TensorFlow版本
try:
    import tensorflow as tf
    print(f"\n2. TensorFlow版本: {tf.__version__}")
except ImportError as e:
    print(f"\n2. TensorFlow未安装: {e}")
    sys.exit(1)

# 3. 检查CUDA库
print("\n3. CUDA库检查:")
try:
    from tensorflow.python.platform import build_info
    print(f"   - CUDA版本: {build_info.build_info.get('cuda_version', 'N/A')}")
    print(f"   - cuDNN版本: {build_info.build_info.get('cudnn_version', 'N/A')}")
except Exception as e:
    print(f"   - 无法获取CUDA信息: {e}")

# 4. 检查GPU设备
print("\n4. GPU设备检查:")
gpus = tf.config.list_physical_devices('GPU')
print(f"   - 检测到的GPU数量: {len(gpus)}")
if gpus:
    for i, gpu in enumerate(gpus):
        print(f"   - GPU {i}: {gpu}")
else:
    print("   - 未检测到GPU设备")

# 5. 检查所有设备
print("\n5. 所有设备:")
all_devices = tf.config.list_physical_devices()
for device in all_devices:
    print(f"   - {device}")

# 6. 检查CUDA是否可用
print("\n6. CUDA可用性:")
print(f"   - tf.test.is_built_with_cuda(): {tf.test.is_built_with_cuda()}")
print(f"   - tf.test.is_gpu_available(): {tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)}")

# 7. 检查环境变量
print("\n7. 相关环境变量:")
env_vars = ['CUDA_PATH', 'CUDA_HOME', 'LD_LIBRARY_PATH', 'PATH']
for var in env_vars:
    value = os.environ.get(var, 'Not set')
    if var == 'PATH' and value != 'Not set':
        # 只显示包含cuda或nvidia的路径
        paths = value.split(';')
        cuda_paths = [p for p in paths if 'cuda' in p.lower() or 'nvidia' in p.lower()]
        if cuda_paths:
            print(f"   - {var} (CUDA相关):")
            for p in cuda_paths[:5]:  # 只显示前5个
                print(f"     * {p}")
        else:
            print(f"   - {var}: 无CUDA相关路径")
    else:
        print(f"   - {var}: {value}")

# 8. 检查nvidia-smi
print("\n8. nvidia-smi检查:")
try:
    import subprocess
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'], 
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print(f"   - GPU信息: {result.stdout.strip()}")
    else:
        print(f"   - nvidia-smi执行失败")
except Exception as e:
    print(f"   - nvidia-smi不可用: {e}")

# 9. 检查CUDA库文件
print("\n9. CUDA库文件检查:")
import site
site_packages = site.getsitepackages()
for sp in site_packages:
    nvidia_path = os.path.join(sp, 'nvidia')
    if os.path.exists(nvidia_path):
        print(f"   - 找到nvidia目录: {nvidia_path}")
        for item in os.listdir(nvidia_path):
            item_path = os.path.join(nvidia_path, item)
            if os.path.isdir(item_path):
                bin_path = os.path.join(item_path, 'bin')
                if os.path.exists(bin_path):
                    dll_files = [f for f in os.listdir(bin_path) if f.endswith('.dll')]
                    print(f"     * {item}/bin: {len(dll_files)} DLL文件")

# 10. 尝试创建简单的GPU操作
print("\n10. GPU操作测试:")
try:
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
        print(f"   - GPU矩阵乘法成功: {c.numpy()}")
except Exception as e:
    print(f"   - GPU操作失败: {e}")

print("\n" + "=" * 60)
print("诊断完成")
print("=" * 60)
