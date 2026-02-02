"""验证GPU支持"""
import tensorflow as tf

print("=" * 60)
print("TensorFlow GPU验证")
print("=" * 60)
print()

print(f"TensorFlow版本: {tf.__version__}")
print(f"CUDA构建: {tf.test.is_built_with_cuda()}")

gpus = tf.config.list_physical_devices('GPU')
print(f"GPU设备: {gpus}")
print(f"GPU数量: {len(gpus)}")

print()
if gpus:
    print("✅ GPU加速已启用！")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")
        # 尝试获取GPU详细信息
        try:
            details = tf.config.experimental.get_device_details(gpu)
            if details:
                print(f"    详细信息: {details}")
        except:
            pass
else:
    print("❌ GPU未检测到，使用CPU")
    print()
    print("可能的原因:")
    print("  1. CUDA Toolkit未安装")
    print("  2. cuDNN未安装")
    print("  3. TensorFlow版本与CUDA版本不兼容")

print()
print("=" * 60)
