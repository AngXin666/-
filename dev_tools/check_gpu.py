import torch

print("=" * 60)
print("GPU环境检查")
print("=" * 60)

print(f"\nPyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"cuDNN版本: {torch.backends.cudnn.version()}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  名称: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  显存: {props.total_memory / 1024**3:.1f} GB")
        print(f"  计算能力: {props.major}.{props.minor}")
else:
    print("\n⚠️  CUDA不可用！")
    print("可能的原因:")
    print("  1. 安装的是CPU版本的PyTorch")
    print("  2. CUDA驱动未安装或版本不匹配")
    print("  3. 没有NVIDIA GPU")

print("\n" + "=" * 60)
