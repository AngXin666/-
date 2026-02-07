"""
快速检查GPU状态
"""
import torch
import time

print("=" * 60)
print("🔍 GPU状态检查")
print("=" * 60)

# 检查CUDA是否可用
print(f"\n1. CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"2. GPU设备数量: {torch.cuda.device_count()}")
    print(f"3. 当前GPU: {torch.cuda.current_device()}")
    print(f"4. GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"5. CUDA版本: {torch.version.cuda}")
    print(f"6. cuDNN版本: {torch.backends.cudnn.version()}")
    print(f"7. cuDNN启用: {torch.backends.cudnn.enabled}")
    
    # 显存信息
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    cached = torch.cuda.memory_reserved(0) / 1024**3
    
    print(f"\n显存信息:")
    print(f"  • 总显存: {total_memory:.2f} GB")
    print(f"  • 已分配: {allocated:.2f} GB")
    print(f"  • 已缓存: {cached:.2f} GB")
    print(f"  • 可用: {total_memory - allocated:.2f} GB")
    
    # 性能测试
    print(f"\n性能测试:")
    
    # CPU测试
    print(f"  测试1: CPU矩阵乘法...")
    cpu_tensor = torch.randn(2000, 2000)
    start = time.time()
    for _ in range(10):
        _ = cpu_tensor @ cpu_tensor
    cpu_time = time.time() - start
    print(f"    CPU耗时: {cpu_time:.3f}秒")
    
    # GPU测试
    print(f"  测试2: GPU矩阵乘法...")
    gpu_tensor = torch.randn(2000, 2000).cuda()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        _ = gpu_tensor @ gpu_tensor
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"    GPU耗时: {gpu_time:.3f}秒")
    
    speedup = cpu_time / gpu_time
    print(f"\n  ⚡ GPU加速比: {speedup:.1f}x")
    
    if speedup < 2:
        print(f"\n  ⚠️  警告: GPU加速效果不明显")
        print(f"     可能原因:")
        print(f"     1. 使用的是集成显卡或低端显卡")
        print(f"     2. GPU驱动未正确安装")
        print(f"     3. PyTorch未正确安装CUDA版本")
    else:
        print(f"\n  ✓ GPU工作正常!")
    
    # 混合精度测试
    print(f"\n  测试3: 混合精度训练(AMP)...")
    try:
        scaler = torch.cuda.amp.GradScaler()
        with torch.cuda.amp.autocast():
            result = gpu_tensor @ gpu_tensor
        print(f"    ✓ AMP支持正常")
    except Exception as e:
        print(f"    ✗ AMP不支持: {e}")
    
else:
    print("\n❌ 未检测到CUDA支持的GPU")
    print("\n可能的原因:")
    print("1. 未安装CUDA版本的PyTorch")
    print("2. 显卡驱动未正确安装")
    print("3. 显卡不支持CUDA")
    
    print("\n解决方案:")
    print("1. 卸载当前PyTorch:")
    print("   pip uninstall torch torchvision")
    print("\n2. 安装CUDA版本的PyTorch:")
    print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    print("\n3. 检查显卡驱动:")
    print("   运行 nvidia-smi 查看显卡状态")

print("\n" + "=" * 60)
