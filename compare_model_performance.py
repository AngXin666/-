"""
对比PyTorch模型性能 - 原始模型 vs 量化模型
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import time
import os


class PageClassifier(nn.Module):
    """页面分类器模型"""
    
    def __init__(self, num_classes):
        super(PageClassifier, self).__init__()
        self.mobilenet = models.mobilenet_v2(weights=None)
        in_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.mobilenet(x)


def load_model(model_path, classes_path, device='cpu', quantized=False):
    """加载模型"""
    # 加载类别
    with open(classes_path, 'r', encoding='utf-8') as f:
        classes = json.load(f)
    num_classes = len(classes)
    
    # 创建模型
    model = PageClassifier(num_classes)
    
    if quantized:
        # 量化模型
        model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8
        )
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model, classes


def benchmark_model(model, device, num_iterations=100):
    """性能测试"""
    # 准备测试数据
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_image = Image.new('RGB', (224, 224), color='white')
    image_tensor = transform(test_image).unsqueeze(0).to(device)
    
    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(image_tensor)
    
    # 测试速度
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.time()
            _ = model(image_tensor)
            end = time.time()
            times.append((end - start) * 1000)  # 转换为毫秒
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    return {
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'fps': 1000 / avg_time
    }


def compare_models():
    """对比模型性能"""
    print("=" * 80)
    print("PyTorch模型性能对比")
    print("=" * 80)
    print()
    
    classes_path = 'page_classes.json'
    original_path = 'page_classifier_pytorch_best.pth'
    quantized_path = 'page_classifier_pytorch_quantized.pth'
    
    # 检查文件
    if not os.path.exists(original_path):
        print(f"✗ 原始模型不存在: {original_path}")
        return
    
    if not os.path.exists(classes_path):
        print(f"✗ 类别文件不存在: {classes_path}")
        return
    
    # 检查CUDA
    has_cuda = torch.cuda.is_available()
    print(f"CUDA可用: {'是' if has_cuda else '否'}")
    if has_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    results = {}
    
    # 测试1: 原始模型 (CPU)
    print("=" * 80)
    print("测试1: 原始模型 (CPU)")
    print("=" * 80)
    print("加载模型...")
    model_cpu, classes = load_model(original_path, classes_path, device='cpu')
    print(f"✓ 模型已加载 (类别数: {len(classes)})")
    
    print("执行性能测试...")
    results['original_cpu'] = benchmark_model(model_cpu, 'cpu')
    print(f"✓ 平均推理时间: {results['original_cpu']['avg_time']:.2f}ms")
    print(f"✓ 最小推理时间: {results['original_cpu']['min_time']:.2f}ms")
    print(f"✓ 最大推理时间: {results['original_cpu']['max_time']:.2f}ms")
    print(f"✓ 推理速度: {results['original_cpu']['fps']:.2f} FPS")
    print()
    
    # 测试2: 原始模型 (GPU)
    if has_cuda:
        print("=" * 80)
        print("测试2: 原始模型 (GPU)")
        print("=" * 80)
        print("加载模型...")
        model_gpu, _ = load_model(original_path, classes_path, device='cuda')
        print("✓ 模型已加载")
        
        print("执行性能测试...")
        results['original_gpu'] = benchmark_model(model_gpu, 'cuda')
        print(f"✓ 平均推理时间: {results['original_gpu']['avg_time']:.2f}ms")
        print(f"✓ 最小推理时间: {results['original_gpu']['min_time']:.2f}ms")
        print(f"✓ 最大推理时间: {results['original_gpu']['max_time']:.2f}ms")
        print(f"✓ 推理速度: {results['original_gpu']['fps']:.2f} FPS")
        print()
    
    # 测试3: 量化模型 (CPU)
    if os.path.exists(quantized_path):
        print("=" * 80)
        print("测试3: 量化模型 (CPU)")
        print("=" * 80)
        print("加载模型...")
        model_quantized, _ = load_model(quantized_path, classes_path, device='cpu', quantized=True)
        print("✓ 量化模型已加载")
        
        print("执行性能测试...")
        results['quantized_cpu'] = benchmark_model(model_quantized, 'cpu')
        print(f"✓ 平均推理时间: {results['quantized_cpu']['avg_time']:.2f}ms")
        print(f"✓ 最小推理时间: {results['quantized_cpu']['min_time']:.2f}ms")
        print(f"✓ 最大推理时间: {results['quantized_cpu']['max_time']:.2f}ms")
        print(f"✓ 推理速度: {results['quantized_cpu']['fps']:.2f} FPS")
        print()
    else:
        print(f"⚠️ 量化模型不存在: {quantized_path}")
        print("   运行 'python quantize_pytorch_model.py' 来创建量化模型")
        print()
    
    # 显示对比结果
    print("=" * 80)
    print("性能对比总结")
    print("=" * 80)
    print()
    
    print(f"{'模型类型':<20} {'设备':<10} {'平均时间':<15} {'速度':<15} {'加速比':<10}")
    print("-" * 80)
    
    baseline = results['original_cpu']['avg_time']
    
    for key, result in results.items():
        model_type = "原始模型" if 'original' in key else "量化模型"
        device = "GPU" if 'gpu' in key else "CPU"
        avg_time = result['avg_time']
        fps = result['fps']
        speedup = baseline / avg_time
        
        print(f"{model_type:<20} {device:<10} {avg_time:>10.2f}ms {fps:>10.2f} FPS {speedup:>8.2f}x")
    
    print()
    
    # 显示建议
    print("=" * 80)
    print("使用建议")
    print("=" * 80)
    
    if has_cuda and 'original_gpu' in results:
        gpu_time = results['original_gpu']['avg_time']
        cpu_time = results['original_cpu']['avg_time']
        
        if gpu_time < cpu_time * 0.5:
            print("✓ 推荐使用GPU版本（原始模型）")
            print(f"  - GPU比CPU快 {cpu_time/gpu_time:.2f}x")
            print(f"  - 平均推理时间: {gpu_time:.2f}ms")
        else:
            print("⚠️ GPU加速效果不明显")
            print("  - 可能是因为模型较小，GPU开销大于收益")
    
    if 'quantized_cpu' in results:
        quantized_time = results['quantized_cpu']['avg_time']
        original_time = results['original_cpu']['avg_time']
        
        print()
        print("✓ CPU环境推荐使用量化模型")
        print(f"  - 量化模型比原始模型快 {original_time/quantized_time:.2f}x")
        print(f"  - 平均推理时间: {quantized_time:.2f}ms")
        print(f"  - 模型大小减少约75%")
    
    print()
    print("=" * 80)


if __name__ == '__main__':
    compare_models()
