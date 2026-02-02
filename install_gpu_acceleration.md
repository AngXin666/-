# GPU加速安装指南

## 当前状态
- ✅ NVIDIA RTX 3060 GPU
- ✅ PyTorch + CUDA 12.4（YOLO已支持GPU）
- ❌ TensorFlow CPU版本（页面分类器无GPU加速）

## 安装TensorFlow GPU版本

### 步骤1：卸载CPU版本
```bash
pip uninstall tensorflow
```

### 步骤2：安装GPU版本
```bash
# TensorFlow 2.20+ 自动包含GPU支持
pip install tensorflow[and-cuda]
```

或者指定版本：
```bash
pip install tensorflow==2.20.0
```

### 步骤3：验证GPU支持
```bash
python -c "import tensorflow as tf; print('GPU可用:', tf.config.list_physical_devices('GPU'))"
```

预期输出：
```
GPU可用: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

## 性能提升预期

### 安装前（CPU）
- 页面分类器：~180ms
- YOLO检测：~9ms
- **总耗时：~190ms**

### 安装后（GPU）
- 页面分类器：~20-30ms（**6倍提升**）
- YOLO检测：~2-3ms（**3倍提升**）
- **总耗时：~30ms**（**6倍提升**）

## 注意事项

1. **CUDA版本兼容性**
   - 您的PyTorch使用CUDA 12.4
   - TensorFlow 2.20+ 支持CUDA 12.x
   - 应该兼容

2. **显存占用**
   - 页面分类器：约500MB
   - YOLO模型：约100MB
   - RTX 3060有12GB显存，完全够用

3. **首次运行**
   - GPU首次运行会有预热时间（约1-2秒）
   - 后续运行会非常快

## 快速安装命令

```bash
# 一键安装（推荐）
pip uninstall -y tensorflow && pip install tensorflow[and-cuda]
```

## 验证安装

运行以下脚本验证GPU加速是否生效：

```python
import tensorflow as tf
import time

print("TensorFlow版本:", tf.__version__)
print("GPU可用:", tf.config.list_physical_devices('GPU'))

# 测试GPU性能
if tf.config.list_physical_devices('GPU'):
    print("\n✅ GPU加速已启用")
    
    # 简单性能测试
    with tf.device('/GPU:0'):
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        
        start = time.time()
        for _ in range(100):
            c = tf.matmul(a, b)
        gpu_time = time.time() - start
        print(f"GPU计算时间: {gpu_time:.3f}s")
else:
    print("\n❌ GPU加速未启用，使用CPU")
```

## 故障排除

### 问题1：找不到CUDA
```
Could not load dynamic library 'cudart64_12.dll'
```

**解决方案**：
1. 确认CUDA Toolkit已安装
2. 添加CUDA路径到系统PATH
3. 重启终端

### 问题2：显存不足
```
ResourceExhaustedError: OOM when allocating tensor
```

**解决方案**：
```python
# 在代码开头添加
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

### 问题3：版本冲突
```
ImportError: DLL load failed
```

**解决方案**：
1. 卸载所有TensorFlow相关包
2. 重新安装：`pip install tensorflow[and-cuda]`
3. 确保PyTorch和TensorFlow的CUDA版本兼容
