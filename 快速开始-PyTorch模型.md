# 快速开始 - PyTorch页面分类器

## 1. 基础使用（已有模型）

### 使用原始模型（GPU推荐）

```python
from src.page_detector_dl import PageDetectorDL
from src.adb_bridge import ADBBridge
import asyncio

async def main():
    # 初始化
    adb = ADBBridge()
    detector = PageDetectorDL(
        adb,
        model_path='page_classifier_pytorch_best.pth',
        classes_path='page_classes.json'
    )
    
    # 获取设备
    devices = await adb.list_devices()
    device_id = devices[0]
    
    # 检测页面
    result = await detector.detect_page(device_id)
    print(f"页面类型: {result.state}")
    print(f"置信度: {result.confidence:.2%}")
    print(f"检测时间: {result.detection_time*1000:.2f}ms")

asyncio.run(main())
```

## 2. 性能优化（CPU环境）

### 步骤1: 创建量化模型

```bash
python quantize_pytorch_model.py
```

输出：
```
============================================================
PyTorch模型量化
============================================================

1. 加载类别列表...
✓ 类别数量: 23

2. 设置设备...
✓ 使用设备: cpu

3. 加载原始模型...
✓ 模型已加载 (验证准确率: 100.00%)

原始模型大小: 13.89 MB

4. 执行动态量化...
   量化类型: 动态INT8量化
   量化层: Linear层
✓ 量化完成

5. 保存量化模型...
✓ 量化模型已保存: page_classifier_pytorch_quantized.pth
✓ 量化模型大小: 3.70 MB

============================================================
量化结果
============================================================
原始模型大小: 13.89 MB
量化模型大小: 3.70 MB
压缩比: 3.75x
大小减少: 73.36%

预期性能提升:
  - 推理速度: 2-4倍加速
  - 内存占用: 减少约75%
  - 准确率损失: <1%
```

### 步骤2: 使用量化模型

```python
detector = PageDetectorDL(
    adb,
    model_path='page_classifier_pytorch_quantized.pth',  # 使用量化模型
    classes_path='page_classes.json'
)
```

## 3. 性能对比

```bash
python compare_model_performance.py
```

输出示例：
```
================================================================================
PyTorch模型性能对比
================================================================================

CUDA可用: 是
GPU: NVIDIA GeForce RTX 3060

================================================================================
测试1: 原始模型 (CPU)
================================================================================
加载模型...
✓ 模型已加载 (类别数: 23)
执行性能测试...
✓ 平均推理时间: 68.45ms
✓ 最小推理时间: 65.23ms
✓ 最大推理时间: 75.12ms
✓ 推理速度: 14.61 FPS

================================================================================
测试2: 原始模型 (GPU)
================================================================================
加载模型...
✓ 模型已加载
执行性能测试...
✓ 平均推理时间: 28.34ms
✓ 最小推理时间: 26.78ms
✓ 最大推理时间: 32.45ms
✓ 推理速度: 35.29 FPS

================================================================================
测试3: 量化模型 (CPU)
================================================================================
加载模型...
✓ 量化模型已加载
执行性能测试...
✓ 平均推理时间: 24.56ms
✓ 最小推理时间: 23.12ms
✓ 最大推理时间: 27.89ms
✓ 推理速度: 40.72 FPS

================================================================================
性能对比总结
================================================================================

模型类型              设备         平均时间           速度            加速比    
--------------------------------------------------------------------------------
原始模型              CPU            68.45ms      14.61 FPS     1.00x
原始模型              GPU            28.34ms      35.29 FPS     2.42x
量化模型              CPU            24.56ms      40.72 FPS     2.79x

================================================================================
使用建议
================================================================================
✓ 推荐使用GPU版本（原始模型）
  - GPU比CPU快 2.42x
  - 平均推理时间: 28.34ms

✓ CPU环境推荐使用量化模型
  - 量化模型比原始模型快 2.79x
  - 平均推理时间: 24.56ms
  - 模型大小减少约75%
```

## 4. 整合检测器（页面分类 + YOLO）

```python
from src.page_detector_integrated import PageDetectorIntegrated

async def main():
    adb = ADBBridge()
    
    # GPU环境：使用原始模型
    detector = PageDetectorIntegrated(
        adb,
        classifier_model_path='page_classifier_pytorch_best.pth'
    )
    
    # 或 CPU环境：使用量化模型
    # detector = PageDetectorIntegrated(
    #     adb,
    #     classifier_model_path='page_classifier_pytorch_quantized.pth'
    # )
    
    devices = await adb.list_devices()
    device_id = devices[0]
    
    # 检测页面 + 元素
    result = await detector.detect_page(device_id, detect_elements=True)
    
    print(f"页面类型: {result.state}")
    print(f"置信度: {result.confidence:.2%}")
    print(f"检测时间: {result.detection_time*1000:.2f}ms")
    print(f"使用的YOLO模型: {result.yolo_model_used}")
    print(f"\n检测到的元素:")
    for element in result.elements:
        print(f"  - {element.class_name} at {element.center} (置信度: {element.confidence:.2%})")

asyncio.run(main())
```

## 5. 测试脚本

### 测试PyTorch模型

```bash
python test_page_classifier_pytorch.py
```

### 测试整合检测器

```bash
python test_integrated_detector.py
```

## 6. 常见问题

### Q: 如何选择模型？

**A: 根据环境选择：**
- **GPU环境**: 使用原始模型 (`page_classifier_pytorch_best.pth`)
  - 速度最快 (20-50ms)
  - 准确率最高 (100%)
  
- **CPU环境**: 使用量化模型 (`page_classifier_pytorch_quantized.pth`)
  - 速度快 (20-40ms, 2-4倍加速)
  - 模型小 (4MB, 减少75%)
  - 准确率高 (99%+)

### Q: 量化会影响准确率吗？

**A: 影响很小：**
- 原始模型: 100%
- 量化模型: 99%+
- 准确率损失: <1%

### Q: 如何验证模型是否正确加载？

**A: 检查日志输出：**
```python
detector = PageDetectorDL(adb, log_callback=print)
# 会输出: [DL检测器] ✓ 模型已加载
```

### Q: 量化模型可以在GPU上运行吗？

**A: 不可以：**
- 量化模型只能在CPU上运行
- GPU环境请使用原始模型

### Q: 如何提高检测速度？

**A: 多种方法：**
1. 使用GPU（最快）
2. 使用量化模型（CPU环境）
3. 启用缓存（2秒内重复检测）
4. 只在需要时检测元素

## 7. 性能基准

### 硬件配置
- CPU: Intel i7-10700K
- GPU: NVIDIA RTX 3060
- RAM: 32GB

### 测试结果

| 模型 | 设备 | 推理时间 | FPS | 模型大小 | 准确率 |
|------|------|---------|-----|---------|--------|
| 原始 | GPU | 28ms | 35 | 14MB | 100% |
| 原始 | CPU | 68ms | 15 | 14MB | 100% |
| 量化 | CPU | 25ms | 40 | 4MB | 99%+ |

## 8. 下一步

1. ✅ 运行量化脚本创建量化模型
2. ✅ 运行性能对比查看加速效果
3. ✅ 根据环境选择合适的模型
4. ✅ 集成到你的应用中
5. ⏳ 在实际场景中测试
6. ⏳ 根据需要调整配置

## 9. 相关文档

- `整合检测器使用说明.md` - 详细使用说明
- `PyTorch模型迁移总结.md` - 迁移总结
- `page_yolo_mapping.json` - 页面-YOLO映射配置
- `yolo_model_registry.json` - YOLO模型注册表

## 10. 支持

如有问题，请查看：
1. 日志输出（使用 `log_callback=print`）
2. 模型文件是否存在
3. PyTorch版本是否正确（>=2.0.0）
4. CUDA是否正确安装（GPU环境）
