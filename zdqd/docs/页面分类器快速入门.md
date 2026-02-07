# 分页器快速开始指南

## 概述

分页器（Page Classifier）已经完全集成到主程序中，无需任何额外配置即可使用。

## 自动使用

主程序会自动使用整合检测器（PageDetectorIntegrated），它内置了分页器功能：

```python
# 在 src/ximeng_automation.py 中
from .model_manager import ModelManager

model_manager = ModelManager.get_instance()

# 自动加载整合检测器（包含分页器）
self.integrated_detector = model_manager.get_page_detector_integrated()
self.detector = self.integrated_detector  # 优先使用整合检测器
```

## 工作流程

1. **启动主程序** → 自动加载分页器模型
2. **检测页面** → 使用GPU加速的分页器识别页面类型（20-50ms）
3. **执行操作** → 根据页面类型执行相应的自动化操作

## 检测流程

```
截图 (50-200ms)
  ↓
分页器识别页面类型 (20-50ms, GPU加速)
  ↓
返回页面状态 + 置信度
  ↓
(可选) YOLO检测页面元素
```

## 使用示例

### 1. 基本页面检测

```python
# 在任何自动化模块中
result = await self.detector.detect_page(device_id)

print(f"页面类型: {result.state.name}")
print(f"置信度: {result.confidence:.2%}")
print(f"检测时间: {result.detection_time:.3f}秒")
```

### 2. 带元素检测

```python
# 检测页面并识别元素
result = await self.detector.detect_page(device_id, detect_elements=True)

print(f"页面类型: {result.state.name}")
print(f"检测到 {len(result.elements)} 个元素")

for element in result.elements:
    print(f"  - {element.class_name} at {element.center}")
```

### 3. 查找并点击元素

```python
# 查找签到按钮
element = await self.detector.get_element(device_id, "每日签到按钮")

if element:
    # 点击按钮
    await self.detector.click_element(device_id, "每日签到按钮")
```

### 4. 使用YOLO查找按钮

```python
# 使用YOLO模型查找按钮
button_pos = await self.detector.find_button_yolo(
    device_id,
    page_type="checkin",  # 签到页
    button_name="签到按钮",
    conf_threshold=0.5
)

if button_pos:
    x, y = button_pos
    await self.adb.tap(device_id, x, y)
```

## 支持的页面类型

分页器可以识别23种页面类型：

| 页面类型 | PageState | 准确率 |
|---------|-----------|--------|
| 首页 | HOME | 100% |
| 登录页 | LOGIN | 100% |
| 个人页_已登录 | PROFILE_LOGGED | 100% |
| 个人页_未登录 | PROFILE | 100% |
| 签到页 | CHECKIN | 70% |
| 签到弹窗 | CHECKIN_POPUP | 80% |
| 转账页 | TRANSFER | 100% |
| 钱包页 | WALLET | 100% |
| 交易流水 | TRANSACTION_HISTORY | 100% |
| 温馨提示 | WARMTIP | 90% |
| ... | ... | ... |

完整列表见 `PAGE_CLASSIFIER_INTEGRATION_SUMMARY.md`

## 性能特点

### 速度

- **页面分类**: 20-50ms (GPU)
- **截图**: 50-200ms
- **总检测时间**: 100-300ms

### 准确率

- **总体准确率**: 97.39%
- **完美识别**: 20个类别（100%）
- **良好识别**: 3个类别（70-90%）

### 优化

- ✅ GPU加速（CUDA）
- ✅ 检测缓存（0.5秒TTL）
- ✅ 自动降级（GPU不可用时使用CPU）
- ✅ 模型预加载（启动时加载）

## 配置文件

### 模型配置 (`model_config.json`)

```json
{
  "models": {
    "page_detector_integrated": {
      "enabled": true,
      "model_path": "page_classifier_pytorch_best.pth",
      "classes_path": "page_classes.json",
      "yolo_registry_path": "yolo_model_registry.json",
      "mapping_path": "page_yolo_mapping.json",
      "device": "cuda",
      "cache_ttl": 0.5
    }
  }
}
```

### 类别列表 (`page_classes.json`)

```json
[
  "个人页_已登录",
  "个人页_未登录",
  "个人页广告",
  "交易流水",
  "优惠劵",
  ...
]
```

## 常见问题

### Q: 如何禁用分页器？

A: 在 `model_config.json` 中设置：

```json
{
  "models": {
    "page_detector_integrated": {
      "enabled": false
    }
  }
}
```

### Q: 如何切换到CPU模式？

A: 在 `model_config.json` 中设置：

```json
{
  "models": {
    "page_detector_integrated": {
      "device": "cpu"
    }
  }
}
```

### Q: 如何调整缓存时间？

A: 在 `model_config.json` 中设置：

```json
{
  "models": {
    "page_detector_integrated": {
      "cache_ttl": 1.0  # 缓存1秒
    }
  }
}
```

### Q: 检测速度慢怎么办？

A: 检查以下几点：

1. 确认GPU加速已启用（查看日志中的"设备: cuda"）
2. 检查GPU驱动是否正常
3. 减少缓存TTL（更频繁使用缓存）
4. 关闭元素检测（`detect_elements=False`）

### Q: 识别准确率低怎么办？

A: 可能的原因：

1. 页面类型不在训练数据中 → 添加新类别并重新训练
2. 页面变化较大 → 收集新数据并重新训练
3. 截图质量差 → 检查分辨率和清晰度

## 调试技巧

### 1. 查看检测详情

```python
result = await self.detector.detect_page(device_id)
print(f"详情: {result.details}")
# 输出: 页面分类: 首页 (置信度: 94.78%), 检测到 0 个元素
```

### 2. 查看检测时间

```python
result = await self.detector.detect_page(device_id)
print(f"检测时间: {result.detection_time:.3f}秒")

# 如果超过0.5秒，会自动输出性能警告
# [性能警告] detect_page耗时0.523秒 (截图:0.123秒, 分类:0.045秒)
```

### 3. 清除缓存

```python
# 清除指定设备的缓存
self.detector.clear_cache(device_id)

# 清除所有缓存
self.detector.clear_cache()
```

### 4. 查看模型信息

```python
from src.model_manager import ModelManager

manager = ModelManager.get_instance()
info = manager.get_model_info('page_detector_integrated')

print(f"模型路径: {info['model_path']}")
print(f"加载时间: {info['load_time']:.2f}秒")
print(f"设备: {info['device']}")
print(f"状态: {info['status']}")
```

## 测试脚本

### 测试分页器

```bash
python test_page_classifier_samples_visual.py
```

### 测试整合检测器

```bash
python test_integrated_page_classifier.py
```

### 测试主程序

```bash
python run.py
```

## 总结

✅ 分页器已完全集成，无需额外配置
✅ 自动使用GPU加速，快速准确
✅ 支持23种页面类型识别
✅ 与现有系统完全兼容
✅ 立即可用

---

**更新时间**: 2026-01-30
**版本**: v1.0
**状态**: 生产就绪
