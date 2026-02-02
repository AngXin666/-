# 测试脚本模板使用说明

## 概述

`test_template.py` 是一个通用的测试脚本模板，包含了自动设备检测、模型初始化、截图、页面检测、OCR识别等常用功能，方便快速创建新的测试脚本。

## 功能特性

✅ **自动设备检测** - 自动获取正在运行的模拟器设备
✅ **模型初始化** - 自动加载整合检测器和OCR线程池
✅ **截图功能** - 快速截图并可选保存到文件
✅ **页面检测** - 使用整合检测器检测页面类型和元素
✅ **OCR识别** - 全屏OCR识别文本
✅ **完整示例** - 包含所有常用功能的示例代码

## 使用方法

### 1. 快速开始

```bash
# 直接运行模板查看示例
python test_template.py
```

### 2. 创建新测试脚本

```bash
# 复制模板
copy test_template.py test_my_feature.py

# 编辑 test_my_feature.py，在"自定义测试逻辑"部分添加你的代码
```

### 3. 修改测试逻辑

在 `main()` 函数的"自定义测试逻辑"部分添加你的测试代码：

```python
# ==================== 自定义测试逻辑 ====================
print("\n" + "=" * 60)
print("我的功能测试")
print("=" * 60)

# 示例1：测试ProfileReader
from src.profile_reader import ProfileReader
profile_reader = ProfileReader(adb, yolo_detector=integrated_detector)
result = await profile_reader.get_full_profile(device_id)
print(f"昵称: {result.get('nickname')}")
print(f"余额: {result.get('balance')}")

# 示例2：性能测试
import time
for i in range(10):
    start = time.time()
    # 你的测试代码
    elapsed = time.time() - start
    print(f"第{i+1}次: {elapsed:.3f}秒")

# 示例3：点击操作
await adb.tap(device_id, 270, 480)  # 点击屏幕中心
await asyncio.sleep(1)

print("✓ 测试完成")
```

## 模板包含的功能

### 1. 设备初始化

```python
# 自动获取设备
adb_path = r"D:\Program Files\Netease\MuMu\nx_device\12.0\shell\adb.exe"
adb = ADBBridge(adb_path)
device_id = devices[0]  # 自动获取第一个设备
```

### 2. 模型初始化

```python
# 初始化ModelManager
model_manager = ModelManager.get_instance()
model_manager.initialize_all_models(adb)

# 获取常用模型
integrated_detector = model_manager.get_page_detector_integrated()
ocr_pool = model_manager.get_ocr_thread_pool()
```

### 3. 截图功能

```python
# 截图
screenshot_data = await adb.screencap(device_id)

# 保存截图（可选）
from PIL import Image
from io import BytesIO
image = Image.open(BytesIO(screenshot_data))
image.save("test_screenshot.png")
```

### 4. 页面检测

```python
# 检测页面类型和元素
page_result = await integrated_detector.detect_page(
    device_id, 
    use_cache=False, 
    detect_elements=True
)

print(f"页面类型: {page_result.state.chinese_name}")
print(f"置信度: {page_result.confidence:.2%}")
print(f"元素数量: {len(page_result.elements)}")
```

### 5. OCR识别

```python
# 全屏OCR识别
from src.ocr_image_processor import enhance_for_ocr

image = Image.open(BytesIO(screenshot_data))
enhanced_image = enhance_for_ocr(image)
ocr_result = await ocr_pool.recognize(enhanced_image)

print(f"识别到 {len(ocr_result.texts)} 个文本")
print(f"文本列表: {ocr_result.texts}")
```

## 常见测试场景

### 场景1：功能测试

```python
# 测试某个功能是否正常工作
from src.profile_reader import ProfileReader

profile_reader = ProfileReader(adb, yolo_detector=integrated_detector)
result = await profile_reader.get_balance(device_id)

if result is not None:
    print(f"✓ 余额获取成功: {result:.2f} 元")
else:
    print("❌ 余额获取失败")
```

### 场景2：性能测试

```python
# 测试某个功能的性能（10次测试）
import time

times = []
for i in range(10):
    start = time.time()
    result = await profile_reader.get_balance(device_id)
    elapsed = time.time() - start
    times.append(elapsed)
    print(f"第{i+1}次: {elapsed:.3f}秒")

avg_time = sum(times) / len(times)
print(f"\n平均耗时: {avg_time:.3f}秒")
print(f"最快: {min(times):.3f}秒")
print(f"最慢: {max(times):.3f}秒")
```

### 场景3：准确性测试

```python
# 测试识别准确性
expected_balance = 18.36
success_count = 0

for i in range(10):
    result = await profile_reader.get_balance(device_id)
    if result == expected_balance:
        success_count += 1

accuracy = (success_count / 10) * 100
print(f"准确率: {accuracy}%")
```

### 场景4：弹窗处理测试

```python
# 测试弹窗检测和关闭
page_result = await integrated_detector.detect_page(device_id, use_cache=False)

if page_result.state.name in ['POPUP', 'PROFILE_AD']:
    print("检测到弹窗，尝试关闭...")
    # 弹窗处理逻辑已集成在 get_full_profile 中
    result = await profile_reader.get_full_profile(device_id)
    print("弹窗处理完成")
```

## 可用的ADB操作

```python
# 截图
screenshot_data = await adb.screencap(device_id)

# 点击
await adb.tap(device_id, x, y)

# 滑动
await adb.swipe(device_id, x1, y1, x2, y2, duration=500)

# 按返回键
await adb.press_back(device_id)

# 按Home键
await adb.press_home(device_id)

# 输入文本
await adb.input_text(device_id, "文本内容")
```

## 注意事项

1. **设备路径**：确保 `adb_path` 指向正确的ADB路径
2. **模型加载**：首次运行会加载模型，需要2-3秒
3. **异步函数**：所有测试代码必须在 `async def main()` 中
4. **错误处理**：建议添加 try-except 处理异常
5. **资源清理**：测试完成后模型会自动释放

## 示例测试脚本

### 示例1：余额获取测试

```python
# test_balance_get.py
async def main():
    # ... 初始化代码（使用模板） ...
    
    from src.profile_reader import ProfileReader
    profile_reader = ProfileReader(adb, yolo_detector=integrated_detector)
    
    print("=" * 60)
    print("余额获取测试（10次）")
    print("=" * 60)
    
    for i in range(10):
        balance = await profile_reader.get_balance(device_id)
        print(f"第{i+1}次: {balance:.2f} 元" if balance else f"第{i+1}次: 失败")
```

### 示例2：页面导航测试

```python
# test_navigation.py
async def main():
    # ... 初始化代码（使用模板） ...
    
    print("=" * 60)
    print("页面导航测试")
    print("=" * 60)
    
    # 点击个人页按钮
    await adb.tap(device_id, 450, 900)
    await asyncio.sleep(1)
    
    # 检测页面
    page_result = await integrated_detector.detect_page(device_id)
    print(f"当前页面: {page_result.state.chinese_name}")
```

## 测试结果示例

```
正在获取设备列表...
✓ 找到设备: 127.0.0.1:16448

正在初始化模型...
✓ 模型初始化完成

============================================================
截图测试
============================================================
✓ 截图成功，大小: 93581 字节

============================================================
页面检测测试
============================================================
页面类型: 个人页（已登录）
置信度: 99.95%
检测到 9 个元素

============================================================
OCR识别测试
============================================================
✓ 识别到 36 个文本
前10个文本: ['1:11', 'pOMqQs', '普通会员', 'ID:371419', ...]

============================================================
自定义测试
============================================================
✓ 测试完成
```

## 常见问题

### Q: 如何修改ADB路径？
A: 修改 `adb_path` 变量为你的ADB路径

### Q: 如何只加载部分模型？
A: 使用 `model_manager.get_xxx()` 只获取需要的模型

### Q: 如何保存测试结果？
A: 将结果写入文件或使用 logging 模块

### Q: 如何测试多个设备？
A: 遍历 `devices` 列表，对每个设备执行测试

## 相关文档

- [模型管理器文档](MODEL_CONFIG_README.md)
- [整合检测器使用指南](整合检测器使用说明.md)
- [OCR优化说明](docs/page_detector_cache_usage.md)

## 更新日志

- 2026-02-02: 创建测试脚本模板
  - 自动设备检测
  - 模型初始化
  - 截图、页面检测、OCR示例
  - 完整的使用说明
