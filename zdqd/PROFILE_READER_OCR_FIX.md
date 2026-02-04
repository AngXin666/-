# ProfileReader OCR 属性错误修复 + 余额获取优化

## 问题描述

用户报告在运行自动化程序时遇到以下问题：

### 问题1: OCR 属性错误
```
'ProfileReader' object has no attribute '_ocr'
```

该错误发生在尝试获取余额、积分、优惠券等信息时，导致 fallback 方法失败。

### 问题2: 余额获取失败
```
[03:47:17] [实例3]   ⚠️ 未能获取最终余额
```

即使修复了OCR属性错误，余额仍然获取失败。原因是使用了固定坐标，在不同分辨率或界面布局下失效。

## 根本原因

在 `src/profile_reader.py` 中，有多个 fallback 方法使用了旧的 `self._ocr` 属性，但该属性已被重构为 `self._ocr_pool`。

### 问题代码位置

以下方法中存在错误的 `self._ocr` 调用：

1. **get_balance_fallback** (约 2377 行)
2. **get_user_id_fallback** (约 2455 行)
3. **get_nickname_fallback** (约 2516 行)
4. **get_phone_fallback** (约 2584 行)
5. **get_points_fallback** (约 2643 行)
6. **get_vouchers_fallback** (约 2721 行)

### 错误模式

```python
# 错误的代码
ocr_result = await asyncio.wait_for(
    asyncio.to_thread(self._ocr, image),  # ❌ self._ocr 不存在
    timeout=10.0
)

# 并且使用了错误的属性名
if not ocr_result or not ocr_result.txts:  # ❌ 应该是 .texts
    return None

texts = list(ocr_result.txts)  # ❌ 应该是 .texts
```

## 修复方案

### 修复1: 替换 OCR 调用

将所有 `self._ocr` 调用替换为 `self._ocr_pool.recognize()`：

```python
# 修复后的代码
ocr_result = await asyncio.wait_for(
    self._ocr_pool.recognize(image, timeout=10.0),  # ✓ 使用 _ocr_pool
    timeout=10.0
)

# 并且使用正确的属性名
if not ocr_result or not ocr_result.texts:  # ✓ 使用 .texts
    return None

texts = list(ocr_result.texts)  # ✓ 使用 .texts
```

### 修复2: 改进余额获取方法（不使用固定坐标）

**旧方法的问题**:
- 使用固定像素坐标 `REGIONS = {'balance': (30, 230, 150, 330), ...}`
- 只适用于特定分辨率（540x960）
- 界面布局变化时失效

**新方法**:
1. **全屏OCR识别** - 识别整个屏幕的所有文本
2. **关键字定位** - 查找"余额"、"积分"、"抵扣券"等关键字
3. **位置关系提取** - 在关键字右侧或下方提取数字

```python
async def _recognize_regions(self, device_id: str, full_image: 'Image.Image'):
    """使用全屏OCR + 关键字定位识别余额、积分、抵扣券、优惠券"""
    
    # 1. 全屏OCR识别
    enhanced_image = enhance_for_ocr(full_image)
    ocr_result = await self._ocr_pool.recognize(enhanced_image, timeout=5.0)
    
    texts = ocr_result.texts
    boxes = ocr_result.boxes  # 文本位置信息
    
    # 2. 使用位置信息提取数值
    result = self._extract_values_with_positions(texts, boxes)
    
    return result

def _extract_values_with_positions(self, texts, boxes):
    """使用位置信息提取数值
    
    策略：
    1. 找到"余额"关键字的位置
    2. 在关键字右侧（x > keyword_x, |y - keyword_y| < 50）查找数字
    3. 或在关键字下方（y > keyword_y, |x - keyword_x| < 100）查找数字
    """
    # 查找"余额"关键字
    keyword_pos = find_keyword_position("余额", texts, boxes)
    
    # 在关键字附近查找数字
    for pos in text_positions:
        # 检查是否在右侧或下方
        if is_near_keyword(pos, keyword_pos):
            value = extract_number(pos['text'])
            return value
```

**优势**:
- ✅ 不依赖固定坐标，适应不同分辨率
- ✅ 适应界面布局变化
- ✅ 更鲁棒，成功率更高

### 修复的方法列表

#### OCR 属性修复（6个方法）

- ✅ `get_balance_fallback()` - 余额获取备选方案
- ✅ `get_user_id_fallback()` - 用户ID获取备选方案
- ✅ `get_nickname_fallback()` - 昵称获取备选方案
- ✅ `get_phone_fallback()` - 手机号获取备选方案
- ✅ `get_points_fallback()` - 积分获取备选方案
- ✅ `get_vouchers_fallback()` - 抵扣券获取备选方案

#### 余额获取优化（3个方法）

- ✅ `_recognize_regions()` - 改为全屏OCR + 关键字定位
- ✅ `_extract_values_with_positions()` - 新增：使用位置信息提取数值
- ✅ `_extract_values_from_texts()` - 新增：无位置信息时的降级方案

## 验证

### 代码验证

运行以下命令验证没有遗留的 `self._ocr` 调用：

```bash
cd zdqd
python -c "import re; content = open('src/profile_reader.py', 'r', encoding='utf-8').read(); matches = re.findall(r'self\._ocr\(', content); print(f'找到 {len(matches)} 个 self._ocr 调用')"
```

预期输出：`找到 0 个 self._ocr 调用`

### 功能测试

#### 测试1: OCR 属性修复
```bash
cd zdqd
python test_profile_reader_ocr_fix.py
```

#### 测试2: 余额获取优化
```bash
cd zdqd
python test_balance_ocr_fix.py
```

需要：
- 连接设备
- 导航到个人页
- 观察日志输出，确认使用全屏OCR + 关键字定位

### 实际设备测试

在实际设备上测试 fallback 方法：

1. 连接设备
2. 运行自动化程序
3. 观察日志，确认不再出现 `'ProfileReader' object has no attribute '_ocr'` 错误
4. 验证余额、积分、优惠券等信息能正常获取

## 影响范围

### 修复的文件

- `zdqd/src/profile_reader.py` - 修复了 6 个 fallback 方法

### 受益功能

- 余额获取（当 YOLO 检测失败时的备选方案）
- 用户ID获取（当 YOLO 检测失败时的备选方案）
- 昵称获取（当 YOLO 检测失败时的备选方案）
- 手机号获取（当 YOLO 检测失败时的备选方案）
- 积分获取（当 YOLO 检测失败时的备选方案）
- 抵扣券获取（当 YOLO 检测失败时的备选方案）

## 技术细节

### OCR Pool 架构

`ProfileReader` 使用 OCR 线程池来管理 OCR 识别：

```python
class ProfileReader:
    def __init__(self, adb_bridge, model_manager):
        # ...
        self._ocr_pool = OCRPool(max_workers=2)  # OCR 线程池
        # 注意：没有 self._ocr 属性
```

### 正确的 OCR 调用方式

```python
# 方式1: 带超时的异步调用
ocr_result = await self._ocr_pool.recognize(image, timeout=5.0)

# 方式2: 不带超时的异步调用
ocr_result = await self._ocr_pool.recognize(image)

# OCR 结果属性
if ocr_result:
    texts = ocr_result.texts  # 识别的文本列表
    boxes = ocr_result.boxes  # 文本框坐标
```

## 相关问题

### 登录缓存路径问题

用户同时报告了登录缓存路径错误，显示路径包含 `temp_databases_DCStorage-shm` 等临时文件名。

该问题在 `src/login_cache_manager.py` 中，路径构造逻辑正常，但可能是在错误处理或日志输出时显示了临时文件路径。需要进一步调查具体的错误场景。

## 总结

本次修复解决了两个关键问题：

### 1. OCR 属性错误
- 修复了所有 fallback 方法的 OCR 调用错误
- 确保当 YOLO 检测失败时，备选的 OCR 方案能够正常工作

### 2. 余额获取失败
- 移除了固定坐标依赖
- 改用全屏OCR + 关键字定位的智能方法
- 提高了不同分辨率和界面布局下的成功率

### 技术改进
- **更鲁棒**: 不依赖固定坐标，适应界面变化
- **更智能**: 使用位置关系提取数值，而非盲目裁剪
- **更可靠**: 多层降级策略（YOLO → 位置OCR → 文本OCR）

---

**修复日期**: 2026-02-04  
**修复人员**: Kiro AI Assistant  
**测试状态**: 代码验证通过，等待实际设备测试
