# 超时设置审查总结

## 审查日期
2026-02-07

## 审查目标
确保所有超时设置统一为 15 秒

## 审查结果

### ✅ 已修改为 15 秒的配置

1. **`TimeoutsConfig.PAGE_LOAD_TIMEOUT`**
   - 修改前：10.0 秒
   - 修改后：15.0 秒
   - 位置：`zdqd/src/timeouts_config.py` 第 19 行

2. **`TimeoutsConfig.OCR_TIMEOUT_LONG`**
   - 修改前：10.0 秒
   - 修改后：15.0 秒
   - 位置：`zdqd/src/timeouts_config.py` 第 38 行

3. **`TimeoutsConfig.HTTP_REQUEST_TIMEOUT`**
   - 修改前：10.0 秒
   - 修改后：15.0 秒
   - 位置：`zdqd/src/timeouts_config.py` 第 44 行

4. **`TimeoutsConfig.SMART_WAIT_TIMEOUT`**
   - 修改前：10.0 秒
   - 修改后：15.0 秒
   - 位置：`zdqd/src/timeouts_config.py` 第 54 行

5. **`ximeng_automation.py` 中的 `asyncio.wait_for`**
   - 修改前：`timeout=10.0`
   - 修改后：`timeout=15.0`
   - 位置：`zdqd/src/ximeng_automation.py` 第 1464 行

6. **`reset_to_defaults()` 方法中的默认值**
   - 修改前：多个 10.0 秒的默认值
   - 修改后：统一为 15.0 秒
   - 位置：`zdqd/src/timeouts_config.py` 第 206-228 行

### ✅ 已经是 15 秒的配置（无需修改）

1. **`TimeoutsConfig.NAVIGATION_TIMEOUT = 15.0`** - 导航超时
2. **`TimeoutsConfig.CHECKIN_TIMEOUT = 15.0`** - 签到超时
3. **智能等待器 `max_wait = 15.0`** - 页面变化等待超时（`smart_waiter.py`）

### ⚠️ 保持原值的配置（有特殊用途）

以下配置保持原值，因为它们有特殊用途：

1. **`TimeoutsConfig.PAGE_TRANSITION_TIMEOUT = 5.0`** - 页面切换超时（快速切换）
2. **`TimeoutsConfig.CHECKIN_PAGE_LOAD = 3.0`** - 签到页面加载等待
3. **`TimeoutsConfig.CHECKIN_POPUP_WAIT = 2.0`** - 签到弹窗等待
4. **`TimeoutsConfig.CHECKIN_BUTTON_WAIT = 0.5`** - 签到按钮点击后等待
5. **`TimeoutsConfig.TRANSFER_TIMEOUT = 20.0`** - 转账超时（需要更长时间）
6. **`TimeoutsConfig.TRANSFER_PAGE_LOAD = 2.0`** - 转账页面加载等待
7. **`TimeoutsConfig.TRANSFER_INPUT_WAIT = 1.0`** - 转账输入后等待
8. **`TimeoutsConfig.TRANSFER_CONFIRM_WAIT = 2.0`** - 转账确认等待
9. **`TimeoutsConfig.OCR_TIMEOUT = 5.0`** - OCR识别超时（常规）
10. **`TimeoutsConfig.OCR_TIMEOUT_SHORT = 2.0`** - OCR短超时（快速识别）
11. **`TimeoutsConfig.PAGE_DETECT_TIMEOUT = 5.0`** - 页面检测超时
12. **`TimeoutsConfig.ELEMENT_DETECT_TIMEOUT = 3.0`** - 元素检测超时
13. **`TimeoutsConfig.HTTP_REQUEST_SHORT = 5.0`** - HTTP短超时
14. **`TimeoutsConfig.WAIT_SHORT = 0.5`** - 短等待
15. **`TimeoutsConfig.WAIT_MEDIUM = 1.0`** - 中等待
16. **`TimeoutsConfig.WAIT_LONG = 2.0`** - 长等待
17. **`TimeoutsConfig.WAIT_EXTRA_LONG = 3.0`** - 超长等待
18. **`TimeoutsConfig.SMART_WAIT_INTERVAL = 0.5`** - 智能等待器检测间隔
19. **`TimeoutsConfig.CACHE_TTL_SHORT = 0.5`** - 短缓存时间
20. **`TimeoutsConfig.CACHE_TTL_MEDIUM = 1.0`** - 中等缓存时间
21. **`TimeoutsConfig.CACHE_TTL_LONG = 3.0`** - 长缓存时间

## 超时配置使用指南

### 主要超时（15 秒）
用于需要等待较长时间的操作：
- 导航到页面
- 页面加载
- 签到流程
- OCR 长时间识别
- HTTP 请求
- 智能等待器

### 转账超时（20 秒）
转账操作需要更长时间，保持 20 秒

### 短超时（2-5 秒）
用于快速操作：
- 页面切换
- 元素检测
- OCR 快速识别

### 等待时间（0.5-3 秒）
用于固定等待：
- 按钮点击后等待
- 页面动画等待
- 输入后等待

## 验证方法

运行以下命令验证配置：

```python
from src.timeouts_config import TimeoutsConfig

# 打印所有配置
TimeoutsConfig.print_config()
```

## 注意事项

1. **不要直接修改超时值**：使用 `TimeoutsConfig` 类统一管理
2. **避免硬编码超时**：使用 `TimeoutsConfig` 中的常量
3. **特殊情况**：如果需要不同的超时，使用 `TimeoutsConfig.set_timeout()` 方法

## 修改历史

- 2026-02-07：统一主要超时为 15 秒
  - PAGE_LOAD_TIMEOUT: 10.0 → 15.0
  - OCR_TIMEOUT_LONG: 10.0 → 15.0
  - HTTP_REQUEST_TIMEOUT: 10.0 → 15.0
  - SMART_WAIT_TIMEOUT: 10.0 → 15.0
  - ximeng_automation.py asyncio.wait_for: 10.0 → 15.0
