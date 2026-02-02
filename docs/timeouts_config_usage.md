# 超时配置模块使用指南

## 概述

`TimeoutsConfig` 模块提供了统一的超时时间配置管理，避免在代码中硬编码超时值，提高代码的可维护性和可配置性。

## 功能特性

- ✅ 统一管理所有超时配置
- ✅ 支持从JSON文件加载配置
- ✅ 支持运行时动态修改配置
- ✅ 支持保存配置到文件
- ✅ 支持重置到默认值
- ✅ 按类别组织配置（导航、签到、转账、OCR等）

## 配置项说明

### 导航相关
- `NAVIGATION_TIMEOUT`: 导航超时（默认30秒）
- `PAGE_LOAD_TIMEOUT`: 页面加载超时（默认10秒）
- `PAGE_TRANSITION_TIMEOUT`: 页面切换超时（默认5秒）

### 签到相关
- `CHECKIN_TIMEOUT`: 签到超时（默认15秒）
- `CHECKIN_PAGE_LOAD`: 签到页面加载等待（默认3秒）
- `CHECKIN_POPUP_WAIT`: 签到弹窗等待（默认2秒）
- `CHECKIN_BUTTON_WAIT`: 签到按钮点击后等待（默认0.5秒）

### 转账相关
- `TRANSFER_TIMEOUT`: 转账超时（默认20秒）
- `TRANSFER_PAGE_LOAD`: 转账页面加载等待（默认2秒）
- `TRANSFER_INPUT_WAIT`: 转账输入后等待（默认1秒）
- `TRANSFER_CONFIRM_WAIT`: 转账确认等待（默认2秒）

### OCR识别
- `OCR_TIMEOUT`: OCR识别超时（默认5秒）
- `OCR_TIMEOUT_SHORT`: OCR短超时（默认2秒）
- `OCR_TIMEOUT_LONG`: OCR长超时（默认10秒）

### 页面检测
- `PAGE_DETECT_TIMEOUT`: 页面检测超时（默认5秒）
- `ELEMENT_DETECT_TIMEOUT`: 元素检测超时（默认3秒）

### 网络请求
- `HTTP_REQUEST_TIMEOUT`: HTTP请求超时（默认10秒）
- `HTTP_REQUEST_SHORT`: HTTP短超时（默认5秒）

### 等待时间
- `WAIT_SHORT`: 短等待（默认0.5秒）
- `WAIT_MEDIUM`: 中等待（默认1秒）
- `WAIT_LONG`: 长等待（默认2秒）
- `WAIT_EXTRA_LONG`: 超长等待（默认3秒）

### 智能等待器
- `SMART_WAIT_TIMEOUT`: 智能等待器默认超时（默认10秒）
- `SMART_WAIT_INTERVAL`: 智能等待器检测间隔（默认0.5秒）

### 缓存时间
- `CACHE_TTL_SHORT`: 短缓存时间（默认0.5秒）
- `CACHE_TTL_MEDIUM`: 中等缓存时间（默认1秒）
- `CACHE_TTL_LONG`: 长缓存时间（默认3秒）

## 使用方法

### 1. 在代码中使用超时配置

```python
from src.timeouts_config import TimeoutsConfig
import asyncio

# 使用配置的超时时间
await asyncio.sleep(TimeoutsConfig.WAIT_SHORT)  # 0.5秒
await asyncio.sleep(TimeoutsConfig.WAIT_MEDIUM)  # 1秒
await asyncio.sleep(TimeoutsConfig.CHECKIN_PAGE_LOAD)  # 3秒

# 使用配置的超时参数
ocr_result = await ocr_pool.recognize(image, timeout=TimeoutsConfig.OCR_TIMEOUT)
```

### 2. 使用便捷函数

```python
from src.timeouts_config import get_timeout

# 获取超时配置
navigation_timeout = get_timeout("NAVIGATION_TIMEOUT")  # 30.0

# 获取不存在的配置（使用默认值）
custom_timeout = get_timeout("CUSTOM_TIMEOUT", default=15.0)  # 15.0
```

### 3. 从配置文件加载

创建配置文件 `config/timeouts.json`:

```json
{
  "NAVIGATION_TIMEOUT": 45.0,
  "CHECKIN_TIMEOUT": 20.0,
  "OCR_TIMEOUT": 8.0
}
```

在代码中加载：

```python
from src.timeouts_config import TimeoutsConfig

# 加载配置文件
TimeoutsConfig.load_from_file("config/timeouts.json")

# 现在配置已更新
print(TimeoutsConfig.NAVIGATION_TIMEOUT)  # 45.0
```

### 4. 运行时修改配置

```python
from src.timeouts_config import TimeoutsConfig

# 修改单个配置
TimeoutsConfig.set_timeout("NAVIGATION_TIMEOUT", 60.0)

# 验证修改
print(TimeoutsConfig.NAVIGATION_TIMEOUT)  # 60.0
```

### 5. 保存配置到文件

```python
from src.timeouts_config import TimeoutsConfig

# 修改一些配置
TimeoutsConfig.set_timeout("NAVIGATION_TIMEOUT", 45.0)
TimeoutsConfig.set_timeout("CHECKIN_TIMEOUT", 20.0)

# 保存到文件
TimeoutsConfig.save_to_file("config/timeouts.json")
```

### 6. 重置到默认值

```python
from src.timeouts_config import TimeoutsConfig

# 重置所有配置到默认值
TimeoutsConfig.reset_to_defaults()

# 验证已重置
print(TimeoutsConfig.NAVIGATION_TIMEOUT)  # 30.0
```

### 7. 打印当前配置

```python
from src.timeouts_config import TimeoutsConfig

# 打印所有配置
TimeoutsConfig.print_config()
```

输出示例：
```
============================================================
超时配置 - 当前设置
============================================================

导航相关:
  NAVIGATION_TIMEOUT             =  30.00 秒
  PAGE_LOAD_TIMEOUT              =  10.00 秒
  PAGE_TRANSITION_TIMEOUT        =   5.00 秒

签到相关:
  CHECKIN_TIMEOUT                =  15.00 秒
  CHECKIN_PAGE_LOAD              =   3.00 秒
  CHECKIN_POPUP_WAIT             =   2.00 秒
  CHECKIN_BUTTON_WAIT            =   0.50 秒

...
============================================================
```

## 自动初始化

模块会在导入时自动尝试加载以下位置的配置文件（按优先级）：

1. `config/timeouts.json`
2. `config/timeouts_config.json`
3. `.kiro/timeouts.json`

如果找到配置文件，会自动加载并应用配置。

## 最佳实践

### 1. 使用配置常量而不是硬编码

❌ **不推荐**：
```python
await asyncio.sleep(0.5)  # 硬编码
await asyncio.sleep(1.0)  # 硬编码
```

✅ **推荐**：
```python
await asyncio.sleep(TimeoutsConfig.WAIT_SHORT)
await asyncio.sleep(TimeoutsConfig.WAIT_MEDIUM)
```

### 2. 为不同场景使用合适的超时

```python
# 快速操作使用短超时
await asyncio.sleep(TimeoutsConfig.WAIT_SHORT)  # 0.5秒

# 页面加载使用中等超时
await asyncio.sleep(TimeoutsConfig.PAGE_LOAD_TIMEOUT)  # 10秒

# 复杂操作使用长超时
await asyncio.sleep(TimeoutsConfig.NAVIGATION_TIMEOUT)  # 30秒
```

### 3. 在测试环境中调整超时

```python
# 测试环境中可以缩短超时时间以加快测试
if is_test_environment():
    TimeoutsConfig.set_timeout("NAVIGATION_TIMEOUT", 5.0)
    TimeoutsConfig.set_timeout("PAGE_LOAD_TIMEOUT", 2.0)
```

### 4. 为生产环境创建配置文件

创建 `config/timeouts.production.json`:

```json
{
  "NAVIGATION_TIMEOUT": 45.0,
  "PAGE_LOAD_TIMEOUT": 15.0,
  "CHECKIN_TIMEOUT": 20.0,
  "TRANSFER_TIMEOUT": 30.0
}
```

在生产环境启动时加载：

```python
if is_production():
    TimeoutsConfig.load_from_file("config/timeouts.production.json")
```

## 配置文件示例

完整的配置文件示例请参考 `config/timeouts.json.example`。

## 注意事项

1. **配置值必须为正数**：超时时间必须大于0
2. **配置名称区分大小写**：必须使用正确的大写名称
3. **配置文件格式**：必须是有效的JSON格式
4. **线程安全**：配置修改不是线程安全的，建议在程序启动时加载配置

## 故障排除

### 问题：配置文件加载失败

**原因**：配置文件不存在或格式错误

**解决方案**：
1. 检查文件路径是否正确
2. 验证JSON格式是否有效
3. 查看控制台错误信息

### 问题：配置修改不生效

**原因**：配置名称错误或值无效

**解决方案**：
1. 检查配置名称是否正确（区分大小写）
2. 确保配置值为正数
3. 使用 `print_config()` 验证当前配置

## 相关文档

- [代码质量改进设计文档](../.kiro/specs/code-quality-improvement/design.md)
- [冗余报告](../.kiro/specs/code-quality-improvement/redundancy_report.md)
