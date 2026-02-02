# 统一日志配置使用指南

## 概述

`logging_config.py` 模块提供了统一的日志配置功能，确保所有模块的日志输出保持一致的格式和行为。

## 主要特性

1. **统一的日志格式**：所有模块使用相同的日志格式
2. **灵活的配置**：支持不同的日志级别和输出方式
3. **单例模式**：避免重复配置同一个logger
4. **文件和控制台双输出**：同时支持文件记录和控制台显示
5. **第三方库日志管理**：自动降低第三方库的日志输出级别

## 快速开始

### 1. 在模块中使用

在任何模块的顶部，使用以下方式设置logger：

```python
from logging_config import setup_logger

# 使用模块名称作为logger名称
logger = setup_logger(__name__)

# 使用logger
logger.info("这是一条信息日志")
logger.error("这是一条错误日志")
logger.debug("这是一条调试日志")
```

### 2. 在应用入口初始化

在应用的主入口（如 `run.py`），初始化日志系统：

```python
from logging_config import init_logging

# 初始化日志系统
init_logging(
    log_dir="logs",           # 日志目录
    level="INFO",             # 全局日志级别
    configure_third_party=True  # 配置第三方库日志
)
```

## 详细使用

### 设置不同的日志级别

```python
from logging_config import setup_logger

# 设置为DEBUG级别
logger = setup_logger(__name__, level="DEBUG")

# 设置为WARNING级别
logger = setup_logger(__name__, level="WARNING")
```

### 只输出到控制台（不写文件）

```python
from logging_config import setup_logger

logger = setup_logger(__name__, log_to_file=False)
```

### 只输出到文件（不显示在控制台）

```python
from logging_config import setup_logger

logger = setup_logger(__name__, log_to_console=False)
```

### 自定义日志文件前缀

```python
from logging_config import setup_logger

logger = setup_logger(__name__, file_prefix="my_app")
# 日志文件名：my_app_20260201.log
```

### 临时修改日志级别

使用 `LogContext` 上下文管理器临时修改日志级别：

```python
from logging_config import setup_logger, LogContext

logger = setup_logger(__name__, level="INFO")

# 正常情况下，DEBUG日志不会输出
logger.debug("这条不会显示")

# 在特定代码块中启用DEBUG级别
with LogContext(logger, "DEBUG"):
    logger.debug("这条会显示")
    # 执行需要详细日志的代码

# 退出上下文后，恢复原来的级别
logger.debug("这条又不会显示了")
```

### 配置第三方库日志

降低第三方库的日志输出，避免干扰应用日志：

```python
from logging_config import configure_third_party_loggers

# 将第三方库日志级别设置为WARNING
configure_third_party_loggers(level="WARNING")

# 或设置为ERROR，进一步减少输出
configure_third_party_loggers(level="ERROR")
```

### 快速获取logger

如果不需要特殊配置，可以使用快速方式：

```python
from logging_config import quick_logger

logger = quick_logger("my_module")
logger.info("快速日志")
```

## 日志格式

统一的日志格式为：

```
时间 - 模块名 - 级别 - 消息
```

示例：

```
2026-02-01 20:32:29 - my_module - INFO - 应用启动
2026-02-01 20:32:30 - my_module - ERROR - 发生错误
```

## 日志级别

从低到高的日志级别：

1. **DEBUG**：详细的调试信息
2. **INFO**：一般信息
3. **WARNING**：警告信息
4. **ERROR**：错误信息
5. **CRITICAL**：严重错误

## 最佳实践

### 1. 模块级别的logger

在每个模块的顶部创建logger：

```python
from logging_config import setup_logger

logger = setup_logger(__name__)

class MyClass:
    def my_method(self):
        logger.info("执行方法")
```

### 2. 使用合适的日志级别

- **DEBUG**：用于开发和调试，记录详细的执行流程
- **INFO**：记录重要的业务事件（如用户登录、任务完成）
- **WARNING**：记录可能的问题（如配置缺失、使用默认值）
- **ERROR**：记录错误（如操作失败、异常捕获）
- **CRITICAL**：记录严重错误（如系统崩溃、数据损坏）

### 3. 记录异常信息

使用 `exc_info=True` 记录完整的异常堆栈：

```python
try:
    risky_operation()
except Exception as e:
    logger.error(f"操作失败: {e}", exc_info=True)
```

### 4. 避免日志泄露敏感信息

不要在日志中记录密码、密钥等敏感信息：

```python
# ❌ 错误：记录了密码
logger.info(f"用户登录: {username}, 密码: {password}")

# ✅ 正确：不记录密码
logger.info(f"用户登录: {username}")
```

### 5. 使用结构化日志

使用清晰的日志消息格式：

```python
# ✅ 好的日志
logger.info(f"账号处理完成: {phone}, 状态: {status}, 耗时: {duration}秒")

# ❌ 不好的日志
logger.info(f"{phone} {status} {duration}")
```

## 迁移指南

### 从旧的日志方式迁移

**旧方式：**

```python
import logging

logger = logging.getLogger(__name__)
```

**新方式：**

```python
from logging_config import setup_logger

logger = setup_logger(__name__)
```

### 从 logging.basicConfig 迁移

**旧方式：**

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
```

**新方式：**

```python
from logging_config import init_logging, setup_logger

# 在应用入口
init_logging(log_dir="logs", level="INFO")

# 在模块中
logger = setup_logger(__name__)
```

## 常见问题

### Q: 为什么我的日志没有输出？

A: 检查以下几点：
1. 日志级别是否正确（如设置为INFO，则DEBUG日志不会输出）
2. 是否调用了 `init_logging()` 初始化日志系统
3. 是否正确导入了 `setup_logger`

### Q: 如何查看日志文件？

A: 日志文件默认保存在 `logs/` 目录下，文件名格式为 `app_YYYYMMDD.log`。

### Q: 如何修改全局日志级别？

A: 使用 `set_log_level()` 函数：

```python
from logging_config import set_log_level

set_log_level("DEBUG")  # 修改为DEBUG级别
```

### Q: 如何禁用某个模块的日志？

A: 获取该模块的logger并设置级别为CRITICAL：

```python
import logging

logging.getLogger("noisy_module").setLevel(logging.CRITICAL)
```

## 示例代码

### 完整示例

```python
# main.py
from logging_config import init_logging, setup_logger

# 初始化日志系统
init_logging(log_dir="logs", level="INFO")

# 创建logger
logger = setup_logger(__name__)

def main():
    logger.info("应用启动")
    
    try:
        # 执行业务逻辑
        process_data()
        logger.info("数据处理完成")
    except Exception as e:
        logger.error(f"数据处理失败: {e}", exc_info=True)
    
    logger.info("应用退出")

if __name__ == '__main__':
    main()
```

```python
# my_module.py
from logging_config import setup_logger

logger = setup_logger(__name__)

def process_data():
    logger.debug("开始处理数据")
    
    # 处理逻辑
    data = load_data()
    logger.info(f"加载了 {len(data)} 条数据")
    
    result = transform_data(data)
    logger.info(f"转换完成，结果: {result}")
    
    logger.debug("数据处理完成")
    return result
```

## 总结

使用统一的日志配置可以：

1. ✅ 保持所有模块的日志格式一致
2. ✅ 简化日志配置代码
3. ✅ 方便统一管理日志级别
4. ✅ 提高代码可维护性
5. ✅ 便于问题排查和调试

建议所有新模块都使用 `logging_config` 模块来配置日志。
