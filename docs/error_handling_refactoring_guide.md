# 统一错误处理重构指南

## 概述

本指南说明如何使用新的统一错误处理和资源管理模块重构现有代码。

## 新模块介绍

### 1. error_handling.py

提供统一的错误处理装饰器和工具：

- `@handle_errors`: 统一错误处理装饰器
- `@retry`: 重试装饰器
- `@with_fallback`: 降级处理装饰器
- `safe_execute`: 安全执行函数
- `ErrorContext`: 错误上下文管理器

### 2. resource_manager.py

提供资源管理上下文管理器：

- `DatabaseConnectionPool`: 数据库连接池
- `managed_file`: 文件操作上下文管理器
- `ManagedLock`: 锁管理上下文管理器
- `managed_resource`: 通用资源管理器
- `ResourceTracker`: 资源跟踪器

## 重构示例

### 示例 1: 数据库操作重构

**重构前：**

```python
def save_record(self, record):
    try:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO ...", (...))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"保存失败: {e}")
        return False
```

**重构后：**

```python
from error_handling import handle_errors, retry
from resource_manager import DatabaseConnectionPool

# 初始化连接池
self._pool = DatabaseConnectionPool(self.db_path, max_connections=5)

@retry(max_attempts=3, delay=0.1, backoff=2.0, logger=logger)
@handle_errors(logger, "保存记录失败", reraise=False, default_return=False)
def save_record(self, record):
    with self._pool.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO ...", (...))
        conn.commit()
        return True
```

**改进点：**
- 使用连接池避免频繁创建连接
- 自动重试临时性错误（如数据库锁定）
- 统一的错误日志格式
- 自动资源清理

### 示例 2: 文件操作重构

**重构前：**

```python
def read_config(self, file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"文件不存在: {file_path}")
        return {}
    except Exception as e:
        print(f"读取失败: {e}")
        return {}
```

**重构后：**

```python
from error_handling import handle_errors
from resource_manager import managed_file

@handle_errors(logger, "读取配置失败", reraise=False, default_return={})
def read_config(self, file_path):
    with managed_file(file_path, 'r', logger=logger) as f:
        return json.load(f)
```

**改进点：**
- 自动处理文件关闭
- 统一的错误处理
- 自动创建目录（如果需要）

### 示例 3: 线程锁重构

**重构前：**

```python
def update_data(self, data):
    self._lock.acquire()
    try:
        # 更新数据
        self._data.update(data)
    finally:
        self._lock.release()
```

**重构后：**

```python
from resource_manager import ManagedLock

def update_data(self, data):
    with ManagedLock(self._lock, timeout=5.0, logger=logger, name="更新数据"):
        # 更新数据
        self._data.update(data)
```

**改进点：**
- 自动获取和释放锁
- 支持超时机制
- 详细的日志记录
- 避免死锁

### 示例 4: 降级处理

**重构前：**

```python
def get_data(self):
    try:
        return self._get_from_api()
    except Exception as e:
        print(f"API失败: {e}")
        try:
            return self._get_from_cache()
        except Exception as e2:
            print(f"缓存也失败: {e2}")
            return None
```

**重构后：**

```python
from error_handling import with_fallback

@with_fallback(self._get_from_cache, logger, "API失败，使用缓存")
def get_data(self):
    return self._get_from_api()
```

**改进点：**
- 自动降级处理
- 清晰的降级逻辑
- 统一的日志记录

### 示例 5: 错误上下文管理

**重构前：**

```python
def process_batch(self, items):
    for item in items:
        try:
            self.process_item(item)
        except Exception as e:
            print(f"处理失败: {item}, 错误: {e}")
            continue
```

**重构后：**

```python
from error_handling import ErrorContext, ErrorSeverity

def process_batch(self, items):
    for item in items:
        with ErrorContext(logger, f"处理项目失败: {item}", 
                         reraise=False, severity=ErrorSeverity.MEDIUM):
            self.process_item(item)
```

**改进点：**
- 统一的错误处理
- 错误严重程度分级
- 自动日志记录
- 继续处理其他项目

## 重构步骤

### 步骤 1: 识别需要重构的代码

查找以下模式：
- 重复的 try-except 块
- 手动管理资源（连接、文件、锁）
- 简单的错误日志记录
- 缺少重试机制的临时性错误

### 步骤 2: 选择合适的工具

| 场景 | 推荐工具 |
|------|---------|
| 数据库操作 | DatabaseConnectionPool + @retry + @handle_errors |
| 文件操作 | managed_file + @handle_errors |
| 线程锁 | ManagedLock |
| 需要重试的操作 | @retry |
| 需要降级的操作 | @with_fallback |
| 批量处理 | ErrorContext |

### 步骤 3: 逐步重构

1. 先重构最关键的模块（如数据库操作）
2. 保持向后兼容
3. 添加测试验证
4. 逐步推广到其他模块

### 步骤 4: 测试验证

- 单元测试：验证正常情况
- 错误测试：验证异常处理
- 并发测试：验证线程安全
- 性能测试：确保性能不降低

## 最佳实践

### 1. 错误严重程度分级

```python
from error_handling import ErrorSeverity

# 低：可以忽略的错误
@handle_errors(logger, "缓存更新失败", severity=ErrorSeverity.LOW)

# 中：需要记录但不影响主流程
@handle_errors(logger, "统计数据失败", severity=ErrorSeverity.MEDIUM)

# 高：影响功能但可以降级
@handle_errors(logger, "主数据源失败", severity=ErrorSeverity.HIGH)

# 严重：必须处理的错误
@handle_errors(logger, "数据库初始化失败", severity=ErrorSeverity.CRITICAL, reraise=True)
```

### 2. 合理使用重试

```python
# 适合重试的场景：网络请求、数据库锁定
@retry(max_attempts=3, delay=1.0, backoff=2.0, 
       exceptions=(requests.RequestException, sqlite3.OperationalError))

# 不适合重试的场景：逻辑错误、数据验证失败
# 这些情况不应该使用 @retry
```

### 3. 资源管理

```python
# 使用连接池
pool = DatabaseConnectionPool(db_path, max_connections=5)

# 使用上下文管理器
with pool.get_connection() as conn:
    # 使用连接
    pass
# 连接自动释放

# 程序退出时关闭连接池
pool.close_all()
```

### 4. 日志记录

```python
# 配置统一的日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 为每个模块创建logger
logger = logging.getLogger(__name__)

# 在装饰器中使用logger
@handle_errors(logger, "操作失败")
```

## 注意事项

### 1. 向后兼容

重构时保持API接口不变，确保不破坏现有代码。

### 2. 性能考虑

- 连接池可以提高性能
- 过度的重试可能降低性能
- 日志记录应该适度

### 3. 错误处理策略

- 不要捕获所有异常（避免隐藏bug）
- 只捕获预期的异常
- 关键错误应该重新抛出

### 4. 测试覆盖

- 测试正常情况
- 测试异常情况
- 测试资源释放
- 测试并发场景

## 迁移计划

### 阶段 1: 核心模块（已完成）

- ✅ 创建 error_handling.py
- ✅ 创建 resource_manager.py
- ✅ 创建示例重构（local_db_refactored.py）

### 阶段 2: 关键模块重构（进行中）

- [ ] 重构 local_db.py
- [ ] 重构 daily_checkin.py
- [ ] 重构 balance_transfer.py

### 阶段 3: 其他模块重构

- [ ] 重构 navigator.py
- [ ] 重构 auto_login.py
- [ ] 重构其他模块

### 阶段 4: 测试和验证

- [ ] 单元测试
- [ ] 集成测试
- [ ] 性能测试
- [ ] 稳定性测试

## 总结

使用统一的错误处理和资源管理模块可以：

1. **提高代码质量**：统一的错误处理模式
2. **减少代码重复**：复用装饰器和上下文管理器
3. **提高可维护性**：清晰的错误处理逻辑
4. **提高稳定性**：自动重试和资源清理
5. **提高可观测性**：统一的日志格式

建议逐步重构，先从最关键的模块开始，确保每一步都经过充分测试。
