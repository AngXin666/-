# 错误处理最佳实践

## 概述

本文档提供了Python错误处理的最佳实践指南，帮助开发者编写健壮、可维护的错误处理代码。

## 核心原则

### 1. 明确区分错误类型

**原则**：不同类型的错误需要不同的处理策略。

**错误分类**：

1. **预期错误**：正常业务流程的一部分
   - 示例：用户输入验证失败、文件不存在
   - 处理：正常的控制流程，返回错误信息

2. **可恢复错误**：可以通过重试或降级处理
   - 示例：网络超时、临时资源不可用
   - 处理：重试、使用备用方案、降级服务

3. **不可恢复错误**：程序无法继续执行
   - 示例：配置错误、系统资源耗尽
   - 处理：记录详细日志、通知管理员、优雅退出

4. **编程错误**：代码bug导致的错误
   - 示例：类型错误、空指针、索引越界
   - 处理：修复代码，不应该在生产环境中捕获

```python
# 预期错误：正常处理
def validate_user_input(data):
    if not data:
        return False, "数据不能为空"
    return True, None

# 可恢复错误：重试
@retry(max_attempts=3)
def fetch_data_from_api():
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

# 不可恢复错误：记录并退出
def load_config():
    try:
        with open('config.yaml') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.critical("配置文件不存在，程序无法启动")
        sys.exit(1)

# 编程错误：不应该捕获
def calculate_average(numbers):
    # 不要捕获 ZeroDivisionError
    # 如果发生，说明代码有bug，应该修复
    return sum(numbers) / len(numbers)
```

### 2. 在正确的层级处理错误

**原则**：在最合适的层级捕获和处理异常。

**分层错误处理**：

```
┌─────────────────────────────────────┐
│  表示层（UI）                        │
│  - 用户友好的错误消息                │
│  - 错误提示和恢复建议                │
├─────────────────────────────────────┤
│  业务逻辑层                          │
│  - 业务规则验证                      │
│  - 业务异常处理                      │
├─────────────────────────────────────┤
│  数据访问层                          │
│  - 数据库错误处理                    │
│  - 数据验证                          │
├─────────────────────────────────────┤
│  基础设施层                          │
│  - 网络错误处理                      │
│  - 文件系统错误处理                  │
└─────────────────────────────────────┘
```

```python
# 基础设施层：捕获底层异常，转换为领域异常
class DatabaseError(Exception):
    """数据库错误"""
    pass

def execute_query(query):
    try:
        return db.execute(query)
    except sqlite3.Error as e:
        logger.error(f"数据库查询失败: {e}")
        raise DatabaseError(f"查询失败: {e}") from e

# 业务逻辑层：处理业务异常
def get_user_profile(user_id):
    try:
        return execute_query(f"SELECT * FROM users WHERE id = {user_id}")
    except DatabaseError:
        logger.warning(f"无法获取用户 {user_id} 的资料")
        return None  # 返回默认值

# 表示层：向用户显示友好消息
def display_user_profile(user_id):
    profile = get_user_profile(user_id)
    if profile is None:
        show_message("无法加载用户资料，请稍后重试")
    else:
        show_profile(profile)
```

### 3. 提供足够的上下文信息

**原则**：异常消息应该包含足够的信息来诊断问题。

```python
# ❌ 错误：信息不足
def process_file(filename):
    try:
        with open(filename) as f:
            return f.read()
    except Exception as e:
        logger.error("处理失败")  # 缺少上下文
        raise

# ✅ 正确：提供详细上下文
def process_file(filename):
    try:
        with open(filename) as f:
            return f.read()
    except FileNotFoundError as e:
        logger.error(
            f"文件不存在: {filename}",
            extra={
                'filename': filename,
                'error_type': 'FileNotFoundError',
                'cwd': os.getcwd()
            }
        )
        raise
    except PermissionError as e:
        logger.error(
            f"没有权限读取文件: {filename}",
            extra={
                'filename': filename,
                'error_type': 'PermissionError',
                'user': os.getenv('USER')
            }
        )
        raise
```

## 异常处理模式

### 模式1：使用装饰器统一错误处理

```python
import functools
import logging
from typing import Callable, Any

def handle_errors(
    logger: logging.Logger,
    error_message: str = "操作失败",
    reraise: bool = False,
    default_return: Any = None
):
    """统一的错误处理装饰器
    
    Args:
        logger: 日志记录器
        error_message: 错误消息前缀
        reraise: 是否重新抛出异常
        default_return: 发生错误时的默认返回值
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"{error_message}: {func.__name__}",
                    extra={
                        'function': func.__name__,
                        'args': args,
                        'kwargs': kwargs,
                        'error_type': type(e).__name__,
                        'error_message': str(e)
                    },
                    exc_info=True
                )
                if reraise:
                    raise
                return default_return
        return wrapper
    return decorator

# 使用示例
@handle_errors(logger, "数据库操作失败", reraise=True)
def save_to_database(data):
    db.insert(data)

@handle_errors(logger, "获取配置失败", default_return={})
def load_config():
    with open('config.json') as f:
        return json.load(f)
```

### 模式2：重试机制

```python
import time
import functools
from typing import Callable, Type, Tuple

def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """重试装饰器
    
    Args:
        max_attempts: 最大尝试次数
        delay: 初始延迟时间（秒）
        backoff: 延迟时间的倍增因子
        exceptions: 需要重试的异常类型
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        logger.error(
                            f"{func.__name__} 失败，已重试 {max_attempts} 次",
                            exc_info=True
                        )
                        raise
                    
                    logger.warning(
                        f"{func.__name__} 失败，{current_delay}秒后重试 "
                        f"(尝试 {attempt}/{max_attempts})"
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            raise last_exception
        return wrapper
    return decorator

# 使用示例
@retry(max_attempts=3, delay=1.0, backoff=2.0, exceptions=(requests.RequestException,))
def fetch_data_from_api(url):
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()
```

### 模式3：降级处理

```python
def with_fallback(primary_func: Callable, fallback_func: Callable):
    """带降级的函数执行
    
    Args:
        primary_func: 主要函数
        fallback_func: 降级函数
    """
    def wrapper(*args, **kwargs):
        try:
            return primary_func(*args, **kwargs)
        except Exception as e:
            logger.warning(
                f"主要方法失败，使用降级方案: {e}",
                exc_info=True
            )
            try:
                return fallback_func(*args, **kwargs)
            except Exception as fallback_error:
                logger.error(
                    f"降级方案也失败: {fallback_error}",
                    exc_info=True
                )
                raise
    return wrapper

# 使用示例
def get_data_from_cache(key):
    """从缓存获取数据"""
    return cache.get(key)

def get_data_from_database(key):
    """从数据库获取数据"""
    return db.query(f"SELECT * FROM data WHERE key = '{key}'")

# 创建带降级的函数
get_data = with_fallback(get_data_from_cache, get_data_from_database)
```

### 模式4：上下文管理器错误处理

```python
from contextlib import contextmanager

@contextmanager
def error_handling_context(operation_name: str, logger: logging.Logger):
    """错误处理上下文管理器
    
    Args:
        operation_name: 操作名称
        logger: 日志记录器
    """
    logger.info(f"开始 {operation_name}")
    try:
        yield
        logger.info(f"完成 {operation_name}")
    except Exception as e:
        logger.error(
            f"{operation_name} 失败: {e}",
            exc_info=True
        )
        raise
    finally:
        logger.debug(f"清理 {operation_name} 的资源")

# 使用示例
def process_data():
    with error_handling_context("数据处理", logger):
        # 处理逻辑
        data = load_data()
        result = transform_data(data)
        save_result(result)
```

## 自定义异常

### 创建异常层次结构

```python
# 基础异常类
class ApplicationError(Exception):
    """应用程序基础异常"""
    
    def __init__(self, message: str, error_code: str = None, **kwargs):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = kwargs
    
    def __str__(self):
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message

# 业务异常
class BusinessError(ApplicationError):
    """业务逻辑异常"""
    pass

class ValidationError(BusinessError):
    """数据验证异常"""
    pass

class AuthenticationError(BusinessError):
    """认证异常"""
    pass

# 技术异常
class TechnicalError(ApplicationError):
    """技术异常"""
    pass

class DatabaseError(TechnicalError):
    """数据库异常"""
    pass

class NetworkError(TechnicalError):
    """网络异常"""
    pass

# 使用示例
def validate_user_input(data):
    if not data.get('username'):
        raise ValidationError(
            "用户名不能为空",
            error_code="VALIDATION_001",
            field="username",
            value=data.get('username')
        )
    
    if len(data['username']) < 3:
        raise ValidationError(
            "用户名长度不能少于3个字符",
            error_code="VALIDATION_002",
            field="username",
            min_length=3,
            actual_length=len(data['username'])
        )
```

### 异常链

```python
# 保留原始异常信息
def process_user_data(user_id):
    try:
        raw_data = fetch_from_database(user_id)
    except sqlite3.Error as e:
        # 使用 from 保留异常链
        raise DatabaseError(
            f"无法获取用户 {user_id} 的数据",
            error_code="DB_001"
        ) from e
    
    try:
        return parse_user_data(raw_data)
    except ValueError as e:
        raise ValidationError(
            f"用户 {user_id} 的数据格式无效",
            error_code="VALIDATION_003"
        ) from e
```

## 日志记录最佳实践

### 1. 使用适当的日志级别

```python
import logging

logger = logging.getLogger(__name__)

# DEBUG：详细的调试信息
logger.debug(f"处理用户 {user_id}，参数: {params}")

# INFO：一般信息
logger.info(f"用户 {user_id} 登录成功")

# WARNING：警告信息，程序可以继续运行
logger.warning(f"用户 {user_id} 尝试访问受限资源")

# ERROR：错误信息，功能失败但程序可以继续
logger.error(f"无法保存用户 {user_id} 的数据", exc_info=True)

# CRITICAL：严重错误，程序可能无法继续
logger.critical("数据库连接失败，程序即将退出", exc_info=True)
```

### 2. 记录异常堆栈

```python
# ❌ 错误：只记录异常消息
try:
    risky_operation()
except Exception as e:
    logger.error(f"操作失败: {e}")  # 丢失堆栈信息

# ✅ 正确：记录完整堆栈
try:
    risky_operation()
except Exception as e:
    logger.error("操作失败", exc_info=True)  # 包含堆栈信息

# ✅ 或者使用 exception() 方法
try:
    risky_operation()
except Exception as e:
    logger.exception("操作失败")  # 自动包含堆栈信息
```

### 3. 添加结构化上下文

```python
# 使用 extra 参数添加结构化数据
try:
    process_order(order_id)
except Exception as e:
    logger.error(
        "订单处理失败",
        extra={
            'order_id': order_id,
            'user_id': user_id,
            'error_type': type(e).__name__,
            'timestamp': datetime.now().isoformat()
        },
        exc_info=True
    )
```

## 错误恢复策略

### 策略1：优雅降级

```python
class DataService:
    """数据服务，支持优雅降级"""
    
    def get_user_data(self, user_id):
        """获取用户数据，支持多级降级"""
        # 尝试从缓存获取
        try:
            data = self._get_from_cache(user_id)
            if data:
                logger.debug(f"从缓存获取用户 {user_id} 的数据")
                return data
        except Exception as e:
            logger.warning(f"缓存访问失败: {e}")
        
        # 尝试从主数据库获取
        try:
            data = self._get_from_primary_db(user_id)
            logger.info(f"从主数据库获取用户 {user_id} 的数据")
            return data
        except Exception as e:
            logger.error(f"主数据库访问失败: {e}")
        
        # 尝试从备份数据库获取
        try:
            data = self._get_from_backup_db(user_id)
            logger.warning(f"从备份数据库获取用户 {user_id} 的数据")
            return data
        except Exception as e:
            logger.error(f"备份数据库访问失败: {e}")
        
        # 所有方法都失败，返回默认值
        logger.critical(f"无法获取用户 {user_id} 的数据，使用默认值")
        return self._get_default_data()
```

### 策略2：断路器模式

```python
import time
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"      # 正常状态
    OPEN = "open"          # 断路状态
    HALF_OPEN = "half_open"  # 半开状态

class CircuitBreaker:
    """断路器"""
    
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    def call(self, func, *args, **kwargs):
        """通过断路器调用函数"""
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.timeout:
                self.state = CircuitState.HALF_OPEN
                logger.info("断路器进入半开状态")
            else:
                raise Exception("断路器开启，拒绝请求")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """成功时的处理"""
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            logger.info("断路器关闭")
    
    def _on_failure(self):
        """失败时的处理"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"断路器开启，失败次数: {self.failure_count}")

# 使用示例
breaker = CircuitBreaker(failure_threshold=5, timeout=60)

def call_external_api():
    return breaker.call(requests.get, "https://api.example.com/data")
```

### 策略3：补偿事务

```python
class Transaction:
    """支持补偿的事务"""
    
    def __init__(self):
        self.operations = []
        self.compensations = []
    
    def add_operation(self, operation, compensation):
        """添加操作和对应的补偿操作"""
        self.operations.append(operation)
        self.compensations.append(compensation)
    
    def execute(self):
        """执行事务"""
        executed = []
        
        try:
            for i, operation in enumerate(self.operations):
                logger.info(f"执行操作 {i+1}/{len(self.operations)}")
                operation()
                executed.append(i)
            
            logger.info("事务执行成功")
            return True
            
        except Exception as e:
            logger.error(f"事务执行失败: {e}，开始补偿")
            
            # 按相反顺序执行补偿操作
            for i in reversed(executed):
                try:
                    logger.info(f"执行补偿操作 {i+1}")
                    self.compensations[i]()
                except Exception as comp_error:
                    logger.error(f"补偿操作 {i+1} 失败: {comp_error}")
            
            raise

# 使用示例
def transfer_money(from_account, to_account, amount):
    transaction = Transaction()
    
    # 添加扣款操作和补偿
    transaction.add_operation(
        lambda: deduct_money(from_account, amount),
        lambda: add_money(from_account, amount)
    )
    
    # 添加加款操作和补偿
    transaction.add_operation(
        lambda: add_money(to_account, amount),
        lambda: deduct_money(to_account, amount)
    )
    
    # 添加记录操作和补偿
    transaction.add_operation(
        lambda: record_transaction(from_account, to_account, amount),
        lambda: delete_transaction_record(from_account, to_account, amount)
    )
    
    transaction.execute()
```

## 常见陷阱

### 陷阱1：捕获过于宽泛的异常

```python
# ❌ 错误：捕获所有异常
try:
    process_data()
except Exception:  # 太宽泛
    logger.error("处理失败")

# ✅ 正确：捕获特定异常
try:
    process_data()
except ValueError as e:
    logger.error(f"数据格式错误: {e}")
except IOError as e:
    logger.error(f"IO错误: {e}")
except Exception as e:
    logger.error(f"未预期的错误: {e}", exc_info=True)
    raise  # 重新抛出未知异常
```

### 陷阱2：吞掉异常

```python
# ❌ 错误：静默失败
try:
    important_operation()
except Exception:
    pass  # 吞掉异常，没有任何记录

# ✅ 正确：至少记录日志
try:
    important_operation()
except Exception as e:
    logger.error(f"操作失败: {e}", exc_info=True)
    # 根据情况决定是否重新抛出
```

### 陷阱3：在finally中抛出异常

```python
# ❌ 错误：finally中的异常会覆盖原始异常
try:
    risky_operation()
finally:
    cleanup()  # 如果cleanup抛出异常，原始异常会丢失

# ✅ 正确：保护finally块
try:
    risky_operation()
finally:
    try:
        cleanup()
    except Exception as e:
        logger.error(f"清理失败: {e}")
```

### 陷阱4：异常消息包含敏感信息

```python
# ❌ 错误：泄露敏感信息
def login(username, password):
    try:
        authenticate(username, password)
    except AuthError as e:
        logger.error(f"登录失败: {username}, {password}")  # 泄露密码！

# ✅ 正确：不记录敏感信息
def login(username, password):
    try:
        authenticate(username, password)
    except AuthError as e:
        logger.error(f"用户 {username} 登录失败")  # 不记录密码
```

## 测试错误处理

### 1. 测试异常抛出

```python
import pytest

def test_validation_error():
    """测试验证错误"""
    with pytest.raises(ValidationError) as exc_info:
        validate_user_input({'username': ''})
    
    assert "用户名不能为空" in str(exc_info.value)
    assert exc_info.value.error_code == "VALIDATION_001"
```

### 2. 测试错误恢复

```python
def test_fallback_mechanism():
    """测试降级机制"""
    # 模拟主要方法失败
    with patch('module.primary_method', side_effect=Exception("主要方法失败")):
        # 模拟降级方法成功
        with patch('module.fallback_method', return_value="降级结果"):
            result = get_data_with_fallback()
            assert result == "降级结果"
```

### 3. 测试重试机制

```python
def test_retry_mechanism():
    """测试重试机制"""
    call_count = 0
    
    def flaky_function():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception("临时失败")
        return "成功"
    
    result = retry(max_attempts=3)(flaky_function)()
    assert result == "成功"
    assert call_count == 3
```

## 总结

### 关键要点

1. **明确区分错误类型**：预期错误、可恢复错误、不可恢复错误
2. **在正确的层级处理**：不要在底层吞掉异常
3. **提供足够的上下文**：帮助快速诊断问题
4. **使用装饰器统一处理**：减少重复代码
5. **实现重试和降级**：提高系统健壮性
6. **记录详细日志**：包含堆栈和上下文信息
7. **测试错误场景**：确保错误处理正确工作

### 推荐工具

- `logging`模块：Python标准日志库
- `tenacity`：重试库
- `pytest`：测试框架
- `sentry`：错误追踪服务

### 进一步阅读

- Python官方文档：[异常处理](https://docs.python.org/3/tutorial/errors.html)
- 《Effective Python》第4章：异常
- 《Python最佳实践指南》：错误处理章节
