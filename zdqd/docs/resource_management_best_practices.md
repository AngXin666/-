# 资源管理最佳实践

## 概述

本文档提供了Python资源管理的最佳实践指南，帮助开发者避免资源泄漏、提高程序的健壮性和性能。

## 核心原则

### 1. 资源必须被释放

**原则**：所有获取的资源（文件、数据库连接、网络连接、锁等）都必须在使用完毕后释放。

**为什么**：
- 避免资源泄漏
- 防止资源耗尽
- 提高系统稳定性
- 避免死锁和阻塞

### 2. 使用上下文管理器

**原则**：优先使用`with`语句管理资源，确保资源总是被正确释放。

```python
# ❌ 错误：手动管理资源
def read_file(filename):
    f = open(filename)
    try:
        data = f.read()
        return data
    finally:
        f.close()  # 容易忘记

# ✅ 正确：使用上下文管理器
def read_file(filename):
    with open(filename) as f:
        data = f.read()
        return data
    # 文件自动关闭，即使发生异常
```

### 3. 异常情况下也要释放资源

**原则**：无论是正常退出还是异常退出，资源都必须被释放。

```python
# ❌ 错误：异常时资源未释放
def process_data():
    conn = database.connect()
    data = conn.query("SELECT * FROM users")
    process(data)
    conn.close()  # 如果process()抛出异常，连接不会关闭

# ✅ 正确：使用try-finally
def process_data():
    conn = database.connect()
    try:
        data = conn.query("SELECT * FROM users")
        process(data)
    finally:
        conn.close()  # 总是会执行

# ✅ 更好：使用上下文管理器
def process_data():
    with database.connect() as conn:
        data = conn.query("SELECT * FROM users")
        process(data)
    # 连接自动关闭
```

## 常见资源类型

### 1. 文件资源

#### 基本文件操作

```python
# ✅ 读取文件
def read_file(filename):
    """读取文件内容"""
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()

# ✅ 写入文件
def write_file(filename, content):
    """写入文件内容"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)

# ✅ 追加文件
def append_file(filename, content):
    """追加文件内容"""
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(content)

# ✅ 二进制文件
def read_binary_file(filename):
    """读取二进制文件"""
    with open(filename, 'rb') as f:
        return f.read()
```

#### 处理大文件

```python
# ✅ 逐行读取大文件
def process_large_file(filename):
    """逐行处理大文件，避免内存溢出"""
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:  # 逐行读取，不会一次性加载整个文件
            process_line(line.strip())

# ✅ 使用缓冲区读取
def read_in_chunks(filename, chunk_size=8192):
    """分块读取文件"""
    with open(filename, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk
```

#### 临时文件

```python
import tempfile
import os

# ✅ 使用临时文件
def process_with_temp_file():
    """使用临时文件"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp:
        temp.write("临时数据")
        temp_path = temp.name
    
    try:
        # 使用临时文件
        process_file(temp_path)
    finally:
        # 清理临时文件
        os.unlink(temp_path)

# ✅ 更好：使用临时目录
def process_with_temp_dir():
    """使用临时目录"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # 在临时目录中工作
        temp_file = os.path.join(temp_dir, 'data.txt')
        with open(temp_file, 'w') as f:
            f.write("数据")
        process_file(temp_file)
    # 临时目录及其内容自动删除
```

### 2. 数据库连接

#### 基本数据库操作

```python
import sqlite3
from contextlib import contextmanager

# ✅ 使用上下文管理器
def query_database(db_path, query):
    """查询数据库"""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        return cursor.fetchall()
    # 连接自动关闭

# ✅ 事务管理
def update_database(db_path, updates):
    """更新数据库（事务）"""
    with sqlite3.connect(db_path) as conn:
        try:
            cursor = conn.cursor()
            for update in updates:
                cursor.execute(update)
            conn.commit()  # 提交事务
        except Exception as e:
            conn.rollback()  # 回滚事务
            raise
    # 连接自动关闭
```

#### 数据库连接池

```python
import sqlite3
import threading
from queue import Queue, Empty

class DatabaseConnectionPool:
    """数据库连接池"""
    
    def __init__(self, db_path, max_connections=5):
        self.db_path = db_path
        self.max_connections = max_connections
        self._pool = Queue(maxsize=max_connections)
        self._lock = threading.Lock()
        self._created_connections = 0
    
    def _create_connection(self):
        """创建新连接"""
        return sqlite3.connect(self.db_path, check_same_thread=False)
    
    @contextmanager
    def get_connection(self):
        """获取连接（上下文管理器）"""
        conn = None
        try:
            # 尝试从池中获取连接
            conn = self._pool.get_nowait()
        except Empty:
            # 池为空，创建新连接
            with self._lock:
                if self._created_connections < self.max_connections:
                    conn = self._create_connection()
                    self._created_connections += 1
                else:
                    # 等待可用连接
                    conn = self._pool.get()
        
        try:
            yield conn
        finally:
            # 归还连接到池
            self._pool.put(conn)
    
    def close_all(self):
        """关闭所有连接"""
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except Empty:
                break

# 使用示例
pool = DatabaseConnectionPool('data.db', max_connections=5)

def query_with_pool(query):
    """使用连接池查询"""
    with pool.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        return cursor.fetchall()
```

### 3. 网络连接

#### HTTP请求

```python
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# ✅ 使用Session管理连接
def fetch_data_with_session(urls):
    """使用Session复用连接"""
    with requests.Session() as session:
        # 配置重试策略
        retry = Retry(total=3, backoff_factor=0.3)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        results = []
        for url in urls:
            response = session.get(url, timeout=10)
            response.raise_for_status()
            results.append(response.json())
        
        return results
    # Session自动关闭

# ✅ 单次请求
def fetch_single_url(url):
    """单次HTTP请求"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    finally:
        # requests会自动管理连接
        pass
```

#### Socket连接

```python
import socket
from contextlib import closing

# ✅ 使用closing确保socket关闭
def send_data(host, port, data):
    """发送数据到socket"""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.connect((host, port))
        sock.sendall(data.encode())
        response = sock.recv(1024)
        return response.decode()
    # socket自动关闭
```

### 4. 线程和进程

#### 线程管理

```python
import threading
from concurrent.futures import ThreadPoolExecutor

# ✅ 使用ThreadPoolExecutor
def process_items_parallel(items):
    """并行处理项目"""
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_item, item) for item in items]
        results = [future.result() for future in futures]
        return results
    # 线程池自动清理

# ❌ 错误：手动管理线程
def bad_parallel_processing(items):
    threads = []
    for item in items:
        t = threading.Thread(target=process_item, args=(item,))
        t.start()
        threads.append(t)
    
    for t in threads:
        t.join()  # 容易忘记
```

#### 进程管理

```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# ✅ 使用ProcessPoolExecutor
def process_items_multiprocess(items):
    """多进程处理项目"""
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_item, items))
        return results
    # 进程池自动清理
```

### 5. 锁和同步原语

#### 锁管理

```python
import threading

# ✅ 使用上下文管理器管理锁
class ThreadSafeCounter:
    """线程安全的计数器"""
    
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()
    
    def increment(self):
        """增加计数"""
        with self._lock:  # 锁自动释放
            self._value += 1
    
    def get_value(self):
        """获取值"""
        with self._lock:
            return self._value

# ❌ 错误：手动管理锁
def bad_increment(self):
    self._lock.acquire()
    try:
        self._value += 1
    finally:
        self._lock.release()  # 容易忘记
```

#### 信号量

```python
import threading

class ResourcePool:
    """资源池（使用信号量限制并发）"""
    
    def __init__(self, max_resources=5):
        self._semaphore = threading.Semaphore(max_resources)
        self._resources = []
    
    def acquire_resource(self):
        """获取资源"""
        with self._semaphore:  # 信号量自动释放
            # 获取资源
            resource = self._get_or_create_resource()
            try:
                yield resource
            finally:
                # 归还资源
                self._return_resource(resource)
```

## 创建自定义上下文管理器

### 方法1：使用类

```python
class ManagedResource:
    """自定义资源管理器（类方式）"""
    
    def __init__(self, resource_name):
        self.resource_name = resource_name
        self.resource = None
    
    def __enter__(self):
        """进入上下文"""
        print(f"获取资源: {self.resource_name}")
        self.resource = self._acquire_resource()
        return self.resource
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文"""
        print(f"释放资源: {self.resource_name}")
        if self.resource:
            self._release_resource(self.resource)
        
        # 返回False表示不抑制异常
        return False
    
    def _acquire_resource(self):
        """获取资源"""
        return f"Resource: {self.resource_name}"
    
    def _release_resource(self, resource):
        """释放资源"""
        pass

# 使用示例
with ManagedResource("database") as resource:
    print(f"使用资源: {resource}")
# 资源自动释放
```

### 方法2：使用装饰器

```python
from contextlib import contextmanager

@contextmanager
def managed_resource(resource_name):
    """自定义资源管理器（装饰器方式）"""
    # 获取资源
    print(f"获取资源: {resource_name}")
    resource = f"Resource: {resource_name}"
    
    try:
        # 返回资源给with语句
        yield resource
    finally:
        # 释放资源
        print(f"释放资源: {resource_name}")

# 使用示例
with managed_resource("database") as resource:
    print(f"使用资源: {resource}")
# 资源自动释放
```

### 实际示例：数据库事务管理器

```python
import sqlite3
from contextlib import contextmanager

@contextmanager
def database_transaction(db_path):
    """数据库事务管理器"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # 返回cursor给with语句
        yield cursor
        # 如果没有异常，提交事务
        conn.commit()
        print("事务提交成功")
    except Exception as e:
        # 如果有异常，回滚事务
        conn.rollback()
        print(f"事务回滚: {e}")
        raise
    finally:
        # 总是关闭连接
        cursor.close()
        conn.close()
        print("数据库连接已关闭")

# 使用示例
with database_transaction('data.db') as cursor:
    cursor.execute("INSERT INTO users (name) VALUES ('Alice')")
    cursor.execute("INSERT INTO users (name) VALUES ('Bob')")
# 事务自动提交，连接自动关闭
```

### 实际示例：文件锁管理器

```python
import fcntl
from contextlib import contextmanager

@contextmanager
def file_lock(filename):
    """文件锁管理器"""
    f = open(filename, 'a')
    try:
        # 获取文件锁
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        print(f"获取文件锁: {filename}")
        yield f
    finally:
        # 释放文件锁
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        f.close()
        print(f"释放文件锁: {filename}")

# 使用示例
with file_lock('data.txt') as f:
    f.write("写入数据\n")
# 锁自动释放，文件自动关闭
```

## 资源池模式

### 通用资源池

```python
import threading
from queue import Queue, Empty, Full
from contextlib import contextmanager

class ResourcePool:
    """通用资源池"""
    
    def __init__(self, factory, max_size=10, timeout=30):
        """
        Args:
            factory: 创建资源的工厂函数
            max_size: 池的最大大小
            timeout: 获取资源的超时时间
        """
        self._factory = factory
        self._max_size = max_size
        self._timeout = timeout
        self._pool = Queue(maxsize=max_size)
        self._size = 0
        self._lock = threading.Lock()
    
    def _create_resource(self):
        """创建新资源"""
        with self._lock:
            if self._size < self._max_size:
                resource = self._factory()
                self._size += 1
                return resource
        return None
    
    @contextmanager
    def acquire(self):
        """获取资源"""
        resource = None
        
        try:
            # 尝试从池中获取
            resource = self._pool.get(timeout=self._timeout)
        except Empty:
            # 池为空，创建新资源
            resource = self._create_resource()
            if resource is None:
                raise TimeoutError("无法获取资源")
        
        try:
            yield resource
        finally:
            # 归还资源到池
            try:
                self._pool.put(resource, block=False)
            except Full:
                # 池已满，关闭资源
                self._close_resource(resource)
                with self._lock:
                    self._size -= 1
    
    def _close_resource(self, resource):
        """关闭资源"""
        if hasattr(resource, 'close'):
            resource.close()
    
    def close_all(self):
        """关闭所有资源"""
        while not self._pool.empty():
            try:
                resource = self._pool.get_nowait()
                self._close_resource(resource)
            except Empty:
                break
        self._size = 0

# 使用示例：数据库连接池
def create_db_connection():
    return sqlite3.connect('data.db')

db_pool = ResourcePool(create_db_connection, max_size=5)

def query_database(query):
    with db_pool.acquire() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        return cursor.fetchall()
```

## 内存管理

### 大对象处理

```python
import sys

# ✅ 及时删除不需要的大对象
def process_large_data():
    """处理大数据"""
    # 加载大数据
    large_data = load_large_dataset()
    
    # 处理数据
    result = process(large_data)
    
    # 及时删除，释放内存
    del large_data
    
    return result

# ✅ 使用生成器避免内存占用
def read_large_file_generator(filename):
    """使用生成器读取大文件"""
    with open(filename, 'r') as f:
        for line in f:
            yield line.strip()
    # 不会一次性加载整个文件到内存

# 使用生成器
for line in read_large_file_generator('large_file.txt'):
    process_line(line)
```

### 弱引用

```python
import weakref

class CacheWithWeakRef:
    """使用弱引用的缓存"""
    
    def __init__(self):
        self._cache = weakref.WeakValueDictionary()
    
    def get(self, key):
        """获取缓存值"""
        return self._cache.get(key)
    
    def set(self, key, value):
        """设置缓存值"""
        self._cache[key] = value
    
    # 当对象不再被引用时，会自动从缓存中移除
```

## 资源泄漏检测

### 检测文件句柄泄漏

```python
import psutil
import os

def check_open_files():
    """检查当前进程打开的文件"""
    process = psutil.Process(os.getpid())
    open_files = process.open_files()
    
    print(f"当前打开的文件数: {len(open_files)}")
    for f in open_files:
        print(f"  {f.path}")
    
    return len(open_files)

# 使用示例
before = check_open_files()
# 执行操作
process_files()
after = check_open_files()

if after > before:
    print(f"警告：可能存在文件句柄泄漏！增加了 {after - before} 个文件句柄")
```

### 检测内存泄漏

```python
import tracemalloc

def check_memory_usage():
    """检查内存使用"""
    # 开始追踪
    tracemalloc.start()
    
    # 执行操作
    process_data()
    
    # 获取内存快照
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    
    print("内存使用前10名:")
    for stat in top_stats[:10]:
        print(stat)
    
    tracemalloc.stop()
```

## 常见陷阱

### 陷阱1：在循环中打开文件

```python
# ❌ 错误：在循环中打开文件但不关闭
def bad_process_files(filenames):
    for filename in filenames:
        f = open(filename)  # 文件句柄泄漏
        data = f.read()
        process(data)
        # 忘记关闭文件

# ✅ 正确：使用上下文管理器
def good_process_files(filenames):
    for filename in filenames:
        with open(filename) as f:
            data = f.read()
            process(data)
        # 文件自动关闭
```

### 陷阱2：在异常处理中忘记释放资源

```python
# ❌ 错误：异常时资源未释放
def bad_database_operation():
    conn = database.connect()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")
    results = cursor.fetchall()
    conn.close()  # 如果execute或fetchall抛出异常，连接不会关闭

# ✅ 正确：使用try-finally或上下文管理器
def good_database_operation():
    with database.connect() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users")
        results = cursor.fetchall()
    # 连接自动关闭，即使发生异常
```

### 陷阱3：共享资源的并发访问

```python
# ❌ 错误：多线程共享文件句柄
shared_file = open('log.txt', 'a')

def bad_log_message(message):
    shared_file.write(message + '\n')  # 线程不安全

# ✅ 正确：每次打开和关闭文件
def good_log_message(message):
    with open('log.txt', 'a') as f:
        f.write(message + '\n')

# ✅ 或者：使用锁保护
import threading

file_lock = threading.Lock()

def thread_safe_log_message(message):
    with file_lock:
        with open('log.txt', 'a') as f:
            f.write(message + '\n')
```

### 陷阱4：忘记关闭连接池

```python
# ❌ 错误：程序退出时不关闭连接池
pool = DatabaseConnectionPool('data.db')

def main():
    # 使用连接池
    query_database("SELECT * FROM users")
    # 程序退出，连接池未关闭

# ✅ 正确：使用atexit或显式关闭
import atexit

pool = DatabaseConnectionPool('data.db')
atexit.register(pool.close_all)  # 程序退出时自动关闭

# ✅ 或者：使用上下文管理器
class DatabaseConnectionPool:
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_all()
        return False

with DatabaseConnectionPool('data.db') as pool:
    query_database("SELECT * FROM users")
# 连接池自动关闭
```

## 性能优化

### 1. 复用资源

```python
# ❌ 性能较差：每次都创建新连接
def bad_multiple_queries(queries):
    results = []
    for query in queries:
        conn = database.connect()  # 每次都创建新连接
        result = conn.execute(query)
        results.append(result)
        conn.close()
    return results

# ✅ 性能更好：复用连接
def good_multiple_queries(queries):
    results = []
    with database.connect() as conn:  # 复用连接
        for query in queries:
            result = conn.execute(query)
            results.append(result)
    return results
```

### 2. 使用连接池

```python
# ✅ 使用连接池避免频繁创建/销毁连接
pool = DatabaseConnectionPool('data.db', max_connections=5)

def query_with_pool(query):
    with pool.get_connection() as conn:
        return conn.execute(query)
```

### 3. 批量操作

```python
# ❌ 性能较差：逐个插入
def bad_batch_insert(items):
    with database.connect() as conn:
        for item in items:
            conn.execute(f"INSERT INTO table VALUES ('{item}')")

# ✅ 性能更好：批量插入
def good_batch_insert(items):
    with database.connect() as conn:
        conn.executemany(
            "INSERT INTO table VALUES (?)",
            [(item,) for item in items]
        )
```

## 测试资源管理

### 测试资源释放

```python
import pytest

def test_file_is_closed():
    """测试文件是否正确关闭"""
    filename = 'test.txt'
    
    # 写入文件
    with open(filename, 'w') as f:
        f.write("test")
    
    # 验证文件已关闭
    # 如果文件未关闭，下面的操作可能失败
    with open(filename, 'r') as f:
        content = f.read()
    
    assert content == "test"

def test_connection_pool_cleanup():
    """测试连接池清理"""
    pool = DatabaseConnectionPool('test.db', max_connections=2)
    
    # 使用连接
    with pool.get_connection() as conn:
        conn.execute("SELECT 1")
    
    # 关闭连接池
    pool.close_all()
    
    # 验证所有连接已关闭
    assert pool._size == 0
```

## 总结

### 关键要点

1. **使用上下文管理器**：确保资源总是被正确释放
2. **异常安全**：无论是否发生异常，资源都要释放
3. **及时释放**：不再使用的资源应立即释放
4. **使用资源池**：复用资源，避免频繁创建/销毁
5. **检测泄漏**：定期检查资源使用情况
6. **线程安全**：多线程环境下保护共享资源
7. **测试资源管理**：编写测试验证资源正确释放

### 推荐工具

- `contextlib`：上下文管理器工具
- `psutil`：进程和系统监控
- `tracemalloc`：内存追踪
- `weakref`：弱引用
- `atexit`：程序退出时的清理

### 进一步阅读

- Python官方文档：[上下文管理器](https://docs.python.org/3/library/contextlib.html)
- Python官方文档：[with语句](https://docs.python.org/3/reference/compound_stmts.html#with)
- 《Effective Python》第5章：类和接口
- 《Python最佳实践指南》：资源管理章节
