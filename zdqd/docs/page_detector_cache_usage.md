# 页面检测缓存管理器使用指南

## 概述

`PageDetectorCache` 是一个统一的页面检测缓存管理器，提供以下功能：

1. **缓存失效策略**：支持TTL（时间）、手动失效、混合策略
2. **预检测机制**：后台异步预检测，提前缓存结果
3. **多设备隔离**：不同设备的缓存互不干扰
4. **线程安全**：支持多线程并发访问

## 基本使用

### 1. 创建缓存管理器

```python
from src.page_detector_cache import PageDetectorCache, CacheInvalidationStrategy

# 创建缓存管理器（默认TTL为0.5秒）
cache = PageDetectorCache(default_ttl=0.5)

# 或指定其他参数
cache = PageDetectorCache(
    default_ttl=1.0,  # 默认缓存1秒
    strategy=CacheInvalidationStrategy.TTL,  # 使用TTL策略
    max_cache_size=100  # 最多缓存100个条目
)
```

### 2. 设置和获取缓存

```python
# 设置缓存
cache.set("device1", detection_result)

# 获取缓存
result = cache.get("device1")
if result is not None:
    print("使用缓存结果")
else:
    print("缓存不存在或已过期")
```

### 3. 使用多个缓存键

```python
# 为同一设备缓存不同类型的检测结果
cache.set("device1", page_type_result, key="page_type")
cache.set("device1", elements_result, key="elements")

# 获取不同的缓存
page_type = cache.get("device1", key="page_type")
elements = cache.get("device1", key="elements")
```

### 4. 自定义TTL

```python
# 为特定缓存设置不同的TTL
cache.set("device1", result, ttl=2.0)  # 缓存2秒
cache.set("device2", result, ttl=0)    # 永不过期
```

## 高级功能

### 1. 获取或检测（自动缓存）

```python
async def detect_page(device_id: str):
    """执行页面检测"""
    # ... 检测逻辑
    return result

# 自动处理缓存查询和检测
result = await cache.get_or_detect(
    device_id="device1",
    detect_func=detect_page,
    use_cache=True  # 使用缓存
)
```

### 2. 预检测（后台异步）

```python
# 启动预检测（不阻塞）
await cache.predetect(
    device_id="device1",
    detect_func=detect_page
)

# 继续执行其他操作...

# 稍后获取预检测的结果
result = cache.get("device1")  # 可能已经有缓存了
```

### 3. 批量预检测

```python
# 为多个设备同时预检测
device_ids = ["device1", "device2", "device3"]
await cache.predetect_batch(device_ids, detect_page)

# 等待一段时间后，所有设备都有缓存了
```

## 缓存管理

### 1. 手动失效缓存

```python
# 失效单个缓存键
cache.invalidate("device1", key="page_type")

# 失效设备的所有缓存
cache.invalidate("device1", key=None)
```

### 2. 清除缓存

```python
# 清除指定设备的缓存
cache.clear(device_id="device1")

# 清除所有缓存
cache.clear()
```

### 3. 清理过期缓存

```python
# 手动清理所有过期的缓存条目
cache.cleanup_expired()
```

## 监控和统计

### 1. 获取统计信息

```python
# 获取所有设备的统计
stats = cache.get_stats()
print(f"总条目数: {stats['total_entries']}")
print(f"有效条目: {stats['valid_entries']}")
print(f"过期条目: {stats['expired_entries']}")
print(f"平均年龄: {stats['average_age']:.2f}秒")

# 获取特定设备的统计
stats = cache.get_stats(device_id="device1")
```

### 2. 检查缓存状态

```python
# 检查设备是否有缓存
if "device1" in cache:
    print("device1 有缓存")

# 获取缓存条目数量
count = len(cache)
print(f"当前缓存条目数: {count}")
```

## 集成到页面检测器

### 示例：在页面检测器中使用缓存

```python
from src.page_detector_cache import PageDetectorCache

class MyPageDetector:
    def __init__(self):
        # 创建缓存管理器
        self._cache = PageDetectorCache(default_ttl=0.5)
    
    async def detect_page(self, device_id: str, use_cache: bool = True):
        """检测页面（带缓存）"""
        # 使用 get_or_detect 自动处理缓存
        return await self._cache.get_or_detect(
            device_id=device_id,
            detect_func=self._do_detect,
            use_cache=use_cache
        )
    
    async def _do_detect(self, device_id: str):
        """实际的检测逻辑"""
        # ... 执行检测
        return result
    
    def clear_cache(self, device_id: str = None):
        """清除缓存"""
        self._cache.clear(device_id)
```

## 性能优化建议

### 1. 选择合适的TTL

```python
# 快速变化的页面：短TTL
cache = PageDetectorCache(default_ttl=0.3)

# 稳定的页面：长TTL
cache = PageDetectorCache(default_ttl=2.0)

# 静态内容：永不过期
cache.set(device_id, result, ttl=0)
```

### 2. 使用预检测

```python
# 在流程开始前预检测常用页面
async def prepare_workflow(device_ids):
    # 预检测所有设备的首页
    await cache.predetect_batch(device_ids, detect_homepage)
    
    # 继续执行其他准备工作
    # ...
    
    # 稍后使用时，缓存已经准备好了
    for device_id in device_ids:
        result = cache.get(device_id)  # 立即返回
```

### 3. 定期清理过期缓存

```python
import asyncio

async def periodic_cleanup():
    """定期清理过期缓存"""
    while True:
        await asyncio.sleep(60)  # 每分钟清理一次
        cache.cleanup_expired()
        stats = cache.get_stats()
        print(f"清理后剩余 {stats['valid_entries']} 个有效缓存")
```

## 最佳实践

1. **合理设置TTL**：根据页面变化频率设置合适的TTL
2. **使用预检测**：在需要结果之前提前检测，减少等待时间
3. **及时失效缓存**：在页面状态改变后手动失效相关缓存
4. **监控缓存命中率**：定期检查统计信息，优化缓存策略
5. **限制缓存大小**：设置合理的 max_cache_size 避免内存占用过大

## 注意事项

1. **线程安全**：缓存管理器是线程安全的，可以在多线程环境中使用
2. **异步支持**：预检测和 get_or_detect 需要在异步环境中使用
3. **内存管理**：缓存会占用内存，注意设置合理的 max_cache_size
4. **缓存一致性**：在页面状态改变后，记得手动失效相关缓存
