"""
页面检测缓存管理器
Page Detector Cache Manager

提供统一的页面检测缓存管理，支持：
1. 缓存失效策略（TTL、手动失效）
2. 预检测机制
3. 多设备缓存隔离
"""

import time
import asyncio
from typing import Optional, Dict, Any, Callable, Awaitable
from dataclasses import dataclass, field
from threading import Lock
from enum import Enum


class CacheInvalidationStrategy(Enum):
    """缓存失效策略"""
    TTL = "ttl"  # 基于时间的失效
    MANUAL = "manual"  # 手动失效
    HYBRID = "hybrid"  # 混合策略（TTL + 手动）


@dataclass
class CacheEntry:
    """缓存条目"""
    result: Any  # 检测结果
    timestamp: float  # 缓存时间戳
    ttl: float  # 生存时间（秒）
    device_id: str  # 设备ID
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据
    
    def is_expired(self) -> bool:
        """检查缓存是否过期"""
        if self.ttl <= 0:
            return False  # TTL为0表示永不过期
        return (time.time() - self.timestamp) > self.ttl
    
    def age(self) -> float:
        """获取缓存年龄（秒）"""
        return time.time() - self.timestamp


class PageDetectorCache:
    """页面检测缓存管理器
    
    特性：
    1. 支持多种缓存失效策略
    2. 支持预检测机制
    3. 线程安全
    4. 多设备缓存隔离
    """
    
    def __init__(self, 
                 default_ttl: float = 0.5,
                 strategy: CacheInvalidationStrategy = CacheInvalidationStrategy.TTL,
                 max_cache_size: int = 100):
        """初始化缓存管理器
        
        Args:
            default_ttl: 默认缓存生存时间（秒），0表示永不过期
            strategy: 缓存失效策略
            max_cache_size: 最大缓存条目数
        """
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = Lock()
        self._default_ttl = default_ttl
        self._strategy = strategy
        self._max_cache_size = max_cache_size
        
        # 预检测相关
        self._predetect_tasks: Dict[str, asyncio.Task] = {}
        self._predetect_lock = Lock()
    
    def get(self, device_id: str, key: str = "default") -> Optional[Any]:
        """获取缓存的检测结果
        
        Args:
            device_id: 设备ID
            key: 缓存键（用于区分同一设备的不同检测类型）
            
        Returns:
            缓存的检测结果，如果不存在或已过期返回None
        """
        cache_key = self._make_cache_key(device_id, key)
        
        with self._lock:
            entry = self._cache.get(cache_key)
            
            if entry is None:
                return None
            
            # 检查是否过期
            if self._strategy in [CacheInvalidationStrategy.TTL, CacheInvalidationStrategy.HYBRID]:
                if entry.is_expired():
                    # 删除过期缓存
                    del self._cache[cache_key]
                    return None
            
            return entry.result
    
    def set(self, device_id: str, result: Any, key: str = "default", 
            ttl: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None):
        """设置缓存
        
        Args:
            device_id: 设备ID
            result: 检测结果
            key: 缓存键
            ttl: 生存时间（秒），None表示使用默认值
            metadata: 额外元数据
        """
        cache_key = self._make_cache_key(device_id, key)
        
        if ttl is None:
            ttl = self._default_ttl
        
        entry = CacheEntry(
            result=result,
            timestamp=time.time(),
            ttl=ttl,
            device_id=device_id,
            metadata=metadata or {}
        )
        
        with self._lock:
            # 检查缓存大小限制
            if len(self._cache) >= self._max_cache_size:
                # 删除最旧的缓存条目
                self._evict_oldest()
            
            self._cache[cache_key] = entry
    
    def invalidate(self, device_id: str, key: str = "default"):
        """手动失效缓存
        
        Args:
            device_id: 设备ID
            key: 缓存键，None表示失效该设备的所有缓存
        """
        if key is None:
            # 失效该设备的所有缓存
            with self._lock:
                keys_to_delete = [k for k in self._cache.keys() if k.startswith(f"{device_id}:")]
                for k in keys_to_delete:
                    del self._cache[k]
        else:
            cache_key = self._make_cache_key(device_id, key)
            with self._lock:
                if cache_key in self._cache:
                    del self._cache[cache_key]
    
    def clear(self, device_id: Optional[str] = None):
        """清除缓存
        
        Args:
            device_id: 设备ID，None表示清除所有缓存
        """
        with self._lock:
            if device_id is None:
                self._cache.clear()
            else:
                # 只清除指定设备的缓存
                keys_to_delete = [k for k in self._cache.keys() if k.startswith(f"{device_id}:")]
                for k in keys_to_delete:
                    del self._cache[k]
    
    def get_stats(self, device_id: Optional[str] = None) -> Dict[str, Any]:
        """获取缓存统计信息
        
        Args:
            device_id: 设备ID，None表示所有设备
            
        Returns:
            统计信息字典
        """
        with self._lock:
            if device_id is None:
                entries = list(self._cache.values())
            else:
                entries = [e for k, e in self._cache.items() if k.startswith(f"{device_id}:")]
            
            total_count = len(entries)
            expired_count = sum(1 for e in entries if e.is_expired())
            valid_count = total_count - expired_count
            
            avg_age = sum(e.age() for e in entries) / total_count if total_count > 0 else 0
            
            return {
                "total_entries": total_count,
                "valid_entries": valid_count,
                "expired_entries": expired_count,
                "average_age": avg_age,
                "strategy": self._strategy.value,
                "default_ttl": self._default_ttl
            }
    
    async def get_or_detect(self, 
                           device_id: str,
                           detect_func: Callable[[str], Awaitable[Any]],
                           key: str = "default",
                           use_cache: bool = True,
                           ttl: Optional[float] = None) -> Any:
        """获取缓存或执行检测
        
        这是一个便捷方法，自动处理缓存查询和检测逻辑
        
        Args:
            device_id: 设备ID
            detect_func: 检测函数（异步）
            key: 缓存键
            use_cache: 是否使用缓存
            ttl: 生存时间
            
        Returns:
            检测结果
        """
        # 尝试从缓存获取
        if use_cache:
            cached_result = self.get(device_id, key)
            if cached_result is not None:
                return cached_result
        
        # 执行检测
        result = await detect_func(device_id)
        
        # 更新缓存
        if use_cache:
            self.set(device_id, result, key, ttl)
        
        return result
    
    async def predetect(self,
                       device_id: str,
                       detect_func: Callable[[str], Awaitable[Any]],
                       key: str = "default",
                       ttl: Optional[float] = None):
        """预检测（后台异步执行）
        
        在后台执行检测并缓存结果，不阻塞当前操作
        
        Args:
            device_id: 设备ID
            detect_func: 检测函数（异步）
            key: 缓存键
            ttl: 生存时间
        """
        task_key = self._make_cache_key(device_id, key)
        
        # 检查是否已有预检测任务在运行
        with self._predetect_lock:
            if task_key in self._predetect_tasks:
                existing_task = self._predetect_tasks[task_key]
                if not existing_task.done():
                    # 任务还在运行，不重复创建
                    return
        
        # 创建预检测任务
        async def _predetect_task():
            try:
                result = await detect_func(device_id)
                self.set(device_id, result, key, ttl)
            except Exception as e:
                # 预检测失败不影响主流程
                pass
            finally:
                # 清理任务引用
                with self._predetect_lock:
                    if task_key in self._predetect_tasks:
                        del self._predetect_tasks[task_key]
        
        task = asyncio.create_task(_predetect_task())
        
        with self._predetect_lock:
            self._predetect_tasks[task_key] = task
    
    async def predetect_batch(self,
                             device_ids: list[str],
                             detect_func: Callable[[str], Awaitable[Any]],
                             key: str = "default",
                             ttl: Optional[float] = None):
        """批量预检测
        
        为多个设备同时执行预检测
        
        Args:
            device_ids: 设备ID列表
            detect_func: 检测函数（异步）
            key: 缓存键
            ttl: 生存时间
        """
        tasks = []
        for device_id in device_ids:
            task = self.predetect(device_id, detect_func, key, ttl)
            tasks.append(task)
        
        # 等待所有预检测任务启动
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def _make_cache_key(self, device_id: str, key: str) -> str:
        """生成缓存键
        
        Args:
            device_id: 设备ID
            key: 缓存键
            
        Returns:
            完整的缓存键
        """
        return f"{device_id}:{key}"
    
    def _evict_oldest(self):
        """驱逐最旧的缓存条目（LRU策略）
        
        注意：此方法应在持有锁的情况下调用
        """
        if not self._cache:
            return
        
        # 找到最旧的条目
        oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].timestamp)
        del self._cache[oldest_key]
    
    def cleanup_expired(self):
        """清理所有过期的缓存条目
        
        这个方法可以定期调用以释放内存
        """
        with self._lock:
            expired_keys = [
                k for k, e in self._cache.items()
                if e.is_expired()
            ]
            for k in expired_keys:
                del self._cache[k]
    
    def __len__(self) -> int:
        """返回缓存条目数量"""
        with self._lock:
            return len(self._cache)
    
    def __contains__(self, device_id: str) -> bool:
        """检查设备是否有缓存"""
        with self._lock:
            return any(k.startswith(f"{device_id}:") for k in self._cache.keys())
