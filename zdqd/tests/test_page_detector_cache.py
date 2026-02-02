"""
测试页面检测缓存管理器
Test Page Detector Cache Manager
"""

import pytest
import asyncio
import time
from src.page_detector_cache import (
    PageDetectorCache,
    CacheInvalidationStrategy,
    CacheEntry
)


class TestCacheEntry:
    """测试缓存条目"""
    
    def test_cache_entry_creation(self):
        """测试缓存条目创建"""
        entry = CacheEntry(
            result="test_result",
            timestamp=time.time(),
            ttl=1.0,
            device_id="device1"
        )
        
        assert entry.result == "test_result"
        assert entry.device_id == "device1"
        assert entry.ttl == 1.0
        assert not entry.is_expired()
    
    def test_cache_entry_expiration(self):
        """测试缓存过期"""
        entry = CacheEntry(
            result="test_result",
            timestamp=time.time() - 2.0,  # 2秒前
            ttl=1.0,  # 1秒TTL
            device_id="device1"
        )
        
        assert entry.is_expired()
    
    def test_cache_entry_no_expiration(self):
        """测试永不过期的缓存"""
        entry = CacheEntry(
            result="test_result",
            timestamp=time.time() - 100.0,  # 100秒前
            ttl=0,  # TTL为0表示永不过期
            device_id="device1"
        )
        
        assert not entry.is_expired()
    
    def test_cache_entry_age(self):
        """测试缓存年龄"""
        entry = CacheEntry(
            result="test_result",
            timestamp=time.time() - 1.5,
            ttl=10.0,
            device_id="device1"
        )
        
        age = entry.age()
        assert 1.4 < age < 1.6  # 允许小误差


class TestPageDetectorCache:
    """测试页面检测缓存管理器"""
    
    def test_cache_initialization(self):
        """测试缓存初始化"""
        cache = PageDetectorCache(default_ttl=1.0)
        
        assert cache._default_ttl == 1.0
        assert cache._strategy == CacheInvalidationStrategy.TTL
        assert len(cache) == 0
    
    def test_set_and_get(self):
        """测试设置和获取缓存"""
        cache = PageDetectorCache(default_ttl=10.0)
        
        # 设置缓存
        cache.set("device1", "result1")
        
        # 获取缓存
        result = cache.get("device1")
        assert result == "result1"
    
    def test_get_nonexistent(self):
        """测试获取不存在的缓存"""
        cache = PageDetectorCache()
        
        result = cache.get("nonexistent_device")
        assert result is None
    
    def test_cache_expiration(self):
        """测试缓存过期"""
        cache = PageDetectorCache(default_ttl=0.1)  # 0.1秒TTL
        
        # 设置缓存
        cache.set("device1", "result1")
        
        # 立即获取应该成功
        result = cache.get("device1")
        assert result == "result1"
        
        # 等待过期
        time.sleep(0.2)
        
        # 过期后应该返回None
        result = cache.get("device1")
        assert result is None
    
    def test_custom_ttl(self):
        """测试自定义TTL"""
        cache = PageDetectorCache(default_ttl=10.0)
        
        # 使用自定义TTL
        cache.set("device1", "result1", ttl=0.1)
        
        # 立即获取成功
        assert cache.get("device1") == "result1"
        
        # 等待过期
        time.sleep(0.2)
        
        # 过期后返回None
        assert cache.get("device1") is None
    
    def test_multiple_keys(self):
        """测试多个缓存键"""
        cache = PageDetectorCache(default_ttl=10.0)
        
        # 设置多个键
        cache.set("device1", "result1", key="page_type")
        cache.set("device1", "result2", key="elements")
        
        # 获取不同的键
        assert cache.get("device1", key="page_type") == "result1"
        assert cache.get("device1", key="elements") == "result2"
    
    def test_multiple_devices(self):
        """测试多设备缓存隔离"""
        cache = PageDetectorCache(default_ttl=10.0)
        
        # 设置不同设备的缓存
        cache.set("device1", "result1")
        cache.set("device2", "result2")
        
        # 获取应该隔离
        assert cache.get("device1") == "result1"
        assert cache.get("device2") == "result2"
    
    def test_invalidate_single_key(self):
        """测试失效单个缓存键"""
        cache = PageDetectorCache(default_ttl=10.0)
        
        cache.set("device1", "result1", key="key1")
        cache.set("device1", "result2", key="key2")
        
        # 失效key1
        cache.invalidate("device1", key="key1")
        
        # key1应该被删除，key2仍然存在
        assert cache.get("device1", key="key1") is None
        assert cache.get("device1", key="key2") == "result2"
    
    def test_invalidate_all_device_keys(self):
        """测试失效设备的所有缓存"""
        cache = PageDetectorCache(default_ttl=10.0)
        
        cache.set("device1", "result1", key="key1")
        cache.set("device1", "result2", key="key2")
        cache.set("device2", "result3", key="key1")
        
        # 失效device1的所有缓存
        cache.invalidate("device1", key=None)
        
        # device1的缓存应该被删除
        assert cache.get("device1", key="key1") is None
        assert cache.get("device1", key="key2") is None
        
        # device2的缓存应该保留
        assert cache.get("device2", key="key1") == "result3"
    
    def test_clear_all(self):
        """测试清除所有缓存"""
        cache = PageDetectorCache(default_ttl=10.0)
        
        cache.set("device1", "result1")
        cache.set("device2", "result2")
        
        # 清除所有缓存
        cache.clear()
        
        assert len(cache) == 0
        assert cache.get("device1") is None
        assert cache.get("device2") is None
    
    def test_clear_device(self):
        """测试清除指定设备的缓存"""
        cache = PageDetectorCache(default_ttl=10.0)
        
        cache.set("device1", "result1")
        cache.set("device2", "result2")
        
        # 只清除device1
        cache.clear(device_id="device1")
        
        assert cache.get("device1") is None
        assert cache.get("device2") == "result2"
    
    def test_max_cache_size(self):
        """测试最大缓存大小限制"""
        cache = PageDetectorCache(default_ttl=10.0, max_cache_size=3)
        
        # 添加4个缓存条目
        cache.set("device1", "result1")
        cache.set("device2", "result2")
        cache.set("device3", "result3")
        cache.set("device4", "result4")
        
        # 应该只保留3个（最旧的被驱逐）
        assert len(cache) == 3
        assert cache.get("device1") is None  # 最旧的被删除
        assert cache.get("device4") == "result4"  # 最新的保留
    
    def test_get_stats(self):
        """测试获取统计信息"""
        cache = PageDetectorCache(default_ttl=1.0)
        
        cache.set("device1", "result1")
        cache.set("device2", "result2")
        
        stats = cache.get_stats()
        
        assert stats["total_entries"] == 2
        assert stats["valid_entries"] == 2
        assert stats["expired_entries"] == 0
        assert stats["strategy"] == "ttl"
        assert stats["default_ttl"] == 1.0
    
    def test_get_stats_with_expired(self):
        """测试统计信息包含过期条目"""
        cache = PageDetectorCache(default_ttl=0.1)
        
        cache.set("device1", "result1")
        time.sleep(0.2)  # 等待过期
        cache.set("device2", "result2")
        
        stats = cache.get_stats()
        
        assert stats["total_entries"] == 2
        assert stats["valid_entries"] == 1
        assert stats["expired_entries"] == 1
    
    def test_cleanup_expired(self):
        """测试清理过期缓存"""
        cache = PageDetectorCache(default_ttl=0.1)
        
        cache.set("device1", "result1")
        cache.set("device2", "result2", ttl=10.0)  # 不会过期
        
        time.sleep(0.2)  # 等待device1过期
        
        # 清理前有2个条目
        assert len(cache) == 2
        
        # 清理过期缓存
        cache.cleanup_expired()
        
        # 清理后只剩1个
        assert len(cache) == 1
        assert cache.get("device1") is None
        assert cache.get("device2") == "result2"
    
    def test_contains(self):
        """测试设备是否有缓存"""
        cache = PageDetectorCache(default_ttl=10.0)
        
        cache.set("device1", "result1")
        
        assert "device1" in cache
        assert "device2" not in cache
    
    @pytest.mark.asyncio
    async def test_get_or_detect(self):
        """测试获取或检测"""
        cache = PageDetectorCache(default_ttl=10.0)
        
        # 模拟检测函数
        detect_count = 0
        
        async def mock_detect(device_id: str):
            nonlocal detect_count
            detect_count += 1
            await asyncio.sleep(0.01)  # 模拟检测耗时
            return f"result_{device_id}"
        
        # 第一次调用应该执行检测
        result1 = await cache.get_or_detect("device1", mock_detect)
        assert result1 == "result_device1"
        assert detect_count == 1
        
        # 第二次调用应该使用缓存
        result2 = await cache.get_or_detect("device1", mock_detect)
        assert result2 == "result_device1"
        assert detect_count == 1  # 没有增加
    
    @pytest.mark.asyncio
    async def test_get_or_detect_no_cache(self):
        """测试不使用缓存的获取或检测"""
        cache = PageDetectorCache(default_ttl=10.0)
        
        detect_count = 0
        
        async def mock_detect(device_id: str):
            nonlocal detect_count
            detect_count += 1
            return f"result_{detect_count}"
        
        # 禁用缓存，每次都应该执行检测
        result1 = await cache.get_or_detect("device1", mock_detect, use_cache=False)
        result2 = await cache.get_or_detect("device1", mock_detect, use_cache=False)
        
        assert result1 == "result_1"
        assert result2 == "result_2"
        assert detect_count == 2
    
    @pytest.mark.asyncio
    async def test_predetect(self):
        """测试预检测"""
        cache = PageDetectorCache(default_ttl=10.0)
        
        async def mock_detect(device_id: str):
            await asyncio.sleep(0.1)  # 模拟检测耗时
            return f"result_{device_id}"
        
        # 启动预检测（不阻塞）
        await cache.predetect("device1", mock_detect)
        
        # 立即检查缓存应该还没有结果
        result = cache.get("device1")
        assert result is None
        
        # 等待预检测完成
        await asyncio.sleep(0.2)
        
        # 现在应该有缓存了
        result = cache.get("device1")
        assert result == "result_device1"
    
    @pytest.mark.asyncio
    async def test_predetect_batch(self):
        """测试批量预检测"""
        cache = PageDetectorCache(default_ttl=10.0)
        
        async def mock_detect(device_id: str):
            await asyncio.sleep(0.1)
            return f"result_{device_id}"
        
        # 批量预检测
        device_ids = ["device1", "device2", "device3"]
        await cache.predetect_batch(device_ids, mock_detect)
        
        # 等待完成
        await asyncio.sleep(0.2)
        
        # 所有设备都应该有缓存
        for device_id in device_ids:
            result = cache.get(device_id)
            assert result == f"result_{device_id}"
    
    @pytest.mark.asyncio
    async def test_predetect_no_duplicate(self):
        """测试预检测不会重复执行"""
        cache = PageDetectorCache(default_ttl=10.0)
        
        detect_count = 0
        
        async def mock_detect(device_id: str):
            nonlocal detect_count
            detect_count += 1
            await asyncio.sleep(0.2)
            return f"result_{device_id}"
        
        # 连续启动两次预检测
        await cache.predetect("device1", mock_detect)
        await cache.predetect("device1", mock_detect)
        
        # 等待完成
        await asyncio.sleep(0.3)
        
        # 应该只执行一次
        assert detect_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
