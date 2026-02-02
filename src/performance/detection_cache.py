"""
页面检测结果缓存
Detection Result Cache
"""

import time
from typing import Dict, Optional, Tuple


class DetectionCache:
    """页面检测结果缓存
    
    用于缓存页面检测结果，避免短时间内重复检测同一页面。
    默认缓存有效期为0.5秒。
    """
    
    def __init__(self, ttl: float = 0.5):
        """初始化缓存
        
        Args:
            ttl: 缓存有效期（秒），默认0.5秒
        """
        self._cache: Dict[str, Tuple[any, float]] = {}
        self._ttl = ttl
    
    def get(self, device_id: str) -> Optional[any]:
        """获取缓存的检测结果
        
        Args:
            device_id: 设备ID
            
        Returns:
            缓存的检测结果，如果缓存不存在或已过期则返回None
        """
        if device_id not in self._cache:
            return None
        
        result, timestamp = self._cache[device_id]
        
        # 检查是否过期
        if time.time() - timestamp > self._ttl:
            # 缓存已过期，删除并返回None
            del self._cache[device_id]
            return None
        
        return result
    
    def set(self, device_id: str, result: any):
        """设置缓存
        
        Args:
            device_id: 设备ID
            result: 检测结果
        """
        self._cache[device_id] = (result, time.time())
    
    def clear(self, device_id: str = None):
        """清除缓存
        
        Args:
            device_id: 设备ID，如果为None则清除所有缓存
        """
        if device_id is None:
            self._cache.clear()
        elif device_id in self._cache:
            del self._cache[device_id]
    
    def get_stats(self) -> Dict[str, int]:
        """获取缓存统计信息
        
        Returns:
            包含缓存大小和有效缓存数的字典
        """
        current_time = time.time()
        valid_count = sum(
            1 for _, timestamp in self._cache.values()
            if current_time - timestamp <= self._ttl
        )
        
        return {
            'total': len(self._cache),
            'valid': valid_count,
            'expired': len(self._cache) - valid_count
        }
