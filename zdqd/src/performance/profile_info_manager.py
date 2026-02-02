"""
个人信息管理器 - 避免重复获取个人信息
Profile Info Manager - Avoid redundant profile info fetching
"""

import time
import asyncio
from typing import Dict, Optional, Any


class ProfileInfoManager:
    """个人信息管理器 - 使用缓存避免重复获取"""
    
    def __init__(self, ttl: float = 60.0):
        """初始化管理器
        
        Args:
            ttl: 缓存有效期（秒），默认60秒
        """
        self._cache: Dict[str, Dict[str, Any]] = {}  # {device_id: {profile_data, timestamp}}
        self._ttl = ttl
        
        # 性能统计
        self._stats = {
            'cache_hits': 0,      # 缓存命中次数
            'cache_misses': 0,    # 缓存未命中次数
            'total_time_saved': 0.0  # 总节省时间（秒）
        }
    
    async def get_profile_info(
        self, 
        device_id: str, 
        profile_reader, 
        account: Optional[str] = None, 
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """获取个人信息（带缓存）
        
        Args:
            device_id: 设备ID
            profile_reader: ProfileReader实例
            account: 登录账号（可选）
            force_refresh: 是否强制刷新缓存
            
        Returns:
            dict: 个人信息
        """
        now = time.time()
        
        # 检查缓存
        if not force_refresh and device_id in self._cache:
            cached = self._cache[device_id]
            cache_age = now - cached['timestamp']
            
            if cache_age < self._ttl:
                self._stats['cache_hits'] += 1
                self._stats['total_time_saved'] += 5.0  # 假设每次节省5秒
                
                print(f"  ✓ [优化-缓存] 使用缓存的个人信息 (缓存时间: {cache_age:.1f}秒, 节省约5秒)")
                print(f"  ✓ [优化-统计] 缓存命中 {self._stats['cache_hits']} 次, 累计节省 {self._stats['total_time_saved']:.1f} 秒")
                return cached['profile_data']
        
        # 获取新数据前，先检查页面状态
        self._stats['cache_misses'] += 1
        print(f"  → [优化-获取] 获取新的个人信息 (缓存未命中: {self._stats['cache_misses']} 次)...")
        
        # 检查是否在积分页，如果是则需要返回首页再导航到个人页
        try:
            from .page_detector_hybrid import PageDetectorHybrid, PageState
            from .adb_bridge import ADBBridge
            
            # 获取 ADB 实例（从 profile_reader 中获取）
            adb = profile_reader.adb
            
            # 创建临时的页面检测器
            detector = PageDetectorHybrid(adb)
            
            # 检测当前页面
            print(f"  → [页面检查] 检测当前页面状态...")
            page_templates = ['积分页.png', '已登陆个人页.png', '未登陆个人页.png']
            page_result = await detector.detect_page_with_priority(
                device_id, page_templates, use_cache=False
            )
            
            # 如果在积分页，需要返回首页再导航到个人页
            if page_result and page_result.state == PageState.PROFILE_LOGGED:
                # 检查是否是积分页（通过模板名称判断）
                if page_result.details and '积分页.png' in page_result.details:
                    print(f"  ⚠️ [页面检查] 检测到积分页，需要返回首页后重新导航")
                    
                    # 按返回键返回首页
                    await adb.press_back(device_id)
                    await asyncio.sleep(1)
                    
                    # 点击底部"我的"按钮导航到个人页
                    print(f"  → [页面检查] 点击'我的'按钮导航到个人页...")
                    MY_TAB = (446, 949)
                    await adb.tap(device_id, MY_TAB[0], MY_TAB[1])
                    await asyncio.sleep(2)
                    
                    print(f"  ✓ [页面检查] 已重新导航到个人页")
        except Exception as e:
            print(f"  ⚠️ [页面检查] 页面检查失败: {e}，继续尝试获取")
        
        fetch_start = time.time()
        profile_data = await profile_reader.get_full_profile_parallel(device_id, account)
        fetch_time = time.time() - fetch_start
        
        # 更新缓存
        self._cache[device_id] = {
            'profile_data': profile_data,
            'timestamp': now
        }
        
        print(f"  ✓ [优化-缓存] 个人信息已缓存 (耗时: {fetch_time:.2f}秒)")
        return profile_data
    
    def get_stats(self) -> Dict[str, Any]:
        """获取性能统计信息
        
        Returns:
            dict: 统计信息
        """
        total_requests = self._stats['cache_hits'] + self._stats['cache_misses']
        hit_rate = (self._stats['cache_hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_hits': self._stats['cache_hits'],
            'cache_misses': self._stats['cache_misses'],
            'total_requests': total_requests,
            'hit_rate': hit_rate,
            'time_saved': self._stats['total_time_saved']
        }
    
    def print_stats(self):
        """打印性能统计信息"""
        stats = self.get_stats()
        print(f"\n  ═══ [优化统计] ProfileInfoManager ═══")
        print(f"  缓存命中: {stats['cache_hits']} 次")
        print(f"  缓存未命中: {stats['cache_misses']} 次")
        print(f"  命中率: {stats['hit_rate']:.1f}%")
        print(f"  节省时间: {stats['time_saved']:.1f} 秒")
        print(f"  ═══════════════════════════════════\n")
    
    def clear(self, device_id: Optional[str] = None):
        """清除缓存
        
        Args:
            device_id: 设备ID，如果为None则清除所有缓存
        """
        if device_id:
            if device_id in self._cache:
                self._cache.pop(device_id)
                print(f"  [ProfileInfoManager] 已清除设备 {device_id} 的缓存")
        else:
            self._cache.clear()
            print(f"  [ProfileInfoManager] 已清除所有缓存")
    
    def has_cache(self, device_id: str) -> bool:
        """检查是否有缓存
        
        Args:
            device_id: 设备ID
            
        Returns:
            bool: 是否有有效缓存
        """
        if device_id not in self._cache:
            return False
        
        now = time.time()
        cached = self._cache[device_id]
        return now - cached['timestamp'] < self._ttl
