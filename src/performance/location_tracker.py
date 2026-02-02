"""
位置跟踪器 - 避免重复导航
Location Tracker - Avoid redundant navigation
"""

from typing import Dict, Optional, Any
from enum import Enum


class LocationTracker:
    """位置跟踪器 - 跟踪当前所在页面，避免重复导航"""
    
    def __init__(self):
        """初始化跟踪器"""
        self._current_page: Dict[str, Any] = {}  # {device_id: PageState}
        
        # 性能统计
        self._stats = {
            'navigations_saved': 0,  # 避免的导航次数
            'total_time_saved': 0.0  # 总节省时间（秒）
        }
    
    def set_page(self, device_id: str, page_state):
        """设置当前页面
        
        Args:
            device_id: 设备ID
            page_state: 页面状态（PageState枚举）
        """
        self._current_page[device_id] = page_state
        page_name = page_state.value if hasattr(page_state, 'value') else page_state
        print(f"  → [优化-位置] 记录当前页面: {page_name}")
    
    def check_and_save_navigation(self, device_id: str, target_page: str) -> bool:
        """检查是否需要导航,如果不需要则记录节省
        
        Args:
            device_id: 设备ID
            target_page: 目标页面类型 ('profile' 或 'home')
            
        Returns:
            bool: 是否需要导航 (True=需要, False=不需要)
        """
        if target_page == 'profile' and self.is_at_profile(device_id):
            self._stats['navigations_saved'] += 1
            self._stats['total_time_saved'] += 2.0  # 假设每次节省2秒
            print(f"  ✓ [优化-导航] 已在个人页,无需导航 (节省约2秒)")
            print(f"  ✓ [优化-统计] 避免导航 {self._stats['navigations_saved']} 次, 累计节省 {self._stats['total_time_saved']:.1f} 秒")
            return False
        elif target_page == 'home' and self.is_at_home(device_id):
            self._stats['navigations_saved'] += 1
            self._stats['total_time_saved'] += 2.0
            print(f"  ✓ [优化-导航] 已在首页,无需导航 (节省约2秒)")
            print(f"  ✓ [优化-统计] 避免导航 {self._stats['navigations_saved']} 次, 累计节省 {self._stats['total_time_saved']:.1f} 秒")
            return False
        
        return True
    
    def get_page(self, device_id: str):
        """获取当前页面
        
        Args:
            device_id: 设备ID
            
        Returns:
            PageState: 当前页面状态，如果未跟踪则返回None
        """
        return self._current_page.get(device_id)
    
    def is_at_profile(self, device_id: str) -> bool:
        """是否在个人页
        
        Args:
            device_id: 设备ID
            
        Returns:
            bool: 是否在个人页（已登录或未登录）
        """
        from ..page_detector import PageState
        page = self.get_page(device_id)
        return page in [PageState.PROFILE, PageState.PROFILE_LOGGED]
    
    def is_at_home(self, device_id: str) -> bool:
        """是否在首页
        
        Args:
            device_id: 设备ID
            
        Returns:
            bool: 是否在首页
        """
        from ..page_detector import PageState
        page = self.get_page(device_id)
        return page == PageState.HOME
    
    def get_stats(self) -> Dict[str, Any]:
        """获取性能统计信息
        
        Returns:
            dict: 统计信息
        """
        return {
            'navigations_saved': self._stats['navigations_saved'],
            'time_saved': self._stats['total_time_saved']
        }
    
    def print_stats(self):
        """打印性能统计信息"""
        stats = self.get_stats()
        print(f"\n  ═══ [优化统计] LocationTracker ═══")
        print(f"  避免导航: {stats['navigations_saved']} 次")
        print(f"  节省时间: {stats['time_saved']:.1f} 秒")
        print(f"  ═══════════════════════════════════\n")
    
    def clear(self, device_id: Optional[str] = None):
        """清除状态
        
        Args:
            device_id: 设备ID，如果为None则清除所有状态
        """
        if device_id:
            if device_id in self._current_page:
                self._current_page.pop(device_id)
                print(f"  [LocationTracker] 已清除设备 {device_id} 的位置状态")
        else:
            self._current_page.clear()
            print(f"  [LocationTracker] 已清除所有位置状态")
