"""
页面变化检测器
Page Change Detector - 使用感知哈希快速检测页面变化
"""

import asyncio
from typing import Optional
from io import BytesIO

try:
    from PIL import Image
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False


class PageChangeDetector:
    """页面变化检测器 - 使用感知哈希"""
    
    def __init__(self, hash_size: int = 8):
        """初始化检测器
        
        Args:
            hash_size: 哈希大小，越大越精确但越慢（默认8）
        """
        self.hash_size = hash_size
        self._last_hash = None
        self._last_screenshot = None
    
    async def detect_change(self, screenshot_data: bytes, threshold: int = 5) -> tuple[bool, int]:
        """检测页面是否变化
        
        Args:
            screenshot_data: 截图数据（bytes）
            threshold: 变化阈值，哈希距离大于此值认为变化（默认5）
            
        Returns:
            (是否变化, 哈希距离)
        """
        if not HAS_IMAGEHASH:
            return (False, 0)
        
        try:
            # 转换为PIL图像
            image = Image.open(BytesIO(screenshot_data))
            
            # 计算感知哈希
            current_hash = imagehash.phash(image, hash_size=self.hash_size)
            
            # 如果是第一次检测
            if self._last_hash is None:
                self._last_hash = current_hash
                self._last_screenshot = screenshot_data
                return (False, 0)
            
            # 计算哈希距离
            distance = current_hash - self._last_hash
            
            # 判断是否变化
            changed = distance > threshold
            
            # 如果变化，更新记录
            if changed:
                self._last_hash = current_hash
                self._last_screenshot = screenshot_data
            
            return (changed, distance)
            
        except Exception as e:
            return (False, 0)
    
    def reset(self):
        """重置检测器状态"""
        self._last_hash = None
        self._last_screenshot = None
    
    def get_last_screenshot(self) -> Optional[bytes]:
        """获取上次的截图数据"""
        return self._last_screenshot
