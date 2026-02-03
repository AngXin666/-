"""
UI 自动化模块
UI Automation Module
"""

import asyncio
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

from .adb_bridge import ADBBridge
from .screen_capture import ScreenCapture


class ActionType(Enum):
    """动作类型"""
    TAP = "tap"
    SWIPE = "swipe"
    INPUT = "input"
    WAIT = "wait"
    SCREENSHOT = "screenshot"


@dataclass
class Action:
    """UI 动作"""
    action_type: ActionType
    params: Dict[str, Any]
    delay_after: float = 0.5
    retry_count: int = 3
    timeout: int = 10


class UIAutomation:
    """UI 自动化器"""
    
    # 屏幕尺寸常量 (540x960)
    SCREEN_WIDTH = 540
    SCREEN_HEIGHT = 960
    
    def __init__(self, adb_bridge: ADBBridge, screen_capture: ScreenCapture):
        """初始化 UI 自动化器
        
        Args:
            adb_bridge: ADB 桥接器实例
            screen_capture: 屏幕捕获器实例
        """
        self.adb_bridge = adb_bridge
        self.screen_capture = screen_capture
    
    async def click_by_coords(self, device_id: str, x: int, y: int) -> bool:
        """通过坐标点击
        
        Args:
            device_id: 设备 ID
            x: X 坐标
            y: Y 坐标
            
        Returns:
            点击是否成功
        """
        return await self.adb_bridge.tap(device_id, x, y)
    
    async def click_by_image(self, device_id: str, template_path: str, 
                             threshold: float = 0.8, timeout: int = 10) -> bool:
        """通过图像匹配点击
        
        Args:
            device_id: 设备 ID
            template_path: 模板图像文件路径
            threshold: 匹配阈值
            timeout: 超时时间（秒）
            
        Returns:
            点击是否成功
        """
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            location = await self.screen_capture.find_image_from_file(
                device_id, template_path, threshold
            )
            if location:
                x, y = location
                return await self.adb_bridge.tap(device_id, x, y)
            await asyncio.sleep(0.5)
        
        return False
    
    async def click_by_text(self, device_id: str, text: str, 
                            timeout: int = 10) -> bool:
        """通过 OCR 文字匹配点击
        
        Args:
            device_id: 设备 ID
            text: 要点击的文字
            timeout: 超时时间（秒）
            
        Returns:
            点击是否成功
        """
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            location = await self.screen_capture.find_text_location(device_id, text)
            if location:
                x, y = location
                return await self.adb_bridge.tap(device_id, x, y)
            await asyncio.sleep(0.5)
        
        return False

    async def wait_for_element(self, device_id: str, template_path: str, 
                               threshold: float = 0.8, timeout: int = 15) -> bool:
        """等待元素出现
        
        Args:
            device_id: 设备 ID
            template_path: 模板图像文件路径
            threshold: 匹配阈值
            timeout: 超时时间（秒）
            
        Returns:
            元素是否出现
        """
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            location = await self.screen_capture.find_image_from_file(
                device_id, template_path, threshold
            )
            if location:
                return True
            await asyncio.sleep(0.5)
        
        return False
    
    async def wait_for_text(self, device_id: str, text: str, 
                            timeout: int = 15) -> bool:
        """等待文字出现
        
        Args:
            device_id: 设备 ID
            text: 要等待的文字
            timeout: 超时时间（秒）
            
        Returns:
            文字是否出现
        """
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            location = await self.screen_capture.find_text_location(device_id, text)
            if location:
                return True
            await asyncio.sleep(0.5)
        
        return False
    
    async def swipe(self, device_id: str, x1: int, y1: int, 
                    x2: int, y2: int, duration: int = 300) -> bool:
        """滑动操作
        
        Args:
            device_id: 设备 ID
            x1, y1: 起始坐标
            x2, y2: 结束坐标
            duration: 滑动持续时间（毫秒）
            
        Returns:
            滑动是否成功
        """
        return await self.adb_bridge.swipe(device_id, x1, y1, x2, y2, duration)
    
    async def input_text(self, device_id: str, text: str) -> bool:
        """输入文本
        
        Args:
            device_id: 设备 ID
            text: 要输入的文本
            
        Returns:
            输入是否成功
        """
        return await self.adb_bridge.input_text(device_id, text)
    
    async def execute_action(self, device_id: str, action: Action) -> bool:
        """执行单个动作
        
        Args:
            device_id: 设备 ID
            action: 动作对象
            
        Returns:
            执行是否成功
        """
        result = False
        
        for attempt in range(action.retry_count):
            try:
                if action.action_type == ActionType.TAP:
                    x = action.params.get('x', 0)
                    y = action.params.get('y', 0)
                    result = await self.click_by_coords(device_id, x, y)
                
                elif action.action_type == ActionType.SWIPE:
                    x1 = action.params.get('x1', 0)
                    y1 = action.params.get('y1', 0)
                    x2 = action.params.get('x2', 0)
                    y2 = action.params.get('y2', 0)
                    duration = action.params.get('duration', 300)
                    result = await self.swipe(device_id, x1, y1, x2, y2, duration)
                
                elif action.action_type == ActionType.INPUT:
                    text = action.params.get('text', '')
                    result = await self.input_text(device_id, text)
                
                elif action.action_type == ActionType.WAIT:
                    wait_time = action.params.get('seconds', 1)
                    await asyncio.sleep(wait_time)
                    result = True
                
                elif action.action_type == ActionType.SCREENSHOT:
                    filename = action.params.get('filename')
                    filepath = await self.screen_capture.save_screenshot(device_id, filename)
                    result = filepath is not None
                
                if result:
                    break
                    
            except Exception:
                if attempt == action.retry_count - 1:
                    return False
                await asyncio.sleep(0.5)
        
        if result and action.delay_after > 0:
            await asyncio.sleep(action.delay_after)
        
        return result
    
    async def execute_action_sequence(self, device_id: str, 
                                       actions: List[Action]) -> bool:
        """执行动作序列
        
        Args:
            device_id: 设备 ID
            actions: 动作列表
            
        Returns:
            所有动作是否都执行成功
        """
        for action in actions:
            if not await self.execute_action(device_id, action):
                return False
        return True
    
    async def scroll_down(self, device_id: str, distance: int = 300) -> bool:
        """向下滚动
        
        Args:
            device_id: 设备 ID
            distance: 滚动距离
            
        Returns:
            滚动是否成功
        """
        # 基于 540x960 分辨率
        center_x = self.SCREEN_WIDTH // 2  # 270
        start_y = int(self.SCREEN_HEIGHT * 0.7)  # 672
        end_y = start_y - distance
        return await self.swipe(device_id, center_x, start_y, center_x, end_y, 300)
    
    async def scroll_up(self, device_id: str, distance: int = 300) -> bool:
        """向上滚动
        
        Args:
            device_id: 设备 ID
            distance: 滚动距离
            
        Returns:
            滚动是否成功
        """
        center_x = self.SCREEN_WIDTH // 2  # 270
        start_y = int(self.SCREEN_HEIGHT * 0.3)  # 288
        end_y = start_y + distance
        return await self.swipe(device_id, center_x, start_y, center_x, end_y, 300)
    
    async def back(self, device_id: str) -> bool:
        """按返回键
        
        Args:
            device_id: 设备 ID
            
        Returns:
            操作是否成功
        """
        result = await self.adb_bridge.shell(device_id, "input keyevent 4")
        return result is not None
    
    async def home(self, device_id: str) -> bool:
        """按 Home 键
        
        Args:
            device_id: 设备 ID
            
        Returns:
            操作是否成功
        """
        result = await self.adb_bridge.shell(device_id, "input keyevent 3")
        return result is not None
