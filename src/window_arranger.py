"""
窗口自动排列模块
Window Auto Arrangement Module

功能：
1. 自动排列多个模拟器窗口
2. 支持多种预设布局方案
3. 用户可以保存自定义排列方案
4. 自动检测屏幕分辨率并调整
"""

import json
import win32gui
import win32con
import win32api
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict


@dataclass
class WindowLayout:
    """窗口布局配置"""
    name: str                    # 布局名称
    rows: int                    # 行数
    cols: int                    # 列数
    window_width: int            # 窗口宽度
    window_height: int           # 窗口高度
    gap_x: int = 0              # 水平间距
    gap_y: int = 0              # 垂直间距
    start_x: int = 0            # 起始X坐标
    start_y: int = 0            # 起始Y坐标
    description: str = ""        # 布局描述


class WindowArranger:
    """窗口自动排列器"""
    
    # 预设布局方案
    PRESET_LAYOUTS = {
        "1x1": WindowLayout(
            name="单窗口",
            rows=1, cols=1,
            window_width=400, window_height=720,
            description="单个窗口居中显示"
        ),
        "2x1": WindowLayout(
            name="横向双窗口",
            rows=1, cols=2,
            window_width=400, window_height=720,
            gap_x=10,
            description="两个窗口横向排列"
        ),
        "3x1": WindowLayout(
            name="横向三窗口",
            rows=1, cols=3,
            window_width=400, window_height=720,
            gap_x=10,
            description="三个窗口横向排列"
        ),
        "4x1": WindowLayout(
            name="横向四窗口",
            rows=1, cols=4,
            window_width=360, window_height=640,
            gap_x=10,
            description="四个窗口横向排列"
        ),
        "2x2": WindowLayout(
            name="2x2网格",
            rows=2, cols=2,
            window_width=400, window_height=600,
            gap_x=10, gap_y=10,
            description="四个窗口2x2网格排列"
        ),
        "3x2": WindowLayout(
            name="3x2网格",
            rows=2, cols=3,
            window_width=360, window_height=540,
            gap_x=10, gap_y=10,
            description="六个窗口3x2网格排列"
        ),
        "1x2": WindowLayout(
            name="纵向双窗口",
            rows=2, cols=1,
            window_width=400, window_height=500,
            gap_y=10,
            description="两个窗口纵向排列"
        ),
    }
    
    def __init__(self, config_dir: Path = None):
        """初始化窗口排列器
        
        Args:
            config_dir: 配置文件目录，默认为 .kiro/settings/
        """
        if config_dir is None:
            config_dir = Path(".kiro/settings")
        
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.config_dir / "window_layouts.json"
        
        # 加载用户自定义布局
        self.custom_layouts = self._load_custom_layouts()
    
    def get_screen_size(self) -> Tuple[int, int]:
        """获取屏幕分辨率
        
        Returns:
            (宽度, 高度)
        """
        width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
        return width, height
    
    def get_work_area(self) -> Tuple[int, int, int, int]:
        """获取工作区域（排除任务栏）
        
        Returns:
            (left, top, right, bottom)
        """
        # 获取工作区域（排除任务栏）
        work_area = win32api.GetMonitorInfo(win32api.MonitorFromPoint((0, 0)))['Work']
        return work_area
    
    def find_emulator_windows(self, title_keywords: List[str] = None) -> List[int]:
        """查找模拟器窗口
        
        Args:
            title_keywords: 窗口标题关键词列表，默认为 ["MuMu", "雷电", "夜神", "逍遥"]
            
        Returns:
            窗口句柄列表
        """
        if title_keywords is None:
            title_keywords = ["MuMu", "雷电", "夜神", "逍遥", "BlueStacks"]
        
        windows = []
        
        def enum_callback(hwnd, results):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                # 检查标题是否包含关键词
                for keyword in title_keywords:
                    if keyword in title:
                        # 过滤掉模拟器管理器窗口
                        # MuMu模拟器管理器的标题通常是"MuMu模拟器"，尺寸很小（160x28）
                        # 真正的模拟器窗口标题是"MuMu安卓设备"或"MuMu安卓设备-X"
                        
                        # 排除包含"模拟器"但不包含"设备"的窗口（管理器）
                        if "模拟器" in title and "设备" not in title:
                            continue
                        
                        # 获取窗口尺寸，过滤掉太小的窗口（管理器通常很小）
                        try:
                            rect = win32gui.GetWindowRect(hwnd)
                            width = rect[2] - rect[0]
                            height = rect[3] - rect[1]
                            
                            # 过滤掉宽度或高度小于200的窗口（管理器窗口）
                            if width < 200 or height < 200:
                                continue
                        except:
                            pass
                        
                        results.append(hwnd)
                        break
        
        win32gui.EnumWindows(enum_callback, windows)
        return windows
    
    def get_window_info(self, hwnd: int) -> Dict:
        """获取窗口信息
        
        Args:
            hwnd: 窗口句柄
            
        Returns:
            窗口信息字典
        """
        try:
            rect = win32gui.GetWindowRect(hwnd)
            title = win32gui.GetWindowText(hwnd)
            
            return {
                'hwnd': hwnd,
                'title': title,
                'x': rect[0],
                'y': rect[1],
                'width': rect[2] - rect[0],
                'height': rect[3] - rect[1]
            }
        except Exception as e:
            return None
    
    def move_window(self, hwnd: int, x: int, y: int, width: int, height: int) -> bool:
        """移动和调整窗口大小
        
        Args:
            hwnd: 窗口句柄
            x: X坐标
            y: Y坐标
            width: 宽度
            height: 高度
            
        Returns:
            是否成功
        """
        try:
            # 先显示窗口（如果最小化）
            if win32gui.IsIconic(hwnd):
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            
            # 移动和调整大小
            win32gui.MoveWindow(hwnd, x, y, width, height, True)
            
            # 置顶窗口（可选）
            # win32gui.SetWindowPos(hwnd, win32con.HWND_TOP, x, y, width, height, 0)
            
            return True
        except Exception as e:
            print(f"移动窗口失败: {e}")
            return False
    
    def minimize_other_windows(self, keep_windows: List[int] = None):
        """最小化除了指定窗口之外的所有窗口
        
        Args:
            keep_windows: 要保留的窗口句柄列表（不最小化这些窗口）
        """
        if keep_windows is None:
            keep_windows = []
        
        # 获取所有模拟器窗口
        emulator_windows = self.find_emulator_windows()
        
        # 合并要保留的窗口列表
        keep_windows_set = set(keep_windows + emulator_windows)
        
        def enum_callback(hwnd, results):
            # 跳过不可见的窗口
            if not win32gui.IsWindowVisible(hwnd):
                return
            
            # 跳过要保留的窗口
            if hwnd in keep_windows_set:
                return
            
            # 跳过已经最小化的窗口
            if win32gui.IsIconic(hwnd):
                return
            
            # 获取窗口标题
            title = win32gui.GetWindowText(hwnd)
            
            # 跳过没有标题的窗口（通常是系统窗口）
            if not title:
                return
            
            # 跳过特殊窗口
            if title in ["Program Manager", "Windows Shell Experience Host"]:
                return
            
            try:
                # 最小化窗口
                win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
            except:
                pass
        
        win32gui.EnumWindows(enum_callback, None)
    
    def arrange_windows(self, layout_name: str, window_count: int = None, keep_size: bool = True, overlap_percent: int = 50, minimize_others: bool = True, dialog_hwnd: int = None) -> Dict:
        """自动排列窗口
        
        Args:
            layout_name: 布局名称（如 "tile", "stack", "2x1", "3x1", "2x2"）
            window_count: 窗口数量，None表示自动检测
            keep_size: 是否保持窗口原有尺寸（默认True，只移动位置不改变大小）
            minimize_others: 是否最小化其他窗口（默认True）
            dialog_hwnd: 对话框窗口句柄（如果提供，将不会被最小化）
            
        Returns:
            结果字典：
            - success: 是否成功
            - arranged_count: 已排列窗口数
            - total_count: 总窗口数
            - message: 消息
        """
        # 最小化其他窗口（如果启用）
        if minimize_others:
            keep_windows = [dialog_hwnd] if dialog_hwnd else []
            self.minimize_other_windows(keep_windows)
        
        # 查找模拟器窗口
        windows = self.find_emulator_windows()
        if not windows:
            return {
                'success': False,
                'arranged_count': 0,
                'total_count': 0,
                'message': "未找到模拟器窗口"
            }
        
        # 确定要排列的窗口数量
        if window_count:
            windows = windows[:min(window_count, len(windows))]
        
        # 特殊布局：平铺和堆叠
        if layout_name == "tile":
            return self._arrange_tile(windows, keep_size)
        elif layout_name == "stack":
            return self._arrange_stack(windows, keep_size, overlap_percent)
        
        # 获取布局配置
        layout = self.get_layout(layout_name)
        if not layout:
            return {
                'success': False,
                'arranged_count': 0,
                'total_count': 0,
                'message': f"布局方案不存在: {layout_name}"
            }
        
        # 限制窗口数量
        max_windows = layout.rows * layout.cols
        windows = windows[:min(len(windows), max_windows)]
        
        # 获取屏幕工作区域
        work_area = self.get_work_area()
        work_width = work_area[2] - work_area[0]
        work_height = work_area[3] - work_area[1]
        
        # 如果保持原有尺寸，使用第一个窗口的尺寸
        if keep_size and windows:
            first_window_info = self.get_window_info(windows[0])
            if first_window_info:
                window_width = first_window_info['width']
                window_height = first_window_info['height']
            else:
                window_width = layout.window_width
                window_height = layout.window_height
        else:
            window_width = layout.window_width
            window_height = layout.window_height
        
        # 计算总布局大小
        total_width = layout.cols * window_width + (layout.cols - 1) * layout.gap_x
        total_height = layout.rows * window_height + (layout.rows - 1) * layout.gap_y
        
        # 计算起始位置（始终居中，不使用保存的绝对坐标）
        # 这样可以避免窗口移出屏幕
        start_x = work_area[0] + (work_width - total_width) // 2
        start_y = work_area[1] + (work_height - total_height) // 2
        
        # 确保起始位置不会超出屏幕
        start_x = max(work_area[0], min(start_x, work_area[2] - total_width))
        start_y = max(work_area[1], min(start_y, work_area[3] - total_height))
        
        # 排列窗口
        arranged_count = 0
        for i, hwnd in enumerate(windows):
            row = i // layout.cols
            col = i % layout.cols
            
            x = start_x + col * (window_width + layout.gap_x)
            y = start_y + row * (window_height + layout.gap_y)
            
            if keep_size:
                # 只移动位置，不改变大小
                if self.move_window_position_only(hwnd, x, y):
                    arranged_count += 1
            else:
                # 移动位置并调整大小
                if self.move_window(hwnd, x, y, window_width, window_height):
                    arranged_count += 1
        
        return {
            'success': arranged_count > 0,
            'arranged_count': arranged_count,
            'total_count': len(windows),
            'message': f"成功排列 {arranged_count}/{len(windows)} 个窗口"
        }
    
    def _arrange_tile(self, windows: List[int], keep_size: bool = True) -> Dict:
        """平铺排列窗口（横向排列，自动计算重叠量）
        
        窗口横向排列，自动计算重叠量以适应屏幕宽度。
        目的是让用户能看清楚每个窗口的更多内容，重叠量尽可能小。
        
        Args:
            windows: 窗口句柄列表
            keep_size: 是否保持窗口原有尺寸
            
        Returns:
            结果字典
        """
        if not windows:
            return {
                'success': False,
                'arranged_count': 0,
                'total_count': 0,
                'message': "没有窗口需要排列"
            }
        
        # 获取屏幕工作区域
        work_area = self.get_work_area()
        work_width = work_area[2] - work_area[0]
        work_height = work_area[3] - work_area[1]
        
        # 获取第一个窗口的尺寸作为参考
        first_window_info = self.get_window_info(windows[0])
        if not first_window_info:
            return {
                'success': False,
                'arranged_count': 0,
                'total_count': len(windows),
                'message': "无法获取窗口信息"
            }
        
        window_width = first_window_info['width']
        window_height = first_window_info['height']
        
        # 计算理想情况下的总宽度（不重叠）
        ideal_total_width = len(windows) * window_width
        
        # 如果理想宽度小于屏幕宽度，不需要重叠，平均分配间距
        if ideal_total_width <= work_width:
            # 计算间距
            available_gap_space = work_width - ideal_total_width
            gap = available_gap_space // (len(windows) + 1) if len(windows) > 0 else 0
            
            # 计算起始X坐标
            start_x = work_area[0] + gap
            
            # 计算Y坐标（垂直居中）
            start_y = work_area[1] + (work_height - window_height) // 2
            
            # 排列窗口（不重叠）
            arranged_count = 0
            for i, hwnd in enumerate(windows):
                x = start_x + i * (window_width + gap)
                y = start_y
                
                if keep_size:
                    if self.move_window_position_only(hwnd, x, y):
                        arranged_count += 1
                else:
                    if self.move_window(hwnd, x, y, window_width, window_height):
                        arranged_count += 1
            
            return {
                'success': arranged_count > 0,
                'arranged_count': arranged_count,
                'total_count': len(windows),
                'message': f"平铺排列 {arranged_count}/{len(windows)} 个窗口（不重叠，间距{gap}px）"
            }
        
        # 如果理想宽度大于屏幕宽度，需要重叠
        # 计算需要的总宽度（第一个窗口完整显示 + 其他窗口的偏移量）
        # 公式：total_width = window_width + (n-1) * offset
        # 我们希望：total_width <= work_width
        # 所以：offset <= (work_width - window_width) / (n-1)
        
        if len(windows) == 1:
            offset = 0
        else:
            # 计算最大偏移量（留一些边距）
            margin = 50  # 左右各留50px边距
            available_width = work_width - 2 * margin
            max_offset = (available_width - window_width) // (len(windows) - 1)
            
            # 确保偏移量至少是窗口宽度的20%（最多重叠80%）
            min_offset = window_width // 5
            offset = max(min_offset, max_offset)
            
            # 如果偏移量太小，说明窗口太多
            if offset < min_offset:
                return {
                    'success': False,
                    'arranged_count': 0,
                    'total_count': len(windows),
                    'message': f"窗口太多，无法在屏幕内平铺（需要至少{len(windows) * min_offset + window_width}px宽度）"
                }
        
        # 计算实际总宽度
        total_width = window_width + (len(windows) - 1) * offset
        
        # 计算起始X坐标（居中）
        start_x = work_area[0] + (work_width - total_width) // 2
        
        # 计算Y坐标（垂直居中）
        start_y = work_area[1] + (work_height - window_height) // 2
        
        # 排列窗口（有重叠）
        arranged_count = 0
        overlap_amount = window_width - offset
        overlap_percent = (overlap_amount * 100) // window_width
        
        for i, hwnd in enumerate(windows):
            x = start_x + i * offset
            y = start_y
            
            if keep_size:
                if self.move_window_position_only(hwnd, x, y):
                    arranged_count += 1
            else:
                if self.move_window(hwnd, x, y, window_width, window_height):
                    arranged_count += 1
        
        return {
            'success': arranged_count > 0,
            'arranged_count': arranged_count,
            'total_count': len(windows),
            'message': f"平铺排列 {arranged_count}/{len(windows)} 个窗口（重叠{overlap_percent}%，偏移{offset}px）"
        }
    
    def _arrange_stack(self, windows: List[int], keep_size: bool = True, overlap_percent: int = 50) -> Dict:
        """堆叠排列窗口（重叠，可自定义覆盖率）
        
        Args:
            windows: 窗口句柄列表
            keep_size: 是否保持窗口原有尺寸
            overlap_percent: 覆盖率百分比（10-90），默认50表示对半重叠
            
        Returns:
            结果字典
        """
        if not windows:
            return {
                'success': False,
                'arranged_count': 0,
                'total_count': 0,
                'message': "没有窗口需要排列"
            }
        
        # 获取屏幕工作区域
        work_area = self.get_work_area()
        work_width = work_area[2] - work_area[0]
        work_height = work_area[3] - work_area[1]
        
        # 获取第一个窗口的尺寸作为参考
        first_window_info = self.get_window_info(windows[0])
        if not first_window_info:
            return {
                'success': False,
                'arranged_count': 0,
                'total_count': len(windows),
                'message': "无法获取窗口信息"
            }
        
        window_width = first_window_info['width']
        window_height = first_window_info['height']
        
        # 根据覆盖率计算偏移量
        # 覆盖率50% = 对半重叠 = 偏移量为窗口宽度的50%
        # 覆盖率越大，偏移量越小，重叠越多
        offset = window_width * (100 - overlap_percent) // 100
        
        # 计算总宽度（第一个窗口的宽度 + 其他窗口的偏移量）
        total_width = window_width + (len(windows) - 1) * offset
        
        # 计算起始X坐标（靠左，留一些边距）
        margin = 50  # 左边距
        start_x = work_area[0] + margin
        
        # 如果总宽度超过屏幕宽度，调整起始位置
        if start_x + total_width > work_area[2]:
            # 居中显示
            start_x = work_area[0] + (work_width - total_width) // 2
        
        # 计算Y坐标（垂直居中）
        start_y = work_area[1] + (work_height - window_height) // 2
        
        # 排列窗口（堆叠排列）
        arranged_count = 0
        for i, hwnd in enumerate(windows):
            x = start_x + i * offset
            y = start_y
            
            if keep_size:
                if self.move_window_position_only(hwnd, x, y):
                    arranged_count += 1
            else:
                if self.move_window(hwnd, x, y, window_width, window_height):
                    arranged_count += 1
        
        return {
            'success': arranged_count > 0,
            'arranged_count': arranged_count,
            'total_count': len(windows),
            'message': f"堆叠排列 {arranged_count}/{len(windows)} 个窗口（覆盖率{overlap_percent}%）"
        }
    
    def move_window_position_only(self, hwnd: int, x: int, y: int) -> bool:
        """只移动窗口位置，不改变大小
        
        Args:
            hwnd: 窗口句柄
            x: X坐标
            y: Y坐标
            
        Returns:
            是否成功
        """
        try:
            # 先显示窗口（如果最小化）
            if win32gui.IsIconic(hwnd):
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            
            # 获取当前窗口大小
            rect = win32gui.GetWindowRect(hwnd)
            current_width = rect[2] - rect[0]
            current_height = rect[3] - rect[1]
            
            # 只移动位置，保持原有大小
            win32gui.MoveWindow(hwnd, x, y, current_width, current_height, True)
            
            return True
        except Exception as e:
            print(f"移动窗口失败: {e}")
            return False
    
    def get_layout(self, layout_name: str) -> Optional[WindowLayout]:
        """获取布局配置
        
        Args:
            layout_name: 布局名称
            
        Returns:
            布局配置，如果不存在返回None
        """
        # 先查找预设布局
        if layout_name in self.PRESET_LAYOUTS:
            return self.PRESET_LAYOUTS[layout_name]
        
        # 再查找自定义布局
        if layout_name in self.custom_layouts:
            return self.custom_layouts[layout_name]
        
        return None
    
    def get_all_layouts(self) -> Dict[str, WindowLayout]:
        """获取所有布局（预设+自定义）
        
        Returns:
            布局字典
        """
        all_layouts = {}
        all_layouts.update(self.PRESET_LAYOUTS)
        all_layouts.update(self.custom_layouts)
        return all_layouts
    
    def save_custom_layout(self, layout: WindowLayout) -> bool:
        """保存自定义布局
        
        Args:
            layout: 布局配置
            
        Returns:
            是否成功
        """
        try:
            self.custom_layouts[layout.name] = layout
            self._save_custom_layouts()
            return True
        except Exception as e:
            print(f"保存布局失败: {e}")
            return False
    
    def delete_custom_layout(self, layout_name: str) -> bool:
        """删除自定义布局
        
        Args:
            layout_name: 布局名称
            
        Returns:
            是否成功
        """
        if layout_name in self.custom_layouts:
            del self.custom_layouts[layout_name]
            self._save_custom_layouts()
            return True
        return False
    
    def create_layout_from_current(self, name: str, description: str = "") -> WindowLayout:
        """从当前窗口位置创建布局
        
        Args:
            name: 布局名称
            description: 布局描述
            
        Returns:
            新创建的布局
        """
        windows = self.find_emulator_windows()
        if not windows:
            raise ValueError("未找到模拟器窗口")
        
        # 获取所有窗口信息
        window_infos = [self.get_window_info(hwnd) for hwnd in windows]
        window_infos = [w for w in window_infos if w]
        
        if not window_infos:
            raise ValueError("无法获取窗口信息")
        
        # 按X坐标排序（从左到右）
        sorted_by_x = sorted(window_infos, key=lambda w: w['x'])
        
        # 计算平均窗口大小
        avg_width = int(sum(w['width'] for w in window_infos) / len(window_infos))
        avg_height = int(sum(w['height'] for w in window_infos) / len(window_infos))
        
        # 计算窗口之间的偏移量（用于判断是平铺还是堆叠）
        if len(sorted_by_x) > 1:
            # 计算第一个和第二个窗口之间的X偏移
            offset_x = sorted_by_x[1]['x'] - sorted_by_x[0]['x']
            
            # 如果偏移量小于窗口宽度，说明是堆叠布局
            # 否则是平铺布局（可能有间距或少量重叠）
            gap_x = offset_x
        else:
            gap_x = 0
        
        # 对于横向排列，rows=1, cols=窗口数量
        rows = 1
        cols = len(window_infos)
        gap_y = 0
        
        # 创建布局
        layout = WindowLayout(
            name=name,
            rows=rows,
            cols=cols,
            window_width=avg_width,
            window_height=avg_height,
            gap_x=gap_x,  # 保存实际的偏移量（可能是负数，表示重叠）
            gap_y=gap_y,
            start_x=min(w['x'] for w in window_infos),
            start_y=min(w['y'] for w in window_infos),
            description=description or f"横向{cols}窗口布局"
        )
        
        return layout
    
    def _load_custom_layouts(self) -> Dict[str, WindowLayout]:
        """加载自定义布局"""
        if not self.config_file.exists():
            return {}
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            layouts = {}
            for name, layout_dict in data.items():
                layouts[name] = WindowLayout(**layout_dict)
            
            return layouts
        except Exception as e:
            print(f"加载自定义布局失败: {e}")
            return {}
    
    def _save_custom_layouts(self):
        """保存自定义布局"""
        try:
            data = {}
            for name, layout in self.custom_layouts.items():
                data[name] = asdict(layout)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存自定义布局失败: {e}")


def test_window_arranger():
    """测试窗口排列器"""
    arranger = WindowArranger()
    
    print("=" * 60)
    print("窗口自动排列测试")
    print("=" * 60)
    
    # 获取屏幕信息
    screen_width, screen_height = arranger.get_screen_size()
    print(f"\n屏幕分辨率: {screen_width}x{screen_height}")
    
    work_area = arranger.get_work_area()
    print(f"工作区域: {work_area}")
    
    # 查找窗口
    windows = arranger.find_emulator_windows()
    print(f"\n找到 {len(windows)} 个模拟器窗口:")
    
    for i, hwnd in enumerate(windows, 1):
        info = arranger.get_window_info(hwnd)
        if info:
            print(f"  {i}. {info['title']}")
            print(f"     位置: ({info['x']}, {info['y']})")
            print(f"     大小: {info['width']}x{info['height']}")
    
    # 显示可用布局
    print(f"\n可用布局:")
    for name, layout in arranger.get_all_layouts().items():
        print(f"  - {name}: {layout.description}")
    
    # 测试排列
    if windows:
        print(f"\n测试排列（2x1布局）...")
        result = arranger.arrange_windows("2x1")
        print(f"结果: {result['message']}")


if __name__ == '__main__':
    test_window_arranger()
