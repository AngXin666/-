"""
按钮位置学习器 - 从多次运行中学习可靠的按钮位置
Button Position Learner - Learn reliable button positions from multiple runs
"""

import json
import os
from typing import Optional, Tuple, List, Dict
from pathlib import Path
from collections import Counter
import statistics


class ButtonPositionLearner:
    """按钮位置学习器
    
    通过记录和分析多次成功点击的坐标，学习最可靠的按钮位置
    支持按设备分组学习，提高不同模拟器的适配精度
    """
    
    def __init__(self, device_id: Optional[str] = None, 
                 data_dir: str = "runtime_data/button_positions"):
        """初始化学习器
        
        Args:
            device_id: 设备ID（如果提供，则使用设备专属数据）
            data_dir: 数据目录路径
        """
        self.device_id = device_id
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 设备专属数据文件
        if device_id:
            self.device_file = self.data_dir / f"device_{device_id}.json"
        else:
            self.device_file = None
        
        # 全局数据文件（所有设备共享）
        self.global_file = self.data_dir / "global.json"
        
        # 加载数据
        self.device_data = self._load_data(self.device_file) if self.device_file else {}
        self.global_data = self._load_data(self.global_file)
    
    def _load_data(self, data_file: Optional[Path]) -> Dict:
        """加载历史数据
        
        Args:
            data_file: 数据文件路径
            
        Returns:
            数据字典
        """
        if data_file and data_file.exists():
            try:
                with open(data_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[ButtonLearner] ⚠️ 加载数据失败 ({data_file.name}): {e}")
                return {}
        return {}
    
    def _save_data(self, data: Dict, data_file: Path):
        """保存数据
        
        Args:
            data: 数据字典
            data_file: 数据文件路径
        """
        try:
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[ButtonLearner] ⚠️ 保存数据失败 ({data_file.name}): {e}")
    
    def record_success(self, button_name: str, position: Tuple[int, int], 
                      confidence: float = 1.0):
        """记录一次成功的点击
        
        同时记录到设备专属数据和全局数据
        
        Args:
            button_name: 按钮名称（如 "home_checkin_button"）
            position: 按钮坐标 (x, y)
            confidence: 置信度（YOLO检测的置信度，OCR为1.0）
        """
        from datetime import datetime
        timestamp = datetime.now().isoformat()
        
        record = {
            'x': position[0],
            'y': position[1],
            'confidence': confidence,
            'timestamp': timestamp
        }
        
        # 记录到设备专属数据
        if self.device_file:
            if button_name not in self.device_data:
                self.device_data[button_name] = {
                    'positions': [],
                    'total_count': 0,
                    'last_updated': None
                }
            
            self.device_data[button_name]['positions'].append(record)
            self.device_data[button_name]['total_count'] += 1
            
            # 只保留最近500次记录（设备专属数据）
            if len(self.device_data[button_name]['positions']) > 500:
                self.device_data[button_name]['positions'] = \
                    self.device_data[button_name]['positions'][-500:]
            
            self.device_data[button_name]['last_updated'] = timestamp
            self._save_data(self.device_data, self.device_file)
        
        # 记录到全局数据
        if button_name not in self.global_data:
            self.global_data[button_name] = {
                'positions': [],
                'total_count': 0,
                'last_updated': None
            }
        
        self.global_data[button_name]['positions'].append(record)
        self.global_data[button_name]['total_count'] += 1
        
        # 只保留最近1000次记录（全局数据）
        if len(self.global_data[button_name]['positions']) > 1000:
            self.global_data[button_name]['positions'] = \
                self.global_data[button_name]['positions'][-1000:]
        
        self.global_data[button_name]['last_updated'] = timestamp
        self._save_data(self.global_data, self.global_file)
    
    def get_best_position(self, button_name: str, 
                         min_samples: int = 5,
                         prefer_device: bool = True) -> Optional[Tuple[int, int]]:
        """获取统计学习的最佳位置
        
        使用加权中位数算法，置信度高的坐标权重更大
        优先使用设备专属数据，样本不足时降级到全局数据
        
        Args:
            button_name: 按钮名称
            min_samples: 最少样本数（少于此数量则尝试全局数据）
            prefer_device: 是否优先使用设备专属数据
            
        Returns:
            最佳坐标 (x, y)，如果样本不足返回None
        """
        # 优先使用设备专属数据
        if prefer_device and self.device_data and button_name in self.device_data:
            positions = self.device_data[button_name]['positions']
            if len(positions) >= min_samples:
                return self._calculate_best_position(positions)
        
        # 降级到全局数据
        if button_name in self.global_data:
            positions = self.global_data[button_name]['positions']
            if len(positions) >= min_samples:
                return self._calculate_best_position(positions)
        
        return None
    
    def _calculate_best_position(self, positions: List[Dict]) -> Tuple[int, int]:
        """从位置列表计算最佳坐标
        
        Args:
            positions: 位置记录列表
            
        Returns:
            最佳坐标 (x, y)
        """
        # 提取坐标和置信度
        x_coords = [p['x'] for p in positions]
        y_coords = [p['y'] for p in positions]
        confidences = [p['confidence'] for p in positions]
        
        # 使用加权中位数（置信度作为权重）
        best_x = self._weighted_median(x_coords, confidences)
        best_y = self._weighted_median(y_coords, confidences)
        
        return (best_x, best_y)
    
    def _weighted_median(self, values: List[float], weights: List[float]) -> int:
        """计算加权中位数
        
        Args:
            values: 数值列表
            weights: 权重列表
            
        Returns:
            加权中位数（整数）
        """
        # 按值排序
        sorted_pairs = sorted(zip(values, weights))
        
        # 计算累积权重
        total_weight = sum(weights)
        cumulative_weight = 0
        
        for value, weight in sorted_pairs:
            cumulative_weight += weight
            if cumulative_weight >= total_weight / 2:
                return int(value)
        
        # 降级到普通中位数
        return int(statistics.median(values))
    
    def get_valid_range(self, button_name: str, 
                       default_range: Tuple[int, int, int, int],
                       margin: int = 50,
                       prefer_device: bool = True) -> Tuple[int, int, int, int]:
        """获取动态调整的合理范围
        
        基于历史数据计算合理的坐标范围
        优先使用设备专属数据，样本不足时降级到全局数据
        
        Args:
            button_name: 按钮名称
            default_range: 默认范围 (x_min, x_max, y_min, y_max)
            margin: 边距（在统计范围基础上扩展的像素数）
            prefer_device: 是否优先使用设备专属数据
            
        Returns:
            合理范围 (x_min, x_max, y_min, y_max)
        """
        # 优先使用设备专属数据
        positions = None
        if prefer_device and self.device_data and button_name in self.device_data:
            positions = self.device_data[button_name]['positions']
            if len(positions) < 10:  # 设备专属样本不足，尝试全局数据
                positions = None
        
        # 降级到全局数据
        if positions is None and button_name in self.global_data:
            positions = self.global_data[button_name]['positions']
        
        # 如果仍然没有足够样本，返回默认范围
        if not positions or len(positions) < 10:
            return default_range
        
        # 提取坐标
        x_coords = [p['x'] for p in positions]
        y_coords = [p['y'] for p in positions]
        
        # 计算统计范围（去除异常值）
        # 使用sorted + index计算分位数（兼容Python 3.7）
        x_sorted = sorted(x_coords)
        y_sorted = sorted(y_coords)
        x_min = int(x_sorted[int(len(x_sorted) * 0.05)]) - margin  # 5%分位数
        x_max = int(x_sorted[int(len(x_sorted) * 0.95)]) + margin  # 95%分位数
        y_min = int(y_sorted[int(len(y_sorted) * 0.05)]) - margin
        y_max = int(y_sorted[int(len(y_sorted) * 0.95)]) + margin
        
        return (x_min, x_max, y_min, y_max)
    
    def get_statistics(self, button_name: str, 
                      prefer_device: bool = True) -> Optional[Dict]:
        """获取按钮位置的统计信息
        
        优先使用设备专属数据，样本不足时降级到全局数据
        
        Args:
            button_name: 按钮名称
            prefer_device: 是否优先使用设备专属数据
            
        Returns:
            统计信息字典，如果没有数据返回None
        """
        # 优先使用设备专属数据
        positions = None
        data_source = None
        if prefer_device and self.device_data and button_name in self.device_data:
            positions = self.device_data[button_name]['positions']
            data_source = 'device'
            if len(positions) < 5:  # 设备专属样本不足，尝试全局数据
                positions = None
        
        # 降级到全局数据
        if positions is None and button_name in self.global_data:
            positions = self.global_data[button_name]['positions']
            data_source = 'global'
        
        # 如果仍然没有数据，返回None
        if not positions:
            return None
        
        x_coords = [p['x'] for p in positions]
        y_coords = [p['y'] for p in positions]
        
        # 获取总计数和最后更新时间
        if data_source == 'device':
            total_count = self.device_data[button_name]['total_count']
            last_updated = self.device_data[button_name]['last_updated']
        else:
            total_count = self.global_data[button_name]['total_count']
            last_updated = self.global_data[button_name]['last_updated']
        
        return {
            'data_source': data_source,  # 数据来源（device/global）
            'total_count': total_count,
            'sample_count': len(positions),
            'x_mean': statistics.mean(x_coords),
            'x_median': statistics.median(x_coords),
            'x_stdev': statistics.stdev(x_coords) if len(x_coords) > 1 else 0,
            'y_mean': statistics.mean(y_coords),
            'y_median': statistics.median(y_coords),
            'y_stdev': statistics.stdev(y_coords) if len(y_coords) > 1 else 0,
            'last_updated': last_updated
        }
    
    def is_position_valid(self, button_name: str, position: Tuple[int, int],
                         tolerance: float = 2.0,
                         prefer_device: bool = True) -> bool:
        """验证坐标是否在合理范围内
        
        使用统计学方法判断坐标是否异常（基于标准差）
        优先使用设备专属数据，样本不足时降级到全局数据
        
        Args:
            button_name: 按钮名称
            position: 待验证的坐标
            tolerance: 容忍度（标准差的倍数，默认2.0表示95%置信区间）
            prefer_device: 是否优先使用设备专属数据
            
        Returns:
            是否合理
        """
        stats = self.get_statistics(button_name, prefer_device=prefer_device)
        if not stats or stats['sample_count'] < 10:
            return True  # 样本不足，无法判断
        
        x, y = position
        
        # 检查是否在合理范围内（均值 ± tolerance * 标准差）
        x_valid = abs(x - stats['x_mean']) <= tolerance * stats['x_stdev']
        y_valid = abs(y - stats['y_mean']) <= tolerance * stats['y_stdev']
        
        return x_valid and y_valid
