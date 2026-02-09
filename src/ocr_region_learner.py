"""
OCR区域学习器 - 从多次运行中学习可靠的OCR识别区域
OCR Region Learner - Learn reliable OCR regions from multiple runs
"""

import json
from typing import Optional, Tuple, List, Dict
from pathlib import Path
import statistics


class OCRRegionLearner:
    """OCR区域学习器
    
    通过记录和分析多次成功识别的区域坐标，学习最可靠的OCR识别区域
    支持按设备分组学习，提高不同模拟器的适配精度
    """
    
    def __init__(self, device_id: Optional[str] = None, 
                 data_dir: str = "runtime_data/ocr_regions"):
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
                print(f"[OCRRegionLearner] ⚠️ 加载数据失败 ({data_file.name}): {e}")
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
            print(f"[OCRRegionLearner] ⚠️ 保存数据失败 ({data_file.name}): {e}")
    
    def record_success(self, region_name: str, region: Tuple[int, int, int, int], 
                      confidence: float = 1.0, recognized_value: Optional[str] = None):
        """记录一次成功的OCR识别
        
        同时记录到设备专属数据和全局数据
        
        Args:
            region_name: 区域名称（如 "checkin_times_region", "balance_region"）
            region: 区域坐标 (x, y, width, height)
            confidence: 置信度（OCR识别的置信度）
            recognized_value: 识别到的值（可选，用于验证）
        """
        from datetime import datetime
        timestamp = datetime.now().isoformat()
        
        record = {
            'x': region[0],
            'y': region[1],
            'width': region[2],
            'height': region[3],
            'confidence': confidence,
            'timestamp': timestamp
        }
        
        if recognized_value is not None:
            record['value'] = recognized_value
        
        # 记录到设备专属数据
        if self.device_file:
            if region_name not in self.device_data:
                self.device_data[region_name] = {
                    'regions': [],
                    'total_count': 0,
                    'last_updated': None
                }
            
            self.device_data[region_name]['regions'].append(record)
            self.device_data[region_name]['total_count'] += 1
            
            # 只保留最近300次记录（设备专属数据）
            if len(self.device_data[region_name]['regions']) > 300:
                self.device_data[region_name]['regions'] = \
                    self.device_data[region_name]['regions'][-300:]
            
            self.device_data[region_name]['last_updated'] = timestamp
            self._save_data(self.device_data, self.device_file)
        
        # 记录到全局数据
        if region_name not in self.global_data:
            self.global_data[region_name] = {
                'regions': [],
                'total_count': 0,
                'last_updated': None
            }
        
        self.global_data[region_name]['regions'].append(record)
        self.global_data[region_name]['total_count'] += 1
        
        # 只保留最近500次记录（全局数据）
        if len(self.global_data[region_name]['regions']) > 500:
            self.global_data[region_name]['regions'] = \
                self.global_data[region_name]['regions'][-500:]
        
        self.global_data[region_name]['last_updated'] = timestamp
        self._save_data(self.global_data, self.global_file)
    
    def get_best_region(self, region_name: str, 
                       min_samples: int = 5,
                       prefer_device: bool = True) -> Optional[Tuple[int, int, int, int]]:
        """获取统计学习的最佳区域
        
        使用加权中位数算法，置信度高的区域权重更大
        优先使用设备专属数据，样本不足时降级到全局数据
        
        Args:
            region_name: 区域名称
            min_samples: 最少样本数（少于此数量则尝试全局数据）
            prefer_device: 是否优先使用设备专属数据
            
        Returns:
            最佳区域 (x, y, width, height)，如果样本不足返回None
        """
        # 优先使用设备专属数据
        if prefer_device and self.device_data and region_name in self.device_data:
            regions = self.device_data[region_name]['regions']
            if len(regions) >= min_samples:
                return self._calculate_best_region(regions)
        
        # 降级到全局数据
        if region_name in self.global_data:
            regions = self.global_data[region_name]['regions']
            if len(regions) >= min_samples:
                return self._calculate_best_region(regions)
        
        return None
    
    def _calculate_best_region(self, regions: List[Dict]) -> Tuple[int, int, int, int]:
        """从区域列表计算最佳区域
        
        Args:
            regions: 区域记录列表
            
        Returns:
            最佳区域 (x, y, width, height)
        """
        # 提取坐标和置信度
        x_coords = [r['x'] for r in regions]
        y_coords = [r['y'] for r in regions]
        widths = [r['width'] for r in regions]
        heights = [r['height'] for r in regions]
        confidences = [r['confidence'] for r in regions]
        
        # 使用加权中位数（置信度作为权重）
        best_x = self._weighted_median(x_coords, confidences)
        best_y = self._weighted_median(y_coords, confidences)
        best_width = self._weighted_median(widths, confidences)
        best_height = self._weighted_median(heights, confidences)
        
        return (best_x, best_y, best_width, best_height)
    
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
    
    def get_valid_range(self, region_name: str, 
                       default_range: Tuple[int, int, int, int, int, int, int, int],
                       margin: int = 20,
                       prefer_device: bool = True) -> Tuple[int, int, int, int, int, int, int, int]:
        """获取动态调整的合理范围
        
        基于历史数据计算合理的区域范围
        优先使用设备专属数据，样本不足时降级到全局数据
        
        Args:
            region_name: 区域名称
            default_range: 默认范围 (x_min, x_max, y_min, y_max, w_min, w_max, h_min, h_max)
            margin: 边距（在统计范围基础上扩展的像素数）
            prefer_device: 是否优先使用设备专属数据
            
        Returns:
            合理范围 (x_min, x_max, y_min, y_max, w_min, w_max, h_min, h_max)
        """
        # 优先使用设备专属数据
        regions = None
        if prefer_device and self.device_data and region_name in self.device_data:
            regions = self.device_data[region_name]['regions']
            if len(regions) < 10:  # 设备专属样本不足，尝试全局数据
                regions = None
        
        # 降级到全局数据
        if regions is None and region_name in self.global_data:
            regions = self.global_data[region_name]['regions']
        
        # 如果仍然没有足够样本，返回默认范围
        if not regions or len(regions) < 10:
            return default_range
        
        # 提取坐标
        x_coords = [r['x'] for r in regions]
        y_coords = [r['y'] for r in regions]
        widths = [r['width'] for r in regions]
        heights = [r['height'] for r in regions]
        
        # 计算统计范围（去除异常值）
        # 使用sorted + index计算分位数（兼容Python 3.7）
        x_sorted = sorted(x_coords)
        y_sorted = sorted(y_coords)
        w_sorted = sorted(widths)
        h_sorted = sorted(heights)
        x_min = int(x_sorted[int(len(x_sorted) * 0.05)]) - margin
        x_max = int(x_sorted[int(len(x_sorted) * 0.95)]) + margin
        y_min = int(y_sorted[int(len(y_sorted) * 0.05)]) - margin
        y_max = int(y_sorted[int(len(y_sorted) * 0.95)]) + margin
        w_min = int(w_sorted[int(len(w_sorted) * 0.05)]) - margin
        w_max = int(w_sorted[int(len(w_sorted) * 0.95)]) + margin
        h_min = int(h_sorted[int(len(h_sorted) * 0.05)]) - margin
        h_max = int(h_sorted[int(len(h_sorted) * 0.95)]) + margin
        
        return (x_min, x_max, y_min, y_max, w_min, w_max, h_min, h_max)
    
    def get_statistics(self, region_name: str, 
                      prefer_device: bool = True) -> Optional[Dict]:
        """获取区域的统计信息
        
        优先使用设备专属数据，样本不足时降级到全局数据
        
        Args:
            region_name: 区域名称
            prefer_device: 是否优先使用设备专属数据
            
        Returns:
            统计信息字典，如果没有数据返回None
        """
        # 优先使用设备专属数据
        regions = None
        data_source = None
        if prefer_device and self.device_data and region_name in self.device_data:
            regions = self.device_data[region_name]['regions']
            data_source = 'device'
            if len(regions) < 5:  # 设备专属样本不足，尝试全局数据
                regions = None
        
        # 降级到全局数据
        if regions is None and region_name in self.global_data:
            regions = self.global_data[region_name]['regions']
            data_source = 'global'
        
        # 如果仍然没有数据，返回None
        if not regions:
            return None
        
        x_coords = [r['x'] for r in regions]
        y_coords = [r['y'] for r in regions]
        widths = [r['width'] for r in regions]
        heights = [r['height'] for r in regions]
        
        # 获取总计数和最后更新时间
        if data_source == 'device':
            total_count = self.device_data[region_name]['total_count']
            last_updated = self.device_data[region_name]['last_updated']
        else:
            total_count = self.global_data[region_name]['total_count']
            last_updated = self.global_data[region_name]['last_updated']
        
        return {
            'data_source': data_source,  # 数据来源（device/global）
            'total_count': total_count,
            'sample_count': len(regions),
            'x_mean': statistics.mean(x_coords),
            'x_median': statistics.median(x_coords),
            'x_stdev': statistics.stdev(x_coords) if len(x_coords) > 1 else 0,
            'y_mean': statistics.mean(y_coords),
            'y_median': statistics.median(y_coords),
            'y_stdev': statistics.stdev(y_coords) if len(y_coords) > 1 else 0,
            'width_mean': statistics.mean(widths),
            'width_median': statistics.median(widths),
            'width_stdev': statistics.stdev(widths) if len(widths) > 1 else 0,
            'height_mean': statistics.mean(heights),
            'height_median': statistics.median(heights),
            'height_stdev': statistics.stdev(heights) if len(heights) > 1 else 0,
            'last_updated': last_updated
        }
    
    def is_region_valid(self, region_name: str, region: Tuple[int, int, int, int],
                       tolerance: float = 2.0,
                       prefer_device: bool = True) -> bool:
        """验证区域是否在合理范围内
        
        使用统计学方法判断区域是否异常（基于标准差）
        优先使用设备专属数据，样本不足时降级到全局数据
        
        Args:
            region_name: 区域名称
            region: 待验证的区域 (x, y, width, height)
            tolerance: 容忍度（标准差的倍数，默认2.0表示95%置信区间）
            prefer_device: 是否优先使用设备专属数据
            
        Returns:
            是否合理
        """
        stats = self.get_statistics(region_name, prefer_device=prefer_device)
        if not stats or stats['sample_count'] < 10:
            return True  # 样本不足，无法判断
        
        x, y, width, height = region
        
        # 检查是否在合理范围内（均值 ± tolerance * 标准差）
        x_valid = abs(x - stats['x_mean']) <= tolerance * stats['x_stdev']
        y_valid = abs(y - stats['y_mean']) <= tolerance * stats['y_stdev']
        width_valid = abs(width - stats['width_mean']) <= tolerance * stats['width_stdev']
        height_valid = abs(height - stats['height_mean']) <= tolerance * stats['height_stdev']
        
        return x_valid and y_valid and width_valid and height_valid
