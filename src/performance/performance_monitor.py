"""
性能监控器
Performance Monitor
"""

import time
from typing import List, Tuple, Dict, Any, Optional


class PerformanceMonitor:
    """性能监控器
    
    用于记录操作的各个步骤耗时，并生成性能摘要。
    """
    
    def __init__(self, name: str):
        """初始化性能监控器
        
        Args:
            name: 操作名称（如"启动流程"、"导航到个人页"）
        """
        self.name = name
        self._steps: List[Tuple[str, float, str]] = []  # (步骤名, 耗时, 检测方式)
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
    
    def start(self):
        """开始监控"""
        self._start_time = time.time()
        self._steps = []
    
    def record_step(self, step_name: str, duration: float, method: str = ""):
        """记录步骤耗时
        
        Args:
            step_name: 步骤名称
            duration: 耗时（秒）
            method: 检测方式（template/ocr/hybrid/其他）
        """
        self._steps.append((step_name, duration, method))
    
    def end(self) -> Dict[str, Any]:
        """结束监控，返回性能摘要
        
        Returns:
            性能摘要字典
        """
        self._end_time = time.time()
        return self.get_summary()
    
    def get_summary(self) -> Dict[str, Any]:
        """获取性能摘要
        
        Returns:
            包含总耗时、步骤详情、统计信息的字典
        """
        if self._start_time is None:
            return {
                'operation_name': self.name,
                'error': '监控未启动'
            }
        
        total_time = (self._end_time or time.time()) - self._start_time
        
        # 统计检测方式
        template_count = sum(1 for _, _, method in self._steps if method == 'template')
        ocr_count = sum(1 for _, _, method in self._steps if method == 'ocr')
        hybrid_count = sum(1 for _, _, method in self._steps if method == 'hybrid')
        cache_hit_count = sum(1 for _, _, method in self._steps if method == 'cache')
        
        # 计算平均步骤耗时
        step_times = [duration for _, duration, _ in self._steps]
        avg_step_time = sum(step_times) / len(step_times) if step_times else 0
        
        # 构建步骤详情
        steps_detail = [
            {
                'name': name,
                'duration': round(duration, 3),
                'method': method
            }
            for name, duration, method in self._steps
        ]
        
        return {
            'operation_name': self.name,
            'total_time': round(total_time, 3),
            'steps': steps_detail,
            'step_count': len(self._steps),
            'template_count': template_count,
            'ocr_count': ocr_count,
            'hybrid_count': hybrid_count,
            'cache_hit_count': cache_hit_count,
            'avg_step_time': round(avg_step_time, 3)
        }
    
    def log_summary(self, log_callback):
        """输出性能摘要到日志
        
        Args:
            log_callback: 日志回调函数
        """
        summary = self.get_summary()
        
        if 'error' in summary:
            log_callback(f"⚠️ 性能监控错误: {summary['error']}")
            return
        
        log_callback(f"")
        log_callback(f"{'='*60}")
        log_callback(f"📊 性能摘要: {summary['operation_name']}")
        log_callback(f"{'='*60}")
        log_callback(f"⏱️  总耗时: {summary['total_time']:.3f} 秒")
        log_callback(f"📝 步骤数: {summary['step_count']}")
        log_callback(f"⚡ 模板匹配: {summary['template_count']} 次")
        log_callback(f"🔍 OCR识别: {summary['ocr_count']} 次")
        log_callback(f"🔄 混合检测: {summary['hybrid_count']} 次")
        log_callback(f"💾 缓存命中: {summary['cache_hit_count']} 次")
        log_callback(f"📈 平均步骤耗时: {summary['avg_step_time']:.3f} 秒")
        
        # 输出最慢的3个步骤
        if summary['steps']:
            sorted_steps = sorted(summary['steps'], key=lambda x: x['duration'], reverse=True)
            log_callback(f"")
            log_callback(f"🐌 最慢的步骤:")
            for i, step in enumerate(sorted_steps[:3], 1):
                method_str = f" [{step['method']}]" if step['method'] else ""
                log_callback(f"  {i}. {step['name']}: {step['duration']:.3f}秒{method_str}")
        
        log_callback(f"{'='*60}")
        log_callback(f"")
