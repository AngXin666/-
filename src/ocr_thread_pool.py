"""
OCR 线程池管理器 - 多线程并发处理 OCR 识别（已禁用，避免 multiprocessing 问题）
OCR Thread Pool Manager - Multi-threaded concurrent OCR processing (Disabled to avoid multiprocessing issues)
"""

import asyncio
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime, timedelta

try:
    from rapidocr import RapidOCR
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


@dataclass
class OCRResult:
    """OCR 识别结果"""
    texts: List[str]
    boxes: Optional[List] = None
    scores: Optional[List] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class OCRThreadPool:
    """OCR 线程池管理器（单例模式）"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # 避免重复初始化
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        
        # OCR 实例（单例）
        self._ocr = None
        if HAS_OCR:
            import logging
            logging.getLogger("rapidocr").setLevel(logging.WARNING)
            self._ocr = RapidOCR()
        
        # 线程池配置（增加到8个线程以提升并行度）
        self._executor = ThreadPoolExecutor(
            max_workers=8,  # 从4增加到8
            thread_name_prefix="OCR-Worker"
        )
        
        # 结果缓存（最多缓存200个结果，保留60分钟 - 整个会话期间）
        self._cache: Dict[str, OCRResult] = {}
        self._cache_lock = threading.Lock()
        self._cache_max_size = 200  # 从100增加到200
        self._cache_ttl = timedelta(minutes=60)  # 从5分钟增加到60分钟
        
        # 统计信息
        self._stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_time': 0.0
        }
        self._stats_lock = threading.Lock()
        
        # 静默初始化，不输出到控制台
        from .logger import get_silent_logger
        silent_log = get_silent_logger()
        silent_log.log(f"[OCR线程池] 初始化完成，线程数: 8（GPU加速优化）")
        silent_log.log(f"[OCR优化] 缓存配置: 最大{self._cache_max_size}个结果, TTL={self._cache_ttl.total_seconds()/60:.0f}分钟")
    
    def _compute_image_hash(self, image: 'Image.Image') -> str:
        """计算图片哈希值（用于缓存）
        
        Args:
            image: PIL 图片对象
            
        Returns:
            str: 图片哈希值
        """
        # 将图片转换为字节流并计算 MD5
        import io
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
        return hashlib.md5(image_bytes).hexdigest()
    
    def _get_from_cache(self, image_hash: str) -> Optional[OCRResult]:
        """从缓存中获取结果
        
        Args:
            image_hash: 图片哈希值
            
        Returns:
            OCRResult: 缓存的结果，未找到返回 None
        """
        with self._cache_lock:
            if image_hash in self._cache:
                result = self._cache[image_hash]
                # 检查是否过期
                if datetime.now() - result.timestamp < self._cache_ttl:
                    with self._stats_lock:
                        self._stats['cache_hits'] += 1
                    return result
                else:
                    # 过期，删除
                    del self._cache[image_hash]
        
        with self._stats_lock:
            self._stats['cache_misses'] += 1
        return None
    
    def _add_to_cache(self, image_hash: str, result: OCRResult):
        """添加结果到缓存
        
        Args:
            image_hash: 图片哈希值
            result: OCR 结果
        """
        with self._cache_lock:
            # 如果缓存已满，删除最旧的条目
            if len(self._cache) >= self._cache_max_size:
                # 找到最旧的条目
                oldest_key = min(self._cache.keys(), 
                               key=lambda k: self._cache[k].timestamp)
                del self._cache[oldest_key]
            
            self._cache[image_hash] = result
    
    def _ocr_sync(self, image: 'Image.Image', use_cache: bool = True) -> OCRResult:
        """同步 OCR 识别（在线程池中执行）
        
        Args:
            image: PIL 图片对象
            use_cache: 是否使用缓存
            
        Returns:
            OCRResult: 识别结果
        """
        start_time = datetime.now()
        
        # 计算图片哈希
        image_hash = None
        if use_cache:
            image_hash = self._compute_image_hash(image)
            
            # 尝试从缓存获取
            cached_result = self._get_from_cache(image_hash)
            if cached_result is not None:
                return cached_result
        
        # 执行 OCR 识别
        if not self._ocr:
            return OCRResult(texts=[])
        
        try:
            ocr_result = self._ocr(image)
            
            if ocr_result and ocr_result.txts:
                result = OCRResult(
                    texts=list(ocr_result.txts),
                    boxes=ocr_result.boxes if hasattr(ocr_result, 'boxes') else None,
                    scores=ocr_result.scores if hasattr(ocr_result, 'scores') else None
                )
            else:
                result = OCRResult(texts=[])
            
            # 添加到缓存
            if use_cache and image_hash:
                self._add_to_cache(image_hash, result)
            
            # 更新统计
            elapsed = (datetime.now() - start_time).total_seconds()
            with self._stats_lock:
                self._stats['total_requests'] += 1
                self._stats['total_time'] += elapsed
            
            return result
            
        except Exception as e:
            print(f"  [OCR线程池] OCR 识别失败: {e}")
            return OCRResult(texts=[])
    
    async def recognize(self, image: 'Image.Image', 
                       timeout: float = 10.0,
                       use_cache: bool = True) -> OCRResult:
        """异步 OCR 识别（推荐使用）
        
        Args:
            image: PIL 图片对象
            timeout: 超时时间（秒）
            use_cache: 是否使用缓存
            
        Returns:
            OCRResult: 识别结果
        """
        if not HAS_OCR or not HAS_PIL:
            return OCRResult(texts=[])
        
        try:
            # 在线程池中执行 OCR
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    self._executor,
                    self._ocr_sync,
                    image,
                    use_cache
                ),
                timeout=timeout
            )
            return result
            
        except asyncio.TimeoutError:
            print(f"  [OCR线程池] OCR 识别超时（{timeout}秒）")
            return OCRResult(texts=[])
        except Exception as e:
            print(f"  [OCR线程池] OCR 识别异常: {e}")
            return OCRResult(texts=[])
    
    async def recognize_batch(self, images: List['Image.Image'],
                             timeout: float = 10.0,
                             use_cache: bool = True) -> List[OCRResult]:
        """批量异步 OCR 识别（并发处理）
        
        Args:
            images: PIL 图片对象列表
            timeout: 每个任务的超时时间（秒）
            use_cache: 是否使用缓存
            
        Returns:
            List[OCRResult]: 识别结果列表
        """
        if not images:
            return []
        
        # 并发执行所有 OCR 任务
        tasks = [
            self.recognize(image, timeout=timeout, use_cache=use_cache)
            for image in images
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常
        final_results = []
        for result in results:
            if isinstance(result, Exception):
                print(f"  [OCR线程池] 批量识别中的任务失败: {result}")
                final_results.append(OCRResult(texts=[]))
            else:
                final_results.append(result)
        
        return final_results
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            dict: 统计信息
                - total_requests: 总请求数
                - cache_hits: 缓存命中数
                - cache_misses: 缓存未命中数
                - cache_hit_rate: 缓存命中率
                - avg_time: 平均识别时间（秒）
                - cache_size: 当前缓存大小
        """
        with self._stats_lock:
            total = self._stats['total_requests']
            hits = self._stats['cache_hits']
            misses = self._stats['cache_misses']
            total_time = self._stats['total_time']
            
            hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0.0
            avg_time = total_time / total if total > 0 else 0.0
        
        with self._cache_lock:
            cache_size = len(self._cache)
        
        return {
            'total_requests': total,
            'cache_hits': hits,
            'cache_misses': misses,
            'cache_hit_rate': hit_rate,
            'avg_time': avg_time,
            'cache_size': cache_size
        }
    
    def print_stats(self):
        """打印统计信息"""
        stats = self.get_stats()
        print(f"\n  [OCR线程池] 统计信息:")
        print(f"  - 总请求数: {stats['total_requests']}")
        print(f"  - 缓存命中: {stats['cache_hits']}")
        print(f"  - 缓存未命中: {stats['cache_misses']}")
        print(f"  - 缓存命中率: {stats['cache_hit_rate']:.1%}")
        print(f"  - 平均识别时间: {stats['avg_time']:.2f}秒")
        print(f"  - 当前缓存大小: {stats['cache_size']}")
    
    def clear_cache(self):
        """清空缓存"""
        with self._cache_lock:
            self._cache.clear()
        print(f"  [OCR线程池] 缓存已清空")
    
    def shutdown(self):
        """关闭线程池"""
        self._executor.shutdown(wait=True)
        print(f"  [OCR线程池] 已关闭")


# 全局单例实例
_ocr_pool = None


def get_ocr_pool() -> OCRThreadPool:
    """获取全局 OCR 线程池实例
    
    Returns:
        OCRThreadPool: 全局单例实例
    """
    global _ocr_pool
    if _ocr_pool is None:
        _ocr_pool = OCRThreadPool()
    return _ocr_pool
