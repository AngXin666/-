"""
优化的混合页面检测模块 - 支持异步预加载
Optimized Hybrid Page Detection Module - with async preloading
"""

import asyncio
from typing import Optional
from .page_detector_hybrid import PageDetectorHybrid, PageState, PageDetectionResult


class PageDetectorHybridOptimized(PageDetectorHybrid):
    """优化的混合页面检测器
    
    新增功能：
    1. 异步预加载：在操作前提前开始页面识别
    2. 自动使用深度学习检测器（如果可用）
    3. 智能缓存管理
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._preload_task = None
        self._preload_device_id = None
        
        # 如果有深度学习检测器，尝试启用GPU加速
        if self._dl_detector:
            self._enable_gpu_if_available()
    
    def _enable_gpu_if_available(self):
        """尝试启用GPU加速"""
        try:
            import tensorflow as tf
            
            # 检查GPU是否可用
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    # 设置GPU内存增长
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    
                    # 静默启用GPU
                except RuntimeError as e:
                    # 静默记录错误
                    pass
            else:
                # 静默记录：未检测到GPU
                pass
        except Exception as e:
            # 静默记录错误
            pass
    
    def preload_detection(self, device_id: str, use_template: bool = True, use_dl: bool = True):
        """预加载页面检测（异步）
        
        在执行操作（如点击按钮）前调用此方法，可以提前开始页面识别，
        减少感知延迟。
        
        Args:
            device_id: 设备 ID
            use_template: 是否使用模板匹配（默认True）
            use_dl: 是否使用深度学习（默认True）
            
        Example:
            # 在点击按钮前预加载
            detector.preload_detection(device_id)
            await adb.tap(x, y)  # 点击按钮
            await asyncio.sleep(0.5)  # 等待页面切换
            result = await detector.get_preloaded_result(device_id)
        """
        if self._preload_task and not self._preload_task.done():
            # 如果已有预加载任务在运行，取消它
            self._preload_task.cancel()
        
        # 创建新的预加载任务（不使用缓存，因为是预加载）
        self._preload_device_id = device_id
        self._preload_task = asyncio.create_task(
            self.detect_page(device_id, use_template=use_template, use_dl=use_dl)
        )
        
        self._log(f"[优化混合检测器] 🚀 开始预加载页面检测: {device_id}")
    
    async def get_preloaded_result(self, device_id: str, 
                                   timeout: float = 5.0) -> Optional[PageDetectionResult]:
        """获取预加载的检测结果
        
        Args:
            device_id: 设备 ID
            timeout: 超时时间（秒）
            
        Returns:
            页面检测结果，如果超时或没有预加载任务则返回 None
        """
        if not self._preload_task or self._preload_device_id != device_id:
            self._log("[优化混合检测器] ⚠️  没有对应的预加载任务")
            return None
        
        try:
            # 等待预加载任务完成
            result = await asyncio.wait_for(self._preload_task, timeout=timeout)
            self._log(f"[优化混合检测器] ✓ 预加载完成: {result.state.value}")
            return result
        except asyncio.TimeoutError:
            self._log(f"[优化混合检测器] ⚠️  预加载超时 ({timeout}s)")
            return None
        except asyncio.CancelledError:
            self._log("[优化混合检测器] ⚠️  预加载任务被取消")
            return None
        except Exception as e:
            self._log(f"[优化混合检测器] ✗ 预加载失败: {e}")
            return None
        finally:
            self._preload_task = None
            self._preload_device_id = None
    
    async def detect_page_with_preload(self, device_id: str, 
                                      use_template: bool = False,
                                      use_cache: bool = True) -> PageDetectionResult:
        """检测页面（优先使用预加载结果）
        
        如果有预加载结果，直接返回；否则执行正常检测。
        
        Args:
            device_id: 设备 ID
            use_template: 是否使用模板匹配
            use_cache: 是否使用缓存
            
        Returns:
            页面检测结果
        """
        # 尝试获取预加载结果
        if self._preload_task and self._preload_device_id == device_id:
            result = await self.get_preloaded_result(device_id)
            if result:
                return result
        
        # 没有预加载结果，执行正常检测
        return await self.detect_page(device_id, use_template=use_template, use_cache=use_cache)


# 使用示例
"""
# 创建优化的混合检测器
detector = PageDetectorHybridOptimized(adb, log_callback=log)

# 方式1：预加载模式（推荐用于已知会切换页面的操作）
detector.preload_detection(device_id)  # 开始预加载
await adb.tap(x, y)  # 执行操作（点击按钮）
await asyncio.sleep(0.5)  # 等待页面切换
result = await detector.get_preloaded_result(device_id)  # 获取结果

# 方式2：自动模式（自动使用预加载结果）
detector.preload_detection(device_id)
await adb.tap(x, y)
await asyncio.sleep(0.5)
result = await detector.detect_page_with_preload(device_id)

# 方式3：普通模式（不使用预加载）
result = await detector.detect_page(device_id)
"""
