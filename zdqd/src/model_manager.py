"""
模型管理器 - 全局单例模式管理所有深度学习模型

本模块实现了全局模型管理器，负责在程序启动时预加载所有深度学习模型，
并在整个程序生命周期中共享这些模型实例，从而消除重复加载、减少启动时间和内存占用。

核心功能：
1. 单例模式：确保全局只有一个ModelManager实例
2. 预加载模型：在程序启动时加载所有模型
3. 线程安全：支持多线程并发访问
4. 配置驱动：通过配置文件控制模型加载行为
5. 错误处理：完善的错误处理和降级机制

使用示例：

    基本使用：
    --------
    # 1. 在程序启动时初始化ModelManager
    from src.model_manager import ModelManager
    from src.adb_bridge import ADBBridge
    
    # 创建ADB桥接器
    adb = ADBBridge()
    
    # 获取ModelManager单例
    manager = ModelManager.get_instance()
    
    # 初始化所有模型
    stats = manager.initialize_all_models(
        adb_bridge=adb,
        log_callback=print,
        progress_callback=lambda msg, cur, total: print(f"[{cur}/{total}] {msg}")
    )
    
    print(f"模型加载完成，耗时: {stats['total_time']:.2f}秒")
    
    # 2. 在业务代码中获取模型
    from src.model_manager import ModelManager
    
    manager = ModelManager.get_instance()
    
    # 获取页面分类器
    detector = manager.get_page_detector_integrated()
    
    # 获取YOLO检测器
    yolo = manager.get_page_detector_hybrid()
    
    # 获取OCR线程池
    ocr = manager.get_ocr_thread_pool()
    
    # 3. 程序退出时清理资源
    manager.cleanup()
    
    高级使用：
    --------
    # 查询加载统计
    stats = manager.get_loading_stats()
    print(f"已加载模型: {stats['loaded_models']}/{stats['total_models']}")
    print(f"内存占用: {stats['memory_delta_mb']:.1f}MB")
    
    # 生成详细报告
    report = manager.generate_loading_report()
    print(report)
    
    # 获取特定模型信息
    info = manager.get_model_info('page_detector_integrated')
    print(f"加载时间: {info['load_time']:.2f}秒")
    print(f"设备: {info['device']}")
    
    # 获取优化建议
    suggestions = manager.get_optimization_suggestions()
    for suggestion in suggestions:
        print(f"- {suggestion}")
    
    配置文件：
    --------
    在项目根目录创建 model_config.json 文件：
    
    {
      "models": {
        "page_detector_integrated": {
          "enabled": true,
          "model_path": "page_classifier_pytorch_best.pth",
          "classes_path": "page_classes.json",
          "device": "auto",
          "quantize": false
        },
        "page_detector_hybrid": {
          "enabled": true,
          "yolo_registry_path": "yolo_model_registry.json",
          "mapping_path": "page_yolo_mapping.json",
          "device": "auto"
        },
        "ocr_thread_pool": {
          "enabled": true,
          "thread_count": 4,
          "use_gpu": true
        }
      },
      "startup": {
        "show_progress": true,
        "log_loading_time": true,
        "log_memory_usage": true
      }
    }
    
    性能优化：
    --------
    # 并行加载模型（实验性功能）
    stats = manager.initialize_all_models_parallel(
        adb_bridge=adb,
        max_workers=3
    )
    
    # 模型量化（减少内存占用）
    manager.quantize_model('page_detector_integrated')
    
    # 延迟加载（按需加载）
    manager.enable_lazy_loading(
        'optional_model',
        lambda: load_optional_model(adb)
    )
    model = manager.get_model_lazy('optional_model')

注意事项：
--------
1. ModelManager必须在程序启动时初始化，在GUI显示前完成
2. 所有组件应该从ModelManager获取模型，不要自己创建模型实例
3. 模型加载失败会抛出异常，需要妥善处理
4. 程序退出时应该调用cleanup()释放资源
5. 配置文件是可选的，如果不存在会使用默认配置
"""

import threading
import os
import json
import time
import psutil
import concurrent.futures
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class ModelInfo:
    """模型信息数据类"""
    name: str                    # 模型名称
    instance: Any                # 模型实例
    load_time: float             # 加载时间（秒）
    memory_usage: int            # 内存占用（字节）
    device: str                  # 设备类型（cuda/cpu）
    loaded_at: datetime          # 加载时间戳
    config: Dict[str, Any]       # 配置信息


@dataclass
class LoadingStats:
    """加载统计信息数据类"""
    total_models: int            # 总模型数
    loaded_models: int           # 已加载模型数
    failed_models: int           # 加载失败模型数
    total_time: float            # 总加载时间
    memory_before: int           # 加载前内存
    memory_after: int            # 加载后内存
    errors: List[str]            # 错误列表
    model_times: Dict[str, float]  # 各模型加载时间


class ModelManager:
    """全局模型管理器（线程安全单例）
    
    使用双重检查锁定实现单例模式，确保在多线程环境下只创建一个实例。
    负责在程序启动时加载所有深度学习模型，并提供线程安全的访问接口。
    
    使用示例：
        # 获取单例实例
        manager = ModelManager.get_instance()
        
        # 初始化所有模型
        stats = manager.initialize_all_models(adb_bridge)
        
        # 获取模型实例
        detector = manager.get_page_detector_integrated()
    """
    
    # 类级别的单例实例和锁
    _instance: Optional['ModelManager'] = None
    _lock = threading.Lock()
    _initialized = False
    
    def __new__(cls):
        """双重检查锁定的单例实现
        
        使用双重检查锁定模式确保线程安全：
        1. 第一次检查：避免不必要的锁获取
        2. 获取锁：确保线程安全
        3. 第二次检查：防止多个线程同时创建实例
        
        Returns:
            ModelManager: 单例实例
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化模型管理器（只执行一次）
        
        使用_initialized标志确保初始化代码只执行一次，
        即使__init__被多次调用也不会重复初始化。
        """
        if self._initialized:
            return
        
        with self._lock:
            if self._initialized:
                return
            
            # 确定模型目录
            import sys
            if getattr(sys, 'frozen', False):
                # 打包后的EXE环境
                self.base_dir = Path(sys.executable).parent
            else:
                # 开发环境
                self.base_dir = Path(__file__).parent.parent
            
            # 模型目录（外置）
            self.models_dir = self.base_dir / "models"
            
            # 确保模型目录存在
            if not self.models_dir.exists():
                raise FileNotFoundError(
                    f"模型目录不存在: {self.models_dir}\n"
                    f"请确保 models 文件夹与程序在同一目录下"
                )
            
            # 模型存储
            self._models: Dict[str, Any] = {}
            self._model_info: Dict[str, ModelInfo] = {}
            
            # 配置
            self._config = self._load_config()
            
            # 回调
            self._log_callback: Optional[Callable] = None
            
            # 统计信息
            self._loading_stats = LoadingStats(
                total_models=0,
                loaded_models=0,
                failed_models=0,
                total_time=0.0,
                memory_before=0,
                memory_after=0,
                errors=[],
                model_times={}
            )
            
            self._initialized = True
    
    @classmethod
    def get_instance(cls) -> 'ModelManager':
        """获取单例实例
        
        这是获取ModelManager实例的推荐方式。
        
        Returns:
            ModelManager: 单例实例
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件（带默认值）
        
        尝试从model_config.json加载配置，如果文件不存在或加载失败，
        则使用默认配置。用户配置会递归合并到默认配置中。
        
        Returns:
            Dict[str, Any]: 合并后的配置字典
        """
        config_path = 'model_config.json'
        
        # 默认配置
        default_config = {
            'models': {
                'page_detector_integrated': {
                    'enabled': True,
                    'model_path': 'page_classifier_pytorch_best.pth',
                    'classes_path': 'page_classes.json',
                    'device': 'auto',
                    'quantize': False
                },
                'page_detector_hybrid': {
                    'enabled': True,
                    'yolo_registry_path': 'yolo_model_registry.json',
                    'mapping_path': 'page_yolo_mapping.json',
                    'device': 'auto'
                },
                'ocr_thread_pool': {
                    'enabled': True,
                    'thread_count': 4,
                    'use_gpu': True
                }
            },
            'startup': {
                'show_progress': True,
                'log_loading_time': True,
                'log_memory_usage': True
            }
        }
        
        # 尝试加载配置文件
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                
                # 合并配置（用户配置覆盖默认配置）
                config = self._merge_config(default_config, user_config)
                print(f"[OK] 已加载配置文件: {config_path}")
                return config
                
            except Exception as e:
                print(f"[WARNING] 配置文件加载失败，使用默认配置: {e}")
                return default_config
        else:
            print(f"[WARNING] 配置文件不存在，使用默认配置: {config_path}")
            return default_config
    
    def _merge_config(self, default: Dict, user: Dict) -> Dict:
        """递归合并配置
        
        将用户配置递归合并到默认配置中，用户配置的值会覆盖默认值。
        对于嵌套的字典，会递归合并而不是直接替换。
        
        Args:
            default: 默认配置字典
            user: 用户配置字典
        
        Returns:
            Dict: 合并后的配置字典
        """
        result = default.copy()
        
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # 递归合并嵌套字典
                result[key] = self._merge_config(result[key], value)
            else:
                # 直接覆盖
                result[key] = value
        
        return result
    
    def _log(self, message: str):
        """记录日志
        
        如果设置了日志回调函数，则调用回调函数；否则直接打印。
        
        Args:
            message: 日志消息
        """
        if self._log_callback:
            self._log_callback(message)
        else:
            print(message)
    
    def _is_model_enabled(self, model_name: str) -> bool:
        """检查模型是否在配置中启用
        
        Args:
            model_name: 模型名称
        
        Returns:
            bool: 如果模型启用返回True，否则返回False
        """
        if model_name not in self._config['models']:
            return False
        return self._config['models'][model_name].get('enabled', True)
    
    def _is_critical_model(self, model_name: str) -> bool:
        """判断是否是关键模型（加载失败会阻止启动）
        
        关键模型包括：
        - page_detector_integrated: 页面分类是核心功能
        - page_detector_hybrid: YOLO检测是核心功能
        
        Args:
            model_name: 模型名称
        
        Returns:
            bool: 如果是关键模型返回True，否则返回False
        """
        critical_models = [
            'page_detector_integrated',
            'page_detector_hybrid'
        ]
        return model_name in critical_models
    
    def is_initialized(self) -> bool:
        """检查模型是否已初始化
        
        Returns:
            bool: 如果至少有一个模型已加载返回True，否则返回False
        """
        with self._lock:
            return len(self._models) > 0
    
    def get_loading_stats(self) -> Dict[str, Any]:
        """获取模型加载统计信息
        
        返回详细的加载统计信息，包括：
        - 模型数量统计
        - 时间统计（总时间、各模型时间、平均时间）
        - 内存统计（加载前后、增量、各模型占用）
        - 错误信息
        
        Returns:
            Dict[str, Any]: 包含加载统计信息的字典
        """
        with self._lock:
            memory_delta = self._loading_stats.memory_after - self._loading_stats.memory_before
            
            # 计算平均加载时间
            avg_load_time = 0.0
            if self._loading_stats.loaded_models > 0:
                avg_load_time = self._loading_stats.total_time / self._loading_stats.loaded_models
            
            # 收集各模型的内存占用
            model_memory = {}
            for model_name, info in self._model_info.items():
                model_memory[model_name] = info.memory_usage
            
            return {
                # 模型数量统计
                'total_models': self._loading_stats.total_models,
                'loaded_models': self._loading_stats.loaded_models,
                'failed_models': self._loading_stats.failed_models,
                'success_rate': (self._loading_stats.loaded_models / self._loading_stats.total_models * 100) 
                                if self._loading_stats.total_models > 0 else 0.0,
                
                # 时间统计
                'total_time': self._loading_stats.total_time,
                'average_load_time': avg_load_time,
                'model_times': self._loading_stats.model_times.copy(),
                
                # 内存统计
                'memory_before': self._loading_stats.memory_before,
                'memory_after': self._loading_stats.memory_after,
                'memory_delta': memory_delta,
                'memory_before_mb': self._loading_stats.memory_before / 1024 / 1024,
                'memory_after_mb': self._loading_stats.memory_after / 1024 / 1024,
                'memory_delta_mb': memory_delta / 1024 / 1024,
                'model_memory': model_memory,
                
                # 错误信息
                'errors': self._loading_stats.errors.copy(),
                'has_errors': len(self._loading_stats.errors) > 0
            }
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """获取特定模型的信息
        
        Args:
            model_name: 模型名称
        
        Returns:
            Dict[str, Any]: 模型信息字典
        
        Raises:
            KeyError: 如果模型不存在
        """
        with self._lock:
            if model_name not in self._model_info:
                raise KeyError(f"模型不存在: {model_name}")
            
            info = self._model_info[model_name]
            return {
                'name': info.name,
                'load_time': info.load_time,
                'memory_usage': info.memory_usage,
                'memory_usage_mb': info.memory_usage / 1024 / 1024,
                'device': info.device,
                'loaded_at': info.loaded_at.isoformat(),
                'config': info.config.copy()
            }
    
    def generate_loading_report(self) -> str:
        """生成详细的加载统计报告
        
        生成一个格式化的文本报告，包含所有加载统计信息。
        适合在程序启动时显示或保存到日志文件。
        
        Returns:
            str: 格式化的统计报告文本
        """
        stats = self.get_loading_stats()
        
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("模型加载统计报告")
        report_lines.append("=" * 70)
        report_lines.append("")
        
        # 基本统计
        report_lines.append("【基本统计】")
        report_lines.append(f"  总模型数:     {stats['total_models']}")
        report_lines.append(f"  已加载:       {stats['loaded_models']}")
        report_lines.append(f"  失败:         {stats['failed_models']}")
        report_lines.append(f"  成功率:       {stats['success_rate']:.1f}%")
        report_lines.append("")
        
        # 时间统计
        report_lines.append("【时间统计】")
        report_lines.append(f"  总加载时间:   {stats['total_time']:.2f}秒")
        report_lines.append(f"  平均时间:     {stats['average_load_time']:.2f}秒/模型")
        report_lines.append("")
        report_lines.append("  各模型加载时间:")
        for model_name, load_time in stats['model_times'].items():
            percentage = (load_time / stats['total_time'] * 100) if stats['total_time'] > 0 else 0
            report_lines.append(f"    - {model_name:30s} {load_time:6.2f}秒 ({percentage:5.1f}%)")
        report_lines.append("")
        
        # 内存统计
        report_lines.append("【内存统计】")
        report_lines.append(f"  加载前内存:   {stats['memory_before_mb']:.1f}MB")
        report_lines.append(f"  加载后内存:   {stats['memory_after_mb']:.1f}MB")
        report_lines.append(f"  内存增量:     {stats['memory_delta_mb']:.1f}MB")
        
        if stats['model_memory']:
            report_lines.append("")
            report_lines.append("  各模型内存占用:")
            for model_name, memory in stats['model_memory'].items():
                memory_mb = memory / 1024 / 1024
                percentage = (memory / stats['memory_delta'] * 100) if stats['memory_delta'] > 0 else 0
                report_lines.append(f"    - {model_name:30s} {memory_mb:6.1f}MB ({percentage:5.1f}%)")
        report_lines.append("")
        
        # 错误信息
        if stats['has_errors']:
            report_lines.append("【错误信息】")
            for i, error in enumerate(stats['errors'], 1):
                report_lines.append(f"  {i}. {error}")
            report_lines.append("")
        
        # 模型详情
        report_lines.append("【模型详情】")
        with self._lock:
            for model_name, info in self._model_info.items():
                report_lines.append(f"  {model_name}:")
                report_lines.append(f"    设备:       {info.device}")
                report_lines.append(f"    加载时间:   {info.load_time:.2f}秒")
                report_lines.append(f"    内存占用:   {info.memory_usage / 1024 / 1024:.1f}MB")
                report_lines.append(f"    加载时刻:   {info.loaded_at.strftime('%Y-%m-%d %H:%M:%S')}")
                report_lines.append("")
        
        report_lines.append("=" * 70)
        
        return "\n".join(report_lines)
    
    def initialize_all_models(
        self,
        adb_bridge,
        log_callback: Optional[Callable] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> Dict[str, Any]:
        """初始化所有模型
        
        在程序启动时调用此方法加载所有配置的模型。
        支持进度回调和日志回调，可以向用户显示加载进度。
        
        Args:
            adb_bridge: ADB桥接器实例
            log_callback: 日志回调函数，用于记录加载过程
            progress_callback: 进度回调函数 (message, current, total)
        
        Returns:
            Dict包含加载统计信息:
            {
                'success': bool,              # 是否所有关键模型都加载成功
                'models_loaded': List[str],   # 已加载的模型列表
                'total_time': float,          # 总加载时间（秒）
                'memory_before': int,         # 加载前内存（字节）
                'memory_after': int,          # 加载后内存（字节）
                'memory_delta': int,          # 内存增量（字节）
                'errors': List[str]           # 错误列表
            }
        
        Raises:
            RuntimeError: 如果关键模型加载失败
            FileNotFoundError: 如果必需的模型文件不存在
        """
        self._log_callback = log_callback
        start_time = time.time()
        
        # 记录初始内存
        process = psutil.Process(os.getpid())
        self._loading_stats.memory_before = process.memory_info().rss
        
        self._log("=" * 70)
        self._log("开始加载模型...")
        self._log("=" * 70)
        self._log(f"初始内存: {self._loading_stats.memory_before / 1024 / 1024:.1f}MB")
        self._log("")
        
        # 预先验证所有模型文件
        self._log("[验证] 检查模型文件...")
        missing_files = self._validate_model_files()
        if missing_files:
            error_msg = f"以下模型文件缺失:\n" + "\n".join(f"  - {f}" for f in missing_files)
            self._log(f"[ERROR] {error_msg}")
            raise FileNotFoundError(error_msg)
        self._log("[OK] 所有模型文件验证通过")
        self._log("")
        
        # 要加载的模型列表
        # 注意：page_detector_hybrid 已被 page_detector_integrated 替代
        # 默认只加载整合检测器，混合检测器已废弃
        models_to_load = [
            ('page_detector_integrated', self._load_page_detector_integrated),
            # ('page_detector_hybrid', self._load_page_detector_hybrid),  # 已废弃，不再加载
            ('ocr_thread_pool', self._load_ocr_thread_pool)
        ]
        
        self._loading_stats.total_models = len(models_to_load)
        
        # 逐个加载模型
        for idx, (model_name, load_func) in enumerate(models_to_load, 1):
            # 检查配置是否启用
            if not self._is_model_enabled(model_name):
                self._log(f"[{idx}/{len(models_to_load)}] 跳过模型: {model_name} (已禁用)")
                self._log("")
                continue
            
            # 计算进度百分比
            progress_percent = (idx - 1) / len(models_to_load) * 100
            
            # 进度回调
            if progress_callback:
                progress_callback(
                    f"正在加载 {model_name}...",
                    idx,
                    len(models_to_load)
                )
            
            # 显示进度条
            self._log(f"[{idx}/{len(models_to_load)}] [{progress_percent:5.1f}%] 正在加载 {model_name}...")
            
            # 记录当前内存
            current_memory = process.memory_info().rss
            current_memory_mb = current_memory / 1024 / 1024
            self._log(f"  当前内存: {current_memory_mb:.1f}MB")
            
            # 加载模型（带重试机制）
            try:
                model_start = time.time()
                model_instance = self._load_model_with_retry(
                    model_name, 
                    lambda: load_func(adb_bridge),
                    max_retries=3
                )
                model_time = time.time() - model_start
                
                # 计算模型占用的内存
                after_memory = process.memory_info().rss
                model_memory = after_memory - current_memory
                model_memory_mb = model_memory / 1024 / 1024
                
                # 保存模型
                self._models[model_name] = model_instance
                self._loading_stats.model_times[model_name] = model_time
                self._loading_stats.loaded_models += 1
                
                # 记录模型信息
                self._model_info[model_name] = ModelInfo(
                    name=model_name,
                    instance=model_instance,
                    load_time=model_time,
                    memory_usage=model_memory,
                    device=self._get_model_device(model_instance),
                    loaded_at=datetime.now(),
                    config=self._config['models'][model_name].copy()
                )
                
                # 显示加载成功信息（带时间和内存）
                self._log(f"  [OK] 加载成功")
                self._log(f"  ├─ 耗时: {model_time:.2f}秒")
                self._log(f"  ├─ 内存增量: {model_memory_mb:.1f}MB")
                self._log(f"  └─ 设备: {self._model_info[model_name].device}")
                self._log("")
                
            except Exception as e:
                error_msg = f"[ERROR] {model_name} 加载失败: {e}"
                self._log(f"  {error_msg}")
                self._log("")
                self._loading_stats.errors.append(error_msg)
                self._loading_stats.failed_models += 1
                
                # 如果是关键模型，抛出异常
                if self._is_critical_model(model_name):
                    raise RuntimeError(f"关键模型加载失败: {model_name}") from e
                else:
                    self._log(f"  [WARNING] 可选模型加载失败，程序将继续运行")
                    self._log("")
        
        # 记录最终内存
        self._loading_stats.memory_after = process.memory_info().rss
        self._loading_stats.total_time = time.time() - start_time
        
        # 显示加载统计（结构化输出）
        memory_delta = self._loading_stats.memory_after - self._loading_stats.memory_before
        
        self._log("=" * 70)
        self._log("模型加载完成")
        self._log("=" * 70)
        self._log("")
        
        self._log("【统计信息】")
        self._log(f"  总加载时间:   {self._loading_stats.total_time:.2f}秒")
        self._log(f"  已加载模型:   {self._loading_stats.loaded_models}/{self._loading_stats.total_models}")
        self._log(f"  失败模型:     {self._loading_stats.failed_models}")
        
        if self._loading_stats.loaded_models > 0:
            avg_time = self._loading_stats.total_time / self._loading_stats.loaded_models
            self._log(f"  平均时间:     {avg_time:.2f}秒/模型")
        
        self._log("")
        self._log("【内存使用】")
        self._log(f"  加载前:       {self._loading_stats.memory_before / 1024 / 1024:.1f}MB")
        self._log(f"  加载后:       {self._loading_stats.memory_after / 1024 / 1024:.1f}MB")
        self._log(f"  增量:         {memory_delta / 1024 / 1024:.1f}MB")
        
        # 显示各模型的时间占比
        if self._loading_stats.model_times:
            self._log("")
            self._log("【各模型加载时间】")
            for model_name, load_time in self._loading_stats.model_times.items():
                percentage = (load_time / self._loading_stats.total_time * 100) if self._loading_stats.total_time > 0 else 0
                self._log(f"  {model_name:30s} {load_time:6.2f}秒 ({percentage:5.1f}%)")
        
        # 显示各模型的内存占比
        if self._model_info:
            self._log("")
            self._log("【各模型内存占用】")
            for model_name, info in self._model_info.items():
                memory_mb = info.memory_usage / 1024 / 1024
                percentage = (info.memory_usage / memory_delta * 100) if memory_delta > 0 else 0
                self._log(f"  {model_name:30s} {memory_mb:6.1f}MB ({percentage:5.1f}%)")
        
        # 显示错误信息
        if self._loading_stats.errors:
            self._log("")
            self._log("【错误列表】")
            for i, error in enumerate(self._loading_stats.errors, 1):
                self._log(f"  {i}. {error}")
        
        self._log("")
        self._log("=" * 70)
        
        # 返回统计信息
        return {
            'success': self._loading_stats.failed_models == 0,
            'models_loaded': list(self._models.keys()),
            'total_time': self._loading_stats.total_time,
            'memory_before': self._loading_stats.memory_before,
            'memory_after': self._loading_stats.memory_after,
            'memory_delta': memory_delta,
            'errors': self._loading_stats.errors.copy()
        }
    
    def _get_model_device(self, model_instance) -> str:
        """获取模型使用的设备类型
        
        Args:
            model_instance: 模型实例
        
        Returns:
            str: 设备类型 ('cuda', 'cpu', 或 'unknown')
        """
        try:
            # 尝试获取PyTorch模型的设备
            if hasattr(model_instance, '_device'):
                device = model_instance._device
                if hasattr(device, 'type'):
                    return device.type
                return str(device)
            
            # 尝试获取模型的device属性
            if hasattr(model_instance, 'device'):
                return str(model_instance.device)
            
            return 'unknown'
        except Exception:
            return 'unknown'

    
    def _load_page_detector_integrated(self, adb_bridge) -> 'PageDetectorIntegrated':
        """加载深度学习页面分类器
        
        加载PageDetectorIntegrated模型，支持GPU加速和自动降级。
        
        Args:
            adb_bridge: ADB桥接器实例
        
        Returns:
            PageDetectorIntegrated: 页面分类器实例
        
        Raises:
            FileNotFoundError: 如果模型文件不存在
            RuntimeError: 如果模型加载失败
        """
        try:
            from .page_detector_integrated import PageDetectorIntegrated
        except (ImportError, ValueError):
            # 如果相对导入失败，尝试从src包导入
            try:
                from src.page_detector_integrated import PageDetectorIntegrated
            except ImportError:
                # 最后尝试直接导入
                import page_detector_integrated
                PageDetectorIntegrated = page_detector_integrated.PageDetectorIntegrated
        
        config = self._config['models']['page_detector_integrated']
        
        # 验证文件存在（使用models目录）
        model_path = self.models_dir / config['model_path']
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        classes_path = self.models_dir / config['classes_path']
        if not classes_path.exists():
            raise FileNotFoundError(f"类别文件不存在: {classes_path}")
        
        # 检测GPU可用性并自动降级
        device_config = config.get('device', 'auto')
        self._check_gpu_availability(device_config)
        
        # 创建实例（不显示文件路径，用户不需要看到）
        # self._log(f"  - 模型路径: {model_path}")
        # self._log(f"  - 类别文件: {classes_path}")
        
        detector = PageDetectorIntegrated(
            adb=adb_bridge,
            classifier_model_path=model_path,
            classes_path=classes_path,
            yolo_registry_path='yolo_model_registry.json',
            mapping_path='page_yolo_mapping.json',
            log_callback=self._log_callback
        )
        
        return detector
    
    def _load_page_detector_hybrid(self, adb_bridge) -> 'PageDetectorHybrid':
        """加载YOLO检测器
        
        加载PageDetectorHybrid模型，包含所有YOLO模型。
        
        Args:
            adb_bridge: ADB桥接器实例
        
        Returns:
            PageDetectorHybrid: YOLO检测器实例
        
        Raises:
            FileNotFoundError: 如果配置文件不存在
            RuntimeError: 如果模型加载失败
        """
        try:
            from .page_detector_hybrid import PageDetectorHybrid
        except (ImportError, ValueError):
            # 如果相对导入失败，尝试从src包导入
            try:
                from src.page_detector_hybrid import PageDetectorHybrid
            except ImportError:
                # 最后尝试直接导入
                import page_detector_hybrid
                PageDetectorHybrid = page_detector_hybrid.PageDetectorHybrid
        
        config = self._config['models']['page_detector_hybrid']
        
        # 验证配置文件存在（使用models目录）
        mapping_path = self.models_dir / config['mapping_path']
        if not mapping_path.exists():
            raise FileNotFoundError(f"映射文件不存在: {mapping_path}")
        
        registry_path = self.models_dir / config.get('yolo_registry_path', 'yolo_model_registry.json')
        if not registry_path.exists():
            self._log(f"  [WARNING] YOLO注册表文件不存在: {registry_path}")
        
        # 创建实例（不显示文件路径，用户不需要看到）
        # self._log(f"  - 映射文件: {mapping_path}")
        # self._log(f"  - 注册表: {registry_path}")
        
        detector = PageDetectorHybrid(
            adb=adb_bridge,
            log_callback=self._log_callback
        )
        
        return detector
    
    def _load_ocr_thread_pool(self, adb_bridge) -> 'OCRThreadPool':
        """加载OCR线程池
        
        创建或获取OCRThreadPool单例实例。
        
        Args:
            adb_bridge: ADB桥接器实例（未使用，保持接口一致）
        
        Returns:
            OCRThreadPool: OCR线程池实例
        
        Raises:
            RuntimeError: 如果OCR初始化失败
        """
        try:
            from .ocr_thread_pool import OCRThreadPool
        except (ImportError, ValueError):
            # 如果相对导入失败，尝试从src包导入
            try:
                from src.ocr_thread_pool import OCRThreadPool
            except ImportError:
                # 最后尝试直接导入
                import ocr_thread_pool
                OCRThreadPool = ocr_thread_pool.OCRThreadPool
        
        config = self._config['models']['ocr_thread_pool']
        
        # OCRThreadPool是单例，直接创建即可
        self._log(f"  - 线程数: {config.get('thread_count', 4)}")
        self._log(f"  - GPU加速: {config.get('use_gpu', True)}")
        
        pool = OCRThreadPool()
        
        # 验证OCR是否可用
        if not hasattr(pool, '_ocr') or pool._ocr is None:
            raise RuntimeError("OCR引擎初始化失败，请检查rapidocr是否正确安装")
        
        return pool
    
    def _check_gpu_availability(self, device_config: str):
        """检查GPU可用性并记录日志
        
        Args:
            device_config: 设备配置 ('auto', 'cuda', 'cpu')
        """
        try:
            import torch
            
            if device_config == 'auto':
                if torch.cuda.is_available():
                    self._log("  [OK] 检测到GPU，使用CUDA加速")
                else:
                    self._log("  [WARNING] GPU不可用，使用CPU模式")
            elif device_config == 'cuda':
                if not torch.cuda.is_available():
                    self._log("  [WARNING] 配置要求GPU但不可用，降级到CPU模式")
                else:
                    self._log("  [OK] 使用GPU加速")
            else:
                self._log(f"  - 使用设备: {device_config}")
        except ImportError:
            self._log("  [WARNING] PyTorch未安装，无法检测GPU")

    
    def _validate_model_files(self) -> List[str]:
        """验证所有模型文件是否存在
        
        在加载模型前预先验证所有必需的文件，避免加载过程中出错。
        
        Returns:
            List[str]: 缺失的文件路径列表，如果所有文件都存在则返回空列表
        """
        missing_files = []
        
        # 检查页面分类器
        if self._is_model_enabled('page_detector_integrated'):
            config = self._config['models']['page_detector_integrated']
            
            model_path = self.models_dir / config['model_path']
            if not model_path.exists():
                missing_files.append(config['model_path'])
            
            classes_path = self.models_dir / config['classes_path']
            if not classes_path.exists():
                missing_files.append(config['classes_path'])
        
        # 检查YOLO映射文件
        if self._is_model_enabled('page_detector_hybrid'):
            config = self._config['models']['page_detector_hybrid']
            
            mapping_path = self.models_dir / config['mapping_path']
            if not mapping_path.exists():
                missing_files.append(config['mapping_path'])
            
            # YOLO注册表是可选的，不检查
        
        # OCR不需要额外文件
        
        return missing_files
    
    def _load_model_with_retry(
        self, 
        model_name: str, 
        load_func: Callable, 
        max_retries: int = 3
    ):
        """带重试的模型加载
        
        如果模型加载失败，会自动重试指定次数。
        每次重试之间会等待1秒。
        
        Args:
            model_name: 模型名称
            load_func: 加载函数
            max_retries: 最大重试次数
        
        Returns:
            加载的模型实例
        
        Raises:
            RuntimeError: 如果所有重试都失败
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                return load_func()
            except Exception as e:
                last_error = e
                
                if attempt < max_retries - 1:
                    self._log(f"  [WARNING] 加载失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                    self._log(f"  - 等待1秒后重试...")
                    time.sleep(1)
                else:
                    self._log(f"  [ERROR] 加载失败 (尝试 {attempt + 1}/{max_retries}): {e}")
        
        # 所有重试都失败
        raise RuntimeError(
            f"{model_name} 加载失败（已重试{max_retries}次）"
        ) from last_error
    
    def get_page_detector_integrated(self) -> 'PageDetectorIntegrated':
        """获取深度学习页面分类器（线程安全）
        
        Returns:
            PageDetectorIntegrated: 页面分类器实例
        
        Raises:
            RuntimeError: 如果模型未初始化
        """
        with self._lock:
            if 'page_detector_integrated' not in self._models:
                raise RuntimeError(
                    "PageDetectorIntegrated未初始化，请先调用initialize_all_models()"
                )
            return self._models['page_detector_integrated']
    
    def get_page_detector_hybrid(self) -> 'PageDetectorIntegrated':
        """获取页面检测器（向后兼容方法）
        
        注意：PageDetectorHybrid已被PageDetectorIntegrated替代。
        此方法现在返回整合检测器以保持向后兼容性。
        
        Returns:
            PageDetectorIntegrated: 整合检测器实例
        
        Raises:
            RuntimeError: 如果模型未初始化
        """
        import warnings
        warnings.warn(
            "get_page_detector_hybrid() 已废弃，请使用 get_page_detector_integrated()。"
            "混合检测器已被整合检测器替代。",
            DeprecationWarning,
            stacklevel=2
        )
        
        # 返回整合检测器（向后兼容）
        return self.get_page_detector_integrated()
    
    def get_ocr_thread_pool(self) -> 'OCRThreadPool':
        """获取OCR线程池（线程安全）
        
        Returns:
            OCRThreadPool: OCR线程池实例
        
        Raises:
            RuntimeError: 如果模型未初始化
        """
        with self._lock:
            if 'ocr_thread_pool' not in self._models:
                raise RuntimeError(
                    "OCRThreadPool未初始化，请先调用initialize_all_models()"
                )
            return self._models['ocr_thread_pool']
    
    def cleanup(self):
        """清理所有模型资源
        
        在程序退出时调用此方法释放所有已加载的模型。
        会尝试清理GPU内存（如果使用）并强制垃圾回收。
        """
        with self._lock:
            self._log("\n" + "=" * 60)
            self._log("开始清理模型资源...")
            self._log("=" * 60)
            
            # 释放所有模型实例
            for model_name in list(self._models.keys()):
                try:
                    self._log(f"释放模型: {model_name}")
                    del self._models[model_name]
                except Exception as e:
                    self._log(f"[WARNING] 释放模型失败 {model_name}: {e}")
            
            # 清空模型信息
            self._model_info.clear()
            
            # 清空GPU缓存（如果使用PyTorch）
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    self._log("[OK] GPU缓存已清理")
            except ImportError:
                pass
            except Exception as e:
                self._log(f"[WARNING] GPU缓存清理失败: {e}")
            
            # 强制垃圾回收
            import gc
            gc.collect()
            self._log("[OK] 垃圾回收完成")
            
            self._log("=" * 60)
            self._log("资源清理完成")
            self._log("=" * 60)

    # ========================================================================
    # 性能优化方法（可选）
    # ========================================================================
    
    def initialize_all_models_parallel(
        self,
        adb_bridge,
        log_callback: Optional[Callable] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        max_workers: int = 3
    ) -> Dict[str, Any]:
        """并行初始化所有模型（性能优化版本）
        
        使用线程池并行加载多个模型，可以显著减少总加载时间。
        注意：并行加载需要确保各模型加载过程是线程安全的。
        
        Args:
            adb_bridge: ADB桥接器实例
            log_callback: 日志回调函数
            progress_callback: 进度回调函数
            max_workers: 最大并行线程数（默认3）
        
        Returns:
            Dict包含加载统计信息
        
        警告:
            并行加载可能会导致GPU资源竞争，建议在CPU模式下使用，
            或确保GPU有足够的显存支持并行加载。
        """
        self._log_callback = log_callback
        start_time = time.time()
        
        # 记录初始内存
        process = psutil.Process(os.getpid())
        self._loading_stats.memory_before = process.memory_info().rss
        
        self._log("=" * 70)
        self._log("开始并行加载模型...")
        self._log("=" * 70)
        self._log(f"初始内存: {self._loading_stats.memory_before / 1024 / 1024:.1f}MB")
        self._log(f"并行线程数: {max_workers}")
        self._log("")
        
        # 预先验证所有模型文件
        self._log("[验证] 检查模型文件...")
        missing_files = self._validate_model_files()
        if missing_files:
            error_msg = f"以下模型文件缺失:\n" + "\n".join(f"  - {f}" for f in missing_files)
            self._log(f"[ERROR] {error_msg}")
            raise FileNotFoundError(error_msg)
        self._log("[OK] 所有模型文件验证通过")
        self._log("")
        
        # 要加载的模型列表
        # 注意：page_detector_hybrid 已被 page_detector_integrated 替代
        # 默认只加载整合检测器，混合检测器已废弃
        models_to_load = [
            ('page_detector_integrated', self._load_page_detector_integrated),
            # ('page_detector_hybrid', self._load_page_detector_hybrid),  # 已废弃，不再加载
            ('ocr_thread_pool', self._load_ocr_thread_pool)
        ]
        
        # 过滤出启用的模型
        enabled_models = [
            (name, func) for name, func in models_to_load
            if self._is_model_enabled(name)
        ]
        
        self._loading_stats.total_models = len(enabled_models)
        
        # 使用线程池并行加载
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有加载任务
            futures = {}
            for model_name, load_func in enabled_models:
                self._log(f"[提交] {model_name} 加载任务")
                future = executor.submit(self._load_model_safe, model_name, load_func, adb_bridge)
                futures[future] = model_name
            
            self._log("")
            self._log("等待所有模型加载完成...")
            self._log("")
            
            # 等待所有任务完成
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                model_name = futures[future]
                completed += 1
                
                try:
                    result = future.result()
                    
                    if result['success']:
                        # 保存模型
                        self._models[model_name] = result['instance']
                        self._loading_stats.model_times[model_name] = result['load_time']
                        self._loading_stats.loaded_models += 1
                        
                        # 记录模型信息
                        self._model_info[model_name] = ModelInfo(
                            name=model_name,
                            instance=result['instance'],
                            load_time=result['load_time'],
                            memory_usage=result['memory_delta'],
                            device=self._get_model_device(result['instance']),
                            loaded_at=datetime.now(),
                            config=self._config['models'][model_name].copy()
                        )
                        
                        self._log(f"[{completed}/{len(enabled_models)}] [OK] {model_name} 加载完成")
                        self._log(f"  ├─ 耗时: {result['load_time']:.2f}秒")
                        self._log(f"  ├─ 内存增量: {result['memory_delta'] / 1024 / 1024:.1f}MB")
                        self._log(f"  └─ 设备: {self._model_info[model_name].device}")
                        self._log("")
                    else:
                        error_msg = f"[ERROR] {model_name} 加载失败: {result['error']}"
                        self._log(f"[{completed}/{len(enabled_models)}] {error_msg}")
                        self._log("")
                        self._loading_stats.errors.append(error_msg)
                        self._loading_stats.failed_models += 1
                        
                        if self._is_critical_model(model_name):
                            raise RuntimeError(f"关键模型加载失败: {model_name}")
                
                except Exception as e:
                    error_msg = f"[ERROR] {model_name} 加载异常: {e}"
                    self._log(f"[{completed}/{len(enabled_models)}] {error_msg}")
                    self._log("")
                    self._loading_stats.errors.append(error_msg)
                    self._loading_stats.failed_models += 1
                    
                    if self._is_critical_model(model_name):
                        raise
                
                # 进度回调
                if progress_callback:
                    progress_callback(
                        f"已完成 {completed}/{len(enabled_models)} 个模型",
                        completed,
                        len(enabled_models)
                    )
        
        # 记录最终内存
        self._loading_stats.memory_after = process.memory_info().rss
        self._loading_stats.total_time = time.time() - start_time
        
        # 显示加载统计
        memory_delta = self._loading_stats.memory_after - self._loading_stats.memory_before
        
        self._log("=" * 70)
        self._log("并行加载完成")
        self._log("=" * 70)
        self._log("")
        
        self._log("【统计信息】")
        self._log(f"  总加载时间:   {self._loading_stats.total_time:.2f}秒")
        self._log(f"  已加载模型:   {self._loading_stats.loaded_models}/{self._loading_stats.total_models}")
        self._log(f"  失败模型:     {self._loading_stats.failed_models}")
        self._log(f"  并行线程数:   {max_workers}")
        
        if self._loading_stats.loaded_models > 0:
            # 计算理论串行时间
            serial_time = sum(self._loading_stats.model_times.values())
            speedup = serial_time / self._loading_stats.total_time if self._loading_stats.total_time > 0 else 1
            self._log(f"  理论串行时间: {serial_time:.2f}秒")
            self._log(f"  加速比:       {speedup:.2f}x")
        
        self._log("")
        self._log("【内存使用】")
        self._log(f"  加载前:       {self._loading_stats.memory_before / 1024 / 1024:.1f}MB")
        self._log(f"  加载后:       {self._loading_stats.memory_after / 1024 / 1024:.1f}MB")
        self._log(f"  增量:         {memory_delta / 1024 / 1024:.1f}MB")
        self._log("")
        
        self._log("=" * 70)
        
        # 返回统计信息
        return {
            'success': self._loading_stats.failed_models == 0,
            'models_loaded': list(self._models.keys()),
            'total_time': self._loading_stats.total_time,
            'memory_before': self._loading_stats.memory_before,
            'memory_after': self._loading_stats.memory_after,
            'memory_delta': memory_delta,
            'errors': self._loading_stats.errors,
            'model_times': self._loading_stats.model_times.copy()
        }
    
    def _load_model_safe(self, model_name: str, load_func: Callable, adb_bridge) -> Dict[str, Any]:
        """线程安全的模型加载包装器
        
        用于并行加载时捕获异常并返回结果字典。
        
        Args:
            model_name: 模型名称
            load_func: 加载函数
            adb_bridge: ADB桥接器
        
        Returns:
            Dict包含加载结果:
            {
                'success': bool,
                'instance': Any,
                'load_time': float,
                'memory_delta': int,
                'error': str (如果失败)
            }
        """
        try:
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss
            start_time = time.time()
            
            # 加载模型
            instance = load_func(adb_bridge)
            
            load_time = time.time() - start_time
            memory_after = process.memory_info().rss
            memory_delta = memory_after - memory_before
            
            return {
                'success': True,
                'instance': instance,
                'load_time': load_time,
                'memory_delta': memory_delta
            }
        
        except Exception as e:
            return {
                'success': False,
                'instance': None,
                'load_time': 0,
                'memory_delta': 0,
                'error': str(e)
            }
    
    def quantize_model(self, model_name: str) -> bool:
        """量化指定的模型以减少内存占用和提升推理速度
        
        使用PyTorch的动态量化技术将模型转换为int8精度。
        量化可以减少约50%的内存占用，并提升推理速度。
        
        Args:
            model_name: 要量化的模型名称
        
        Returns:
            bool: 量化是否成功
        
        注意:
            - 只支持PyTorch模型
            - 量化会略微降低精度（通常可以接受）
            - 量化后的模型无法在GPU上运行
        """
        with self._lock:
            if model_name not in self._models:
                self._log(f"[ERROR] 模型不存在: {model_name}")
                return False
            
            model_instance = self._models[model_name]
            
            # 检查是否是PyTorch模型
            if not hasattr(model_instance, '_model'):
                self._log(f"[ERROR] {model_name} 不支持量化（非PyTorch模型）")
                return False
            
            try:
                import torch
                
                self._log(f"正在量化模型: {model_name}...")
                
                # 记录量化前的内存
                process = psutil.Process(os.getpid())
                memory_before = process.memory_info().rss
                
                # 执行动态量化
                quantized_model = torch.quantization.quantize_dynamic(
                    model_instance._model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
                
                # 替换模型
                model_instance._model = quantized_model
                
                # 记录量化后的内存
                memory_after = process.memory_info().rss
                memory_saved = memory_before - memory_after
                
                self._log(f"[OK] {model_name} 量化完成")
                self._log(f"  内存节省: {memory_saved / 1024 / 1024:.1f}MB")
                
                # 更新模型信息
                if model_name in self._model_info:
                    self._model_info[model_name].config['quantized'] = True
                
                return True
            
            except Exception as e:
                self._log(f"[ERROR] {model_name} 量化失败: {e}")
                return False
    
    def enable_lazy_loading(self, model_name: str, loader_func: Callable):
        """启用指定模型的延迟加载
        
        延迟加载允许在首次访问时才加载模型，而不是在启动时加载。
        适用于不常用的模型，可以减少启动时间。
        
        Args:
            model_name: 模型名称
            loader_func: 加载函数，签名为 () -> model_instance
        
        示例:
            manager.enable_lazy_loading(
                'optional_model',
                lambda: load_optional_model(adb)
            )
        """
        with self._lock:
            if not hasattr(self, '_lazy_loaders'):
                self._lazy_loaders = {}
            
            self._lazy_loaders[model_name] = loader_func
            self._log(f"[OK] 已启用延迟加载: {model_name}")
    
    def get_model_lazy(self, model_name: str):
        """获取模型（支持延迟加载）
        
        如果模型已加载，直接返回；
        如果模型注册了延迟加载，则执行加载；
        否则抛出异常。
        
        Args:
            model_name: 模型名称
        
        Returns:
            模型实例
        
        Raises:
            RuntimeError: 如果模型未加载且未注册延迟加载
        """
        with self._lock:
            # 如果已加载，直接返回
            if model_name in self._models:
                return self._models[model_name]
            
            # 如果有延迟加载函数，执行加载
            if hasattr(self, '_lazy_loaders') and model_name in self._lazy_loaders:
                self._log(f"延迟加载模型: {model_name}")
                
                try:
                    start_time = time.time()
                    loader = self._lazy_loaders[model_name]
                    model_instance = loader()
                    load_time = time.time() - start_time
                    
                    # 保存模型
                    self._models[model_name] = model_instance
                    self._loading_stats.model_times[model_name] = load_time
                    
                    self._log(f"[OK] {model_name} 延迟加载完成 ({load_time:.2f}秒)")
                    
                    return model_instance
                
                except Exception as e:
                    self._log(f"[ERROR] {model_name} 延迟加载失败: {e}")
                    raise RuntimeError(f"延迟加载失败: {model_name}") from e
            
            # 模型未注册
            raise RuntimeError(f"模型未加载且未注册延迟加载: {model_name}")
    
    def get_optimization_suggestions(self) -> List[str]:
        """获取性能优化建议
        
        基于当前系统状态和模型加载情况，提供优化建议。
        
        Returns:
            List[str]: 优化建议列表
        """
        suggestions = []
        
        # 检查加载时间
        if self._loading_stats.total_time > 10:
            suggestions.append(
                f"模型加载时间较长（{self._loading_stats.total_time:.1f}秒），"
                "建议考虑并行加载或模型量化"
            )
        
        # 检查内存占用
        memory_delta = self._loading_stats.memory_after - self._loading_stats.memory_before
        if memory_delta > 1024 * 1024 * 1024:  # > 1GB
            suggestions.append(
                f"模型内存占用较大（{memory_delta / 1024 / 1024:.0f}MB），"
                "建议考虑模型量化以减少内存占用"
            )
        
        # 检查是否有未使用的模型
        if hasattr(self, '_model_access_count'):
            for model_name, count in self._model_access_count.items():
                if count == 0:
                    suggestions.append(
                        f"模型 {model_name} 从未被访问，建议禁用或使用延迟加载"
                    )
        
        # 检查GPU使用情况
        try:
            import torch
            if torch.cuda.is_available():
                for model_name, info in self._model_info.items():
                    if info.device == 'cpu':
                        suggestions.append(
                            f"模型 {model_name} 运行在CPU上，建议启用GPU加速以提升性能"
                        )
        except ImportError:
            pass
        
        if not suggestions:
            suggestions.append("当前配置已优化，无需调整")
        
        return suggestions
