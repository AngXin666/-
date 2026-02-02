"""
性能基准测试脚本

测试ModelManager优化前后的性能差异：
1. 单账号处理时间
2. 30账号总时间
3. 内存占用
4. 模型加载时间

使用方法:
    python benchmark_model_manager.py --mode [with_manager|without_manager|compare]
"""

import time
import psutil
import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.adb_bridge import ADBBridge
from src.model_manager import ModelManager


class PerformanceBenchmark:
    """性能基准测试类"""
    
    def __init__(self, use_model_manager: bool = True):
        """初始化基准测试
        
        Args:
            use_model_manager: 是否使用ModelManager（True=优化后，False=优化前）
        """
        self.use_model_manager = use_model_manager
        self.process = psutil.Process(os.getpid())
        self.results = {
            'mode': 'with_manager' if use_model_manager else 'without_manager',
            'timestamp': datetime.now().isoformat(),
            'single_account': {},
            'multiple_accounts': {},
            'memory': {},
            'model_loading': {}
        }
    
    def measure_memory(self) -> int:
        """测量当前内存使用（字节）"""
        return self.process.memory_info().rss
    
    def format_memory(self, bytes_value: int) -> str:
        """格式化内存显示"""
        mb = bytes_value / 1024 / 1024
        return f"{mb:.1f}MB"
    
    def benchmark_model_loading(self) -> Dict[str, Any]:
        """基准测试：模型加载时间"""
        print("\n" + "=" * 60)
        print("基准测试：模型加载时间")
        print("=" * 60)
        
        memory_before = self.measure_memory()
        start_time = time.time()
        
        if self.use_model_manager:
            # 使用ModelManager加载
            print("使用ModelManager加载所有模型...")
            adb = ADBBridge()
            manager = ModelManager.get_instance()
            
            stats = manager.initialize_all_models(
                adb_bridge=adb,
                log_callback=print
            )
            
            load_time = time.time() - start_time
            memory_after = self.measure_memory()
            memory_delta = memory_after - memory_before
            
            result = {
                'total_time': load_time,
                'models_loaded': stats['models_loaded'],
                'memory_before': memory_before,
                'memory_after': memory_after,
                'memory_delta': memory_delta,
                'model_times': stats.get('model_times', {})
            }
        else:
            # 模拟不使用ModelManager的情况（每次都重新加载）
            print("模拟不使用ModelManager（每次重新加载）...")
            
            # 这里我们模拟加载一次的时间
            # 实际上在旧系统中，每个账号都会重复这个过程
            from src.page_detector_integrated import PageDetectorIntegrated
            from src.page_detector_hybrid_optimized import PageDetectorHybridOptimized
            from src.ocr_thread_pool import OCRThreadPool
            
            adb = ADBBridge()
            
            # 加载第一个模型
            t1 = time.time()
            detector1 = PageDetectorIntegrated(adb, log_callback=print)
            time1 = time.time() - t1
            
            # 加载第二个模型
            t2 = time.time()
            detector2 = PageDetectorHybridOptimized(adb, log_callback=print)
            time2 = time.time() - t2
            
            # 加载OCR线程池
            t3 = time.time()
            ocr_pool = OCRThreadPool(thread_count=4)
            time3 = time.time() - t3
            
            load_time = time.time() - start_time
            memory_after = self.measure_memory()
            memory_delta = memory_after - memory_before
            
            result = {
                'total_time': load_time,
                'models_loaded': ['page_detector_integrated', 'page_detector_hybrid', 'ocr_thread_pool'],
                'memory_before': memory_before,
                'memory_after': memory_after,
                'memory_delta': memory_delta,
                'model_times': {
                    'page_detector_integrated': time1,
                    'page_detector_hybrid': time2,
                    'ocr_thread_pool': time3
                }
            }
        
        # 打印结果
        print(f"\n总加载时间: {result['total_time']:.2f}秒")
        print(f"内存占用: {self.format_memory(result['memory_delta'])}")
        print(f"已加载模型: {', '.join(result['models_loaded'])}")
        
        if result['model_times']:
            print("\n各模型加载时间:")
            for model_name, model_time in result['model_times'].items():
                print(f"  - {model_name}: {model_time:.2f}秒")
        
        self.results['model_loading'] = result
        return result
    
    def benchmark_single_account(self) -> Dict[str, Any]:
        """基准测试：单账号处理时间"""
        print("\n" + "=" * 60)
        print("基准测试：单账号处理时间")
        print("=" * 60)
        
        if self.use_model_manager:
            # 使用ModelManager（模型已预加载）
            print("使用ModelManager（模型已预加载）...")
            
            memory_before = self.measure_memory()
            start_time = time.time()
            
            # 模拟创建一个自动化实例（不实际执行任务）
            manager = ModelManager.get_instance()
            
            # 获取模型（应该很快，因为已经加载）
            t1 = time.time()
            detector1 = manager.get_page_detector_integrated()
            get_time1 = time.time() - t1
            
            t2 = time.time()
            detector2 = manager.get_page_detector_hybrid()
            get_time2 = time.time() - t2
            
            t3 = time.time()
            ocr_pool = manager.get_ocr_thread_pool()
            get_time3 = time.time() - t3
            
            total_time = time.time() - start_time
            memory_after = self.measure_memory()
            memory_delta = memory_after - memory_before
            
            result = {
                'total_time': total_time,
                'model_access_time': get_time1 + get_time2 + get_time3,
                'model_loading_time': 0,  # 已预加载
                'memory_delta': memory_delta
            }
        else:
            # 不使用ModelManager（每次都重新加载）
            print("不使用ModelManager（每次重新加载）...")
            
            memory_before = self.measure_memory()
            start_time = time.time()
            
            # 模拟每次都重新加载模型
            from src.page_detector_integrated import PageDetectorIntegrated
            from src.page_detector_hybrid_optimized import PageDetectorHybridOptimized
            from src.ocr_thread_pool import OCRThreadPool
            
            adb = ADBBridge()
            
            # 加载模型（每个账号都要做这个）
            load_start = time.time()
            detector1 = PageDetectorIntegrated(adb, log_callback=None)
            detector2 = PageDetectorHybridOptimized(adb, log_callback=None)
            ocr_pool = OCRThreadPool(thread_count=4)
            load_time = time.time() - load_start
            
            total_time = time.time() - start_time
            memory_after = self.measure_memory()
            memory_delta = memory_after - memory_before
            
            result = {
                'total_time': total_time,
                'model_access_time': 0,
                'model_loading_time': load_time,
                'memory_delta': memory_delta
            }
        
        # 打印结果
        print(f"\n单账号处理时间: {result['total_time']:.2f}秒")
        print(f"  - 模型加载时间: {result['model_loading_time']:.2f}秒")
        print(f"  - 模型访问时间: {result['model_access_time']:.4f}秒")
        print(f"  - 内存增量: {self.format_memory(result['memory_delta'])}")
        
        self.results['single_account'] = result
        return result
    
    def benchmark_multiple_accounts(self, num_accounts: int = 30) -> Dict[str, Any]:
        """基准测试：多账号处理时间
        
        Args:
            num_accounts: 账号数量（默认30）
        """
        print("\n" + "=" * 60)
        print(f"基准测试：{num_accounts}账号处理时间")
        print("=" * 60)
        
        memory_before = self.measure_memory()
        start_time = time.time()
        
        account_times = []
        
        if self.use_model_manager:
            # 使用ModelManager（所有账号共享模型）
            print(f"使用ModelManager处理{num_accounts}个账号...")
            
            manager = ModelManager.get_instance()
            
            for i in range(num_accounts):
                account_start = time.time()
                
                # 模拟获取模型（应该很快）
                detector1 = manager.get_page_detector_integrated()
                detector2 = manager.get_page_detector_hybrid()
                ocr_pool = manager.get_ocr_thread_pool()
                
                account_time = time.time() - account_start
                account_times.append(account_time)
                
                if (i + 1) % 10 == 0:
                    print(f"  已处理 {i + 1}/{num_accounts} 个账号")
        else:
            # 不使用ModelManager（每个账号都重新加载）
            print(f"不使用ModelManager处理{num_accounts}个账号...")
            
            from src.page_detector_integrated import PageDetectorIntegrated
            from src.page_detector_hybrid_optimized import PageDetectorHybridOptimized
            from src.ocr_thread_pool import OCRThreadPool
            
            adb = ADBBridge()
            
            for i in range(num_accounts):
                account_start = time.time()
                
                # 模拟每个账号都重新加载模型
                detector1 = PageDetectorIntegrated(adb, log_callback=None)
                detector2 = PageDetectorHybridOptimized(adb, log_callback=None)
                ocr_pool = OCRThreadPool(thread_count=4)
                
                account_time = time.time() - account_start
                account_times.append(account_time)
                
                if (i + 1) % 10 == 0:
                    print(f"  已处理 {i + 1}/{num_accounts} 个账号")
        
        total_time = time.time() - start_time
        memory_after = self.measure_memory()
        memory_delta = memory_after - memory_before
        
        avg_time = sum(account_times) / len(account_times)
        min_time = min(account_times)
        max_time = max(account_times)
        
        result = {
            'num_accounts': num_accounts,
            'total_time': total_time,
            'avg_time_per_account': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'memory_delta': memory_delta,
            'account_times': account_times
        }
        
        # 打印结果
        print(f"\n{num_accounts}账号总时间: {result['total_time']:.2f}秒")
        print(f"  - 平均每账号: {result['avg_time_per_account']:.2f}秒")
        print(f"  - 最快: {result['min_time']:.2f}秒")
        print(f"  - 最慢: {result['max_time']:.2f}秒")
        print(f"  - 内存增量: {self.format_memory(result['memory_delta'])}")
        
        self.results['multiple_accounts'] = result
        return result
    
    def benchmark_memory_usage(self) -> Dict[str, Any]:
        """基准测试：内存占用"""
        print("\n" + "=" * 60)
        print("基准测试：内存占用")
        print("=" * 60)
        
        # 初始内存
        memory_initial = self.measure_memory()
        print(f"初始内存: {self.format_memory(memory_initial)}")
        
        # 加载模型后的内存
        if self.use_model_manager:
            manager = ModelManager.get_instance()
            if not manager.is_initialized():
                adb = ADBBridge()
                manager.initialize_all_models(adb, log_callback=None)
        
        memory_after_loading = self.measure_memory()
        loading_delta = memory_after_loading - memory_initial
        print(f"加载模型后: {self.format_memory(memory_after_loading)} (+{self.format_memory(loading_delta)})")
        
        # 模拟处理10个账号后的内存
        if self.use_model_manager:
            manager = ModelManager.get_instance()
            for i in range(10):
                detector1 = manager.get_page_detector_integrated()
                detector2 = manager.get_page_detector_hybrid()
                ocr_pool = manager.get_ocr_thread_pool()
        else:
            from src.page_detector_integrated import PageDetectorIntegrated
            from src.page_detector_hybrid_optimized import PageDetectorHybridOptimized
            from src.ocr_thread_pool import OCRThreadPool
            
            adb = ADBBridge()
            for i in range(10):
                detector1 = PageDetectorIntegrated(adb, log_callback=None)
                detector2 = PageDetectorHybridOptimized(adb, log_callback=None)
                ocr_pool = OCRThreadPool(thread_count=4)
        
        memory_after_accounts = self.measure_memory()
        accounts_delta = memory_after_accounts - memory_after_loading
        print(f"处理10账号后: {self.format_memory(memory_after_accounts)} (+{self.format_memory(accounts_delta)})")
        
        result = {
            'memory_initial': memory_initial,
            'memory_after_loading': memory_after_loading,
            'memory_after_accounts': memory_after_accounts,
            'loading_delta': loading_delta,
            'accounts_delta': accounts_delta,
            'total_delta': memory_after_accounts - memory_initial
        }
        
        self.results['memory'] = result
        return result
    
    def run_all_benchmarks(self):
        """运行所有基准测试"""
        print("\n" + "=" * 60)
        print(f"性能基准测试 - {'使用ModelManager' if self.use_model_manager else '不使用ModelManager'}")
        print("=" * 60)
        
        # 1. 模型加载时间
        self.benchmark_model_loading()
        
        # 2. 单账号处理时间
        self.benchmark_single_account()
        
        # 3. 多账号处理时间
        self.benchmark_multiple_accounts(num_accounts=30)
        
        # 4. 内存占用
        self.benchmark_memory_usage()
        
        # 保存结果
        self.save_results()
        
        # 打印总结
        self.print_summary()
    
    def save_results(self):
        """保存测试结果到文件"""
        output_dir = Path("benchmark_results")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "with_manager" if self.use_model_manager else "without_manager"
        filename = output_dir / f"benchmark_{mode}_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存到: {filename}")
    
    def print_summary(self):
        """打印测试总结"""
        print("\n" + "=" * 60)
        print("测试总结")
        print("=" * 60)
        
        if 'model_loading' in self.results and self.results['model_loading']:
            ml = self.results['model_loading']
            print(f"\n模型加载:")
            print(f"  总时间: {ml['total_time']:.2f}秒")
            print(f"  内存占用: {self.format_memory(ml['memory_delta'])}")
        
        if 'single_account' in self.results and self.results['single_account']:
            sa = self.results['single_account']
            print(f"\n单账号处理:")
            print(f"  总时间: {sa['total_time']:.2f}秒")
            print(f"  模型加载: {sa['model_loading_time']:.2f}秒")
            print(f"  内存增量: {self.format_memory(sa['memory_delta'])}")
        
        if 'multiple_accounts' in self.results and self.results['multiple_accounts']:
            ma = self.results['multiple_accounts']
            print(f"\n{ma['num_accounts']}账号处理:")
            print(f"  总时间: {ma['total_time']:.2f}秒")
            print(f"  平均每账号: {ma['avg_time_per_account']:.2f}秒")
            print(f"  内存增量: {self.format_memory(ma['memory_delta'])}")
        
        if 'memory' in self.results and self.results['memory']:
            mem = self.results['memory']
            print(f"\n内存占用:")
            print(f"  初始: {self.format_memory(mem['memory_initial'])}")
            print(f"  加载后: {self.format_memory(mem['memory_after_loading'])} (+{self.format_memory(mem['loading_delta'])})")
            print(f"  处理后: {self.format_memory(mem['memory_after_accounts'])} (+{self.format_memory(mem['accounts_delta'])})")


def compare_results():
    """对比优化前后的结果"""
    print("\n" + "=" * 60)
    print("对比优化前后的性能")
    print("=" * 60)
    
    results_dir = Path("benchmark_results")
    if not results_dir.exists():
        print("错误: 未找到测试结果目录")
        return
    
    # 查找最新的两个结果文件
    with_manager_files = sorted(results_dir.glob("benchmark_with_manager_*.json"), reverse=True)
    without_manager_files = sorted(results_dir.glob("benchmark_without_manager_*.json"), reverse=True)
    
    if not with_manager_files or not without_manager_files:
        print("错误: 需要先运行两种模式的测试")
        print("  python benchmark_model_manager.py --mode with_manager")
        print("  python benchmark_model_manager.py --mode without_manager")
        return
    
    # 加载结果
    with open(with_manager_files[0], 'r', encoding='utf-8') as f:
        with_manager = json.load(f)
    
    with open(without_manager_files[0], 'r', encoding='utf-8') as f:
        without_manager = json.load(f)
    
    # 对比模型加载时间
    if 'model_loading' in with_manager and 'model_loading' in without_manager:
        wm_time = with_manager['model_loading']['total_time']
        wo_time = without_manager['model_loading']['total_time']
        diff = wo_time - wm_time
        percent = (diff / wo_time * 100) if wo_time > 0 else 0
        
        print(f"\n模型加载时间:")
        print(f"  优化前: {wo_time:.2f}秒")
        print(f"  优化后: {wm_time:.2f}秒")
        print(f"  改善: {diff:.2f}秒 ({percent:+.1f}%)")
    
    # 对比单账号处理时间
    if 'single_account' in with_manager and 'single_account' in without_manager:
        wm_time = with_manager['single_account']['total_time']
        wo_time = without_manager['single_account']['total_time']
        diff = wo_time - wm_time
        percent = (diff / wo_time * 100) if wo_time > 0 else 0
        
        print(f"\n单账号处理时间:")
        print(f"  优化前: {wo_time:.2f}秒")
        print(f"  优化后: {wm_time:.2f}秒")
        print(f"  改善: {diff:.2f}秒 ({percent:+.1f}%)")
    
    # 对比30账号总时间
    if 'multiple_accounts' in with_manager and 'multiple_accounts' in without_manager:
        wm_time = with_manager['multiple_accounts']['total_time']
        wo_time = without_manager['multiple_accounts']['total_time']
        diff = wo_time - wm_time
        percent = (diff / wo_time * 100) if wo_time > 0 else 0
        
        print(f"\n30账号总时间:")
        print(f"  优化前: {wo_time:.2f}秒")
        print(f"  优化后: {wm_time:.2f}秒")
        print(f"  改善: {diff:.2f}秒 ({percent:+.1f}%)")
    
    # 对比内存占用
    if 'memory' in with_manager and 'memory' in without_manager:
        wm_mem = with_manager['memory']['total_delta']
        wo_mem = without_manager['memory']['total_delta']
        diff = wo_mem - wm_mem
        percent = (diff / wo_mem * 100) if wo_mem > 0 else 0
        
        print(f"\n内存占用:")
        print(f"  优化前: {wo_mem / 1024 / 1024:.1f}MB")
        print(f"  优化后: {wm_mem / 1024 / 1024:.1f}MB")
        print(f"  改善: {diff / 1024 / 1024:.1f}MB ({percent:+.1f}%)")
    
    print("\n" + "=" * 60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='ModelManager性能基准测试')
    parser.add_argument(
        '--mode',
        choices=['with_manager', 'without_manager', 'compare'],
        default='with_manager',
        help='测试模式: with_manager(使用ModelManager), without_manager(不使用), compare(对比结果)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'compare':
        compare_results()
    else:
        use_manager = (args.mode == 'with_manager')
        benchmark = PerformanceBenchmark(use_model_manager=use_manager)
        benchmark.run_all_benchmarks()


if __name__ == '__main__':
    main()
