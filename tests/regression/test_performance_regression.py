"""
性能回归测试

确保ModelManager的性能改善持续，防止未来的代码变更导致性能退化。

测试内容:
1. 模型加载时间不应超过基准值
2. 单账号处理时间不应超过基准值
3. 内存占用不应超过基准值
4. 模型访问时间应保持在毫秒级

运行方法:
    pytest tests/regression/test_performance_regression.py -v
"""

import pytest
import time
import psutil
import os
import sys
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model_manager import ModelManager
from src.adb_bridge import ADBBridge


# 性能基准值（根据实际测试结果设定）
PERFORMANCE_BASELINES = {
    'model_loading_time': 15.0,  # 秒，所有模型加载时间不应超过15秒
    'model_access_time': 0.01,   # 秒，模型访问时间不应超过10ms
    'single_account_time': 1.0,  # 秒，单账号处理时间不应超过1秒
    'memory_per_model': 500 * 1024 * 1024,  # 字节，每个模型内存不应超过500MB
    'total_memory': 1500 * 1024 * 1024,  # 字节，总内存不应超过1.5GB
}

# 性能容忍度（允许的波动范围）
TOLERANCE = 1.2  # 允许20%的波动


class TestPerformanceRegression:
    """性能回归测试类"""
    
    @pytest.fixture(scope='class')
    def model_manager(self):
        """初始化ModelManager（类级别fixture，只初始化一次）"""
        # 重置单例
        ModelManager._instance = None
        ModelManager._initialized = False
        
        manager = ModelManager.get_instance()
        adb = ADBBridge()
        
        # 初始化所有模型
        manager.initialize_all_models(adb_bridge=adb, log_callback=None)
        
        yield manager
        
        # 清理
        manager.cleanup()
    
    def test_model_loading_time(self):
        """测试模型加载时间不应退化"""
        # 重置单例以测试加载时间
        ModelManager._instance = None
        ModelManager._initialized = False
        
        manager = ModelManager.get_instance()
        adb = ADBBridge()
        
        # 测量加载时间
        start_time = time.time()
        stats = manager.initialize_all_models(adb_bridge=adb, log_callback=None)
        load_time = time.time() - start_time
        
        baseline = PERFORMANCE_BASELINES['model_loading_time']
        max_allowed = baseline * TOLERANCE
        
        print(f"\n模型加载时间: {load_time:.2f}秒")
        print(f"基准值: {baseline:.2f}秒")
        print(f"最大允许: {max_allowed:.2f}秒")
        
        assert load_time <= max_allowed, (
            f"模型加载时间退化！"
            f"当前: {load_time:.2f}秒，"
            f"基准: {baseline:.2f}秒，"
            f"最大允许: {max_allowed:.2f}秒"
        )
        
        # 清理
        manager.cleanup()
    
    def test_model_access_time(self, model_manager):
        """测试模型访问时间应保持在毫秒级"""
        # 测试多次访问的平均时间
        access_times = []
        
        for _ in range(100):
            start_time = time.time()
            detector = model_manager.get_page_detector_integrated()
            access_time = time.time() - start_time
            access_times.append(access_time)
        
        avg_access_time = sum(access_times) / len(access_times)
        max_access_time = max(access_times)
        
        baseline = PERFORMANCE_BASELINES['model_access_time']
        max_allowed = baseline * TOLERANCE
        
        print(f"\n模型访问时间:")
        print(f"  平均: {avg_access_time * 1000:.2f}ms")
        print(f"  最大: {max_access_time * 1000:.2f}ms")
        print(f"  基准: {baseline * 1000:.2f}ms")
        print(f"  最大允许: {max_allowed * 1000:.2f}ms")
        
        assert avg_access_time <= max_allowed, (
            f"模型访问时间退化！"
            f"当前平均: {avg_access_time * 1000:.2f}ms，"
            f"基准: {baseline * 1000:.2f}ms，"
            f"最大允许: {max_allowed * 1000:.2f}ms"
        )
    
    def test_single_account_processing_time(self, model_manager):
        """测试单账号处理时间不应退化"""
        # 模拟单账号处理（获取所有模型）
        start_time = time.time()
        
        # 获取所有模型
        detector1 = model_manager.get_page_detector_integrated()
        detector2 = model_manager.get_page_detector_hybrid()
        ocr_pool = model_manager.get_ocr_thread_pool()
        
        # 验证模型可用
        assert detector1 is not None
        assert detector2 is not None
        assert ocr_pool is not None
        
        process_time = time.time() - start_time
        
        baseline = PERFORMANCE_BASELINES['single_account_time']
        max_allowed = baseline * TOLERANCE
        
        print(f"\n单账号处理时间: {process_time:.4f}秒")
        print(f"基准值: {baseline:.4f}秒")
        print(f"最大允许: {max_allowed:.4f}秒")
        
        assert process_time <= max_allowed, (
            f"单账号处理时间退化！"
            f"当前: {process_time:.4f}秒，"
            f"基准: {baseline:.4f}秒，"
            f"最大允许: {max_allowed:.4f}秒"
        )
    
    def test_memory_usage(self, model_manager):
        """测试内存占用不应退化"""
        process = psutil.Process(os.getpid())
        
        # 记录初始内存
        memory_before = process.memory_info().rss
        
        # 获取所有模型（应该不增加内存，因为已经加载）
        detector1 = model_manager.get_page_detector_integrated()
        detector2 = model_manager.get_page_detector_hybrid()
        ocr_pool = model_manager.get_ocr_thread_pool()
        
        # 记录使用后的内存
        memory_after = process.memory_info().rss
        memory_delta = memory_after - memory_before
        
        # 获取加载统计
        stats = model_manager.get_loading_stats()
        total_memory = stats.get('memory_after', 0) - stats.get('memory_before', 0)
        
        baseline = PERFORMANCE_BASELINES['total_memory']
        max_allowed = baseline * TOLERANCE
        
        print(f"\n内存使用:")
        print(f"  总内存占用: {total_memory / 1024 / 1024:.1f}MB")
        print(f"  访问增量: {memory_delta / 1024 / 1024:.1f}MB")
        print(f"  基准值: {baseline / 1024 / 1024:.1f}MB")
        print(f"  最大允许: {max_allowed / 1024 / 1024:.1f}MB")
        
        assert total_memory <= max_allowed, (
            f"内存占用退化！"
            f"当前: {total_memory / 1024 / 1024:.1f}MB，"
            f"基准: {baseline / 1024 / 1024:.1f}MB，"
            f"最大允许: {max_allowed / 1024 / 1024:.1f}MB"
        )
        
        # 验证访问模型不应显著增加内存（应该小于10MB）
        assert memory_delta < 10 * 1024 * 1024, (
            f"访问模型导致内存增加过多: {memory_delta / 1024 / 1024:.1f}MB"
        )
    
    def test_model_instance_reuse(self, model_manager):
        """测试模型实例复用（性能优化的核心）"""
        # 多次获取同一个模型
        instances = []
        access_times = []
        
        for _ in range(10):
            start_time = time.time()
            detector = model_manager.get_page_detector_integrated()
            access_time = time.time() - start_time
            
            instances.append(id(detector))
            access_times.append(access_time)
        
        # 验证所有实例都是同一个对象
        unique_instances = set(instances)
        assert len(unique_instances) == 1, (
            f"模型实例复用失败！发现 {len(unique_instances)} 个不同的实例"
        )
        
        # 验证访问时间稳定（标准差应该很小）
        import statistics
        avg_time = statistics.mean(access_times)
        std_time = statistics.stdev(access_times) if len(access_times) > 1 else 0
        
        print(f"\n模型实例复用:")
        print(f"  唯一实例数: {len(unique_instances)}")
        print(f"  平均访问时间: {avg_time * 1000:.2f}ms")
        print(f"  标准差: {std_time * 1000:.2f}ms")
        
        # 标准差应该很小（小于平均值的50%，或者访问时间非常快时忽略此检查）
        if avg_time > 0.001:  # 只有当平均时间大于1ms时才检查标准差
            assert std_time < avg_time * 0.5, (
                f"访问时间不稳定！标准差: {std_time * 1000:.2f}ms，"
                f"平均值: {avg_time * 1000:.2f}ms"
            )
        else:
            print("  [OK] 访问时间非常快，跳过标准差检查")
    
    def test_concurrent_access_performance(self, model_manager):
        """测试并发访问性能"""
        import threading
        
        access_times = []
        errors = []
        lock = threading.Lock()
        
        def access_model():
            try:
                start_time = time.time()
                detector = model_manager.get_page_detector_integrated()
                access_time = time.time() - start_time
                
                with lock:
                    access_times.append(access_time)
            except Exception as e:
                with lock:
                    errors.append(str(e))
        
        # 创建10个线程并发访问
        threads = [threading.Thread(target=access_model) for _ in range(10)]
        
        start_time = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        total_time = time.time() - start_time
        
        # 验证没有错误
        assert len(errors) == 0, f"并发访问出错: {errors}"
        
        # 验证所有访问都成功
        assert len(access_times) == 10, f"只有 {len(access_times)} 次访问成功"
        
        # 计算性能指标
        avg_access_time = sum(access_times) / len(access_times)
        max_access_time = max(access_times)
        
        baseline = PERFORMANCE_BASELINES['model_access_time']
        max_allowed = baseline * TOLERANCE * 2  # 并发访问允许更大的容忍度
        
        print(f"\n并发访问性能:")
        print(f"  总时间: {total_time:.4f}秒")
        print(f"  平均访问时间: {avg_access_time * 1000:.2f}ms")
        print(f"  最大访问时间: {max_access_time * 1000:.2f}ms")
        print(f"  基准值: {baseline * 1000:.2f}ms")
        print(f"  最大允许: {max_allowed * 1000:.2f}ms")
        
        assert avg_access_time <= max_allowed, (
            f"并发访问性能退化！"
            f"当前平均: {avg_access_time * 1000:.2f}ms，"
            f"基准: {baseline * 1000:.2f}ms，"
            f"最大允许: {max_allowed * 1000:.2f}ms"
        )
    
    def test_initialization_consistency(self):
        """测试初始化的一致性"""
        # 重置单例
        ModelManager._instance = None
        ModelManager._initialized = False
        
        manager = ModelManager.get_instance()
        adb = ADBBridge()
        
        # 第一次初始化
        start_time = time.time()
        stats1 = manager.initialize_all_models(adb_bridge=adb, log_callback=None)
        first_init_time = time.time() - start_time
        
        # 获取第一次加载的模型列表
        models_first = set(stats1['models_loaded'])
        
        # 第二次初始化
        start_time = time.time()
        stats2 = manager.initialize_all_models(adb_bridge=adb, log_callback=None)
        second_init_time = time.time() - start_time
        
        # 获取第二次加载的模型列表
        models_second = set(stats2['models_loaded'])
        
        print(f"\n初始化一致性:")
        print(f"  第一次初始化: {first_init_time:.2f}秒，加载 {len(models_first)} 个模型")
        print(f"  第二次初始化: {second_init_time:.2f}秒，加载 {len(models_second)} 个模型")
        
        # 验证两次初始化加载的模型一致
        assert models_first == models_second, (
            f"两次初始化加载的模型不一致！"
            f"第一次: {models_first}，"
            f"第二次: {models_second}"
        )
        
        # 验证初始化时间在合理范围内（不应超过基准值）
        baseline = PERFORMANCE_BASELINES['model_loading_time']
        max_allowed = baseline * TOLERANCE
        
        assert first_init_time <= max_allowed, (
            f"初始化时间超过基准！"
            f"当前: {first_init_time:.2f}秒，"
            f"最大允许: {max_allowed:.2f}秒"
        )
        
        # 清理
        manager.cleanup()


class TestPerformanceComparison:
    """性能对比测试（与优化前对比）"""
    
    def test_compare_with_baseline(self):
        """对比当前性能与基准性能"""
        print("\n" + "=" * 70)
        print("性能对比测试")
        print("=" * 70)
        
        # 重置单例
        ModelManager._instance = None
        ModelManager._initialized = False
        
        manager = ModelManager.get_instance()
        adb = ADBBridge()
        
        # 测量当前性能
        start_time = time.time()
        stats = manager.initialize_all_models(adb_bridge=adb, log_callback=None)
        load_time = time.time() - start_time
        
        memory_delta = stats.get('memory_after', 0) - stats.get('memory_before', 0)
        
        # 打印对比结果
        print(f"\n当前性能:")
        print(f"  模型加载时间: {load_time:.2f}秒")
        print(f"  内存占用: {memory_delta / 1024 / 1024:.1f}MB")
        
        print(f"\n基准性能:")
        print(f"  模型加载时间: {PERFORMANCE_BASELINES['model_loading_time']:.2f}秒")
        print(f"  内存占用: {PERFORMANCE_BASELINES['total_memory'] / 1024 / 1024:.1f}MB")
        
        # 计算改善百分比
        time_improvement = (PERFORMANCE_BASELINES['model_loading_time'] - load_time) / PERFORMANCE_BASELINES['model_loading_time'] * 100
        memory_improvement = (PERFORMANCE_BASELINES['total_memory'] - memory_delta) / PERFORMANCE_BASELINES['total_memory'] * 100
        
        print(f"\n性能改善:")
        print(f"  时间: {time_improvement:+.1f}%")
        print(f"  内存: {memory_improvement:+.1f}%")
        
        # 验证性能没有退化（允许小幅波动）
        assert load_time <= PERFORMANCE_BASELINES['model_loading_time'] * TOLERANCE, (
            f"性能退化！当前: {load_time:.2f}秒，基准: {PERFORMANCE_BASELINES['model_loading_time']:.2f}秒"
        )
        
        assert memory_delta <= PERFORMANCE_BASELINES['total_memory'] * TOLERANCE, (
            f"内存占用退化！当前: {memory_delta / 1024 / 1024:.1f}MB，"
            f"基准: {PERFORMANCE_BASELINES['total_memory'] / 1024 / 1024:.1f}MB"
        )
        
        # 清理
        manager.cleanup()


if __name__ == '__main__':
    # 运行测试
    pytest.main([__file__, '-v', '-s'])
