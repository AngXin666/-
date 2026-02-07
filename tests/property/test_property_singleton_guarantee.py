"""
属性测试：单例模式保证

Feature: model-singleton-optimization
Property 1: 单例模式保证

For any 程序执行过程中的任意时刻，每种模型类型（PageDetectorIntegrated、
PageDetectorHybridOptimized、OCRThreadPool）应该只有一个实例被创建和加载。

Validates: Requirements 1.2
"""

import pytest
from hypothesis import given, strategies as st, settings
import threading
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.model_manager import ModelManager


class TestPropertySingletonGuarantee:
    """单例模式保证属性测试"""
    
    def setup_method(self):
        """每个测试前重置单例"""
        # 重置单例状态（仅用于测试）
        ModelManager._instance = None
        ModelManager._initialized = False
    
    @given(st.integers(min_value=1, max_value=100))
    @settings(max_examples=100, deadline=None)
    def test_property_singleton_guarantee_sequential(self, num_calls):
        """
        属性测试：单例模式保证（顺序调用）
        
        For any 程序执行过程中的任意时刻，ModelManager应该只有一个实例
        
        测试策略：
        1. 生成随机数量的get_instance()调用（1-100次）
        2. 验证所有调用返回的是同一个对象实例
        3. 验证对象ID完全相同
        """
        # 多次获取实例
        instances = [ModelManager.get_instance() for _ in range(num_calls)]
        
        # 验证所有实例都是同一个对象
        assert all(inst is instances[0] for inst in instances), \
            "所有get_instance()调用应该返回同一个对象"
        
        # 验证所有实例的ID相同
        instance_ids = [id(inst) for inst in instances]
        assert len(set(instance_ids)) == 1, \
            f"所有实例ID应该相同，但得到了 {len(set(instance_ids))} 个不同的ID"
    
    @given(st.integers(min_value=2, max_value=50))
    @settings(max_examples=100, deadline=None)
    def test_property_singleton_guarantee_concurrent(self, num_threads):
        """
        属性测试：单例模式保证（并发调用）
        
        For any 并发的get_instance()调用，应该返回同一个实例
        
        测试策略：
        1. 生成随机数量的并发线程（2-50个）
        2. 每个线程调用get_instance()
        3. 验证所有线程获取的是同一个实例
        4. 验证没有竞态条件
        """
        results = []
        errors = []
        
        def get_instance_thread():
            """线程函数：获取ModelManager实例"""
            try:
                instance = ModelManager.get_instance()
                results.append(id(instance))
            except Exception as e:
                errors.append(e)
        
        # 创建多个线程并发获取实例
        threads = [threading.Thread(target=get_instance_thread) for _ in range(num_threads)]
        
        # 启动所有线程
        for t in threads:
            t.start()
        
        # 等待所有线程完成
        for t in threads:
            t.join()
        
        # 验证：没有错误
        assert len(errors) == 0, f"并发获取实例时出现错误: {errors}"
        
        # 验证：所有线程获取的实例数量正确
        assert len(results) == num_threads, \
            f"期望 {num_threads} 个结果，但得到 {len(results)} 个"
        
        # 验证：所有实例ID相同（只有一个唯一ID）
        unique_ids = set(results)
        assert len(unique_ids) == 1, \
            f"所有线程应该获取同一个实例，但得到了 {len(unique_ids)} 个不同的实例"
    
    @given(
        st.lists(
            st.booleans(),
            min_size=1,
            max_size=20
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_singleton_guarantee_mixed_access(self, access_pattern):
        """
        属性测试：单例模式保证（混合访问模式）
        
        For any 混合的访问模式（get_instance和直接构造），应该返回同一个实例
        
        测试策略：
        1. 生成随机的访问模式（True=get_instance, False=直接构造）
        2. 验证所有方式获取的都是同一个实例
        3. 验证单例模式不会被绕过
        """
        instances = []
        
        for use_get_instance in access_pattern:
            if use_get_instance:
                # 使用get_instance()方法
                instance = ModelManager.get_instance()
            else:
                # 尝试直接构造（应该返回同一个实例）
                instance = ModelManager()
            
            instances.append(instance)
        
        # 验证所有实例都是同一个对象
        assert all(inst is instances[0] for inst in instances), \
            "无论使用get_instance()还是直接构造，都应该返回同一个实例"
        
        # 验证所有实例的ID相同
        instance_ids = [id(inst) for inst in instances]
        assert len(set(instance_ids)) == 1, \
            f"所有实例ID应该相同，但得到了 {len(set(instance_ids))} 个不同的ID"
    
    def test_singleton_initialization_once(self):
        """
        单元测试：单例只初始化一次
        
        验证__init__方法只执行一次，即使多次调用get_instance()
        """
        # 第一次获取实例
        instance1 = ModelManager.get_instance()
        assert instance1._initialized is True
        
        # 记录初始化状态的ID（而不是副本）
        initial_models_id = id(instance1._models)
        initial_config_id = id(instance1._config)
        
        # 第二次获取实例
        instance2 = ModelManager.get_instance()
        
        # 验证是同一个实例
        assert instance1 is instance2
        
        # 验证内部状态对象没有被重新创建（ID相同）
        assert id(instance2._models) == initial_models_id, \
            "第二次获取实例时，_models对象不应该被重新创建"
        assert id(instance2._config) == initial_config_id, \
            "第二次获取实例时，_config对象不应该被重新创建"
    
    def test_singleton_thread_safety_stress(self):
        """
        压力测试：高并发下的单例线程安全
        
        使用大量线程同时访问，验证单例模式的线程安全性
        """
        num_threads = 100
        results = []
        errors = []
        barrier = threading.Barrier(num_threads)  # 同步屏障，确保所有线程同时开始
        
        def stress_test_thread():
            """压力测试线程函数"""
            try:
                # 等待所有线程就绪
                barrier.wait()
                
                # 同时获取实例
                instance = ModelManager.get_instance()
                results.append(id(instance))
            except Exception as e:
                errors.append(e)
        
        # 创建大量线程
        threads = [threading.Thread(target=stress_test_thread) for _ in range(num_threads)]
        
        # 启动所有线程
        for t in threads:
            t.start()
        
        # 等待所有线程完成
        for t in threads:
            t.join()
        
        # 验证：没有错误
        assert len(errors) == 0, f"压力测试中出现错误: {errors}"
        
        # 验证：所有线程获取的是同一个实例
        assert len(results) == num_threads
        assert len(set(results)) == 1, \
            f"高并发下应该只有一个实例，但得到了 {len(set(results))} 个不同的实例"


if __name__ == '__main__':
    # 运行属性测试
    pytest.main([__file__, '-v', '--tb=short'])
