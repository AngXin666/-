"""
属性测试：模型实例复用

Feature: model-singleton-optimization
Property 3: 模型实例复用

For any 组件在任意时刻请求同一类型的模型，返回的应该是完全相同的对象实例（相同的内存地址）。

Validates: Requirements 1.4, 2.3, 3.3, 4.3
"""

import pytest
from hypothesis import given, strategies as st, settings
import threading
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.model_manager import ModelManager


class TestPropertyModelInstanceReuse:
    """模型实例复用属性测试"""
    
    def setup_method(self):
        """每个测试前重置单例"""
        # 重置单例状态（仅用于测试）
        ModelManager._instance = None
        ModelManager._initialized = False
    
    @given(
        st.lists(
            st.sampled_from(['page_detector_integrated', 'page_detector_hybrid', 'ocr_thread_pool']),
            min_size=1,
            max_size=50
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_model_instance_reuse_sequential(self, model_requests):
        """
        属性测试：模型实例复用（顺序请求）
        
        For any 组件在任意时刻请求同一类型的模型，返回的应该是完全相同的对象实例
        
        测试策略：
        1. 生成随机的模型请求序列（1-50个请求）
        2. 每个请求可能是三种模型之一
        3. 验证同一类型的模型返回相同的实例ID
        4. 验证不同类型的模型返回不同的实例ID
        """
        manager = ModelManager.get_instance()
        
        # 模拟已初始化的模型（使用mock对象）
        class MockModel:
            def __init__(self, name):
                self.name = name
        
        # 预先"加载"模型（模拟初始化完成）
        manager._models = {
            'page_detector_integrated': MockModel('integrated'),
            'page_detector_hybrid': MockModel('hybrid'),
            'ocr_thread_pool': MockModel('ocr')
        }
        
        # 记录每种模型的第一个实例ID
        first_instances = {}
        
        for model_name in model_requests:
            # 根据模型名称获取实例
            if model_name == 'page_detector_integrated':
                instance = manager.get_page_detector_integrated()
            elif model_name == 'page_detector_hybrid':
                instance = manager.get_page_detector_hybrid()
            else:  # ocr_thread_pool
                instance = manager.get_ocr_thread_pool()
            
            instance_id = id(instance)
            
            # 如果是第一次请求这个模型，记录ID
            if model_name not in first_instances:
                first_instances[model_name] = instance_id
            else:
                # 验证返回的是同一个实例
                assert instance_id == first_instances[model_name], \
                    f"模型 {model_name} 应该返回相同的实例，但得到了不同的ID"
    
    @given(
        st.lists(
            st.tuples(
                st.sampled_from(['page_detector_integrated', 'page_detector_hybrid', 'ocr_thread_pool']),
                st.integers(min_value=1, max_value=10)  # 每个模型请求的次数
            ),
            min_size=1,
            max_size=10
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_model_instance_reuse_concurrent(self, model_request_patterns):
        """
        属性测试：模型实例复用（并发请求）
        
        For any 并发的模型请求，同一类型的模型应该返回相同的实例
        
        测试策略：
        1. 生成随机的并发请求模式
        2. 每个模式指定模型类型和请求次数
        3. 多个线程并发请求同一模型
        4. 验证所有线程获取的是同一个实例
        """
        manager = ModelManager.get_instance()
        
        # 模拟已初始化的模型
        class MockModel:
            def __init__(self, name):
                self.name = name
        
        manager._models = {
            'page_detector_integrated': MockModel('integrated'),
            'page_detector_hybrid': MockModel('hybrid'),
            'ocr_thread_pool': MockModel('ocr')
        }
        
        # 记录每个线程获取的实例ID
        results = {
            'page_detector_integrated': [],
            'page_detector_hybrid': [],
            'ocr_thread_pool': []
        }
        errors = []
        lock = threading.Lock()
        
        def request_model(model_name):
            """线程函数：请求模型"""
            try:
                if model_name == 'page_detector_integrated':
                    instance = manager.get_page_detector_integrated()
                elif model_name == 'page_detector_hybrid':
                    instance = manager.get_page_detector_hybrid()
                else:  # ocr_thread_pool
                    instance = manager.get_ocr_thread_pool()
                
                with lock:
                    results[model_name].append(id(instance))
            except Exception as e:
                with lock:
                    errors.append(e)
        
        # 创建并启动所有线程
        threads = []
        for model_name, num_requests in model_request_patterns:
            for _ in range(num_requests):
                thread = threading.Thread(target=request_model, args=(model_name,))
                threads.append(thread)
                thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证：没有错误
        assert len(errors) == 0, f"并发请求模型时出现错误: {errors}"
        
        # 验证：每种模型的所有请求返回相同的实例
        for model_name, instance_ids in results.items():
            if len(instance_ids) > 0:
                unique_ids = set(instance_ids)
                assert len(unique_ids) == 1, \
                    f"模型 {model_name} 的 {len(instance_ids)} 个并发请求应该返回相同的实例，" \
                    f"但得到了 {len(unique_ids)} 个不同的实例"
    
    @given(
        st.lists(
            st.sampled_from(['page_detector_integrated', 'page_detector_hybrid', 'ocr_thread_pool']),
            min_size=2,
            max_size=20
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_different_models_different_instances(self, model_requests):
        """
        属性测试：不同模型返回不同实例
        
        For any 不同类型的模型请求，应该返回不同的实例
        
        测试策略：
        1. 生成包含多种模型类型的请求序列
        2. 验证不同类型的模型返回不同的实例
        3. 验证相同类型的模型返回相同的实例
        """
        manager = ModelManager.get_instance()
        
        # 模拟已初始化的模型
        class MockModel:
            def __init__(self, name):
                self.name = name
        
        manager._models = {
            'page_detector_integrated': MockModel('integrated'),
            'page_detector_hybrid': MockModel('hybrid'),
            'ocr_thread_pool': MockModel('ocr')
        }
        
        # 收集所有实例
        instances = {}
        
        for model_name in model_requests:
            if model_name == 'page_detector_integrated':
                instance = manager.get_page_detector_integrated()
            elif model_name == 'page_detector_hybrid':
                instance = manager.get_page_detector_hybrid()
            else:  # ocr_thread_pool
                instance = manager.get_ocr_thread_pool()
            
            if model_name not in instances:
                instances[model_name] = instance
        
        # 如果有多种模型类型，验证它们是不同的实例
        if len(instances) > 1:
            instance_ids = [id(inst) for inst in instances.values()]
            unique_ids = set(instance_ids)
            
            assert len(unique_ids) == len(instances), \
                f"不同类型的模型应该返回不同的实例，" \
                f"但 {len(instances)} 种模型只有 {len(unique_ids)} 个唯一实例"
    
    def test_model_instance_reuse_across_components(self):
        """
        单元测试：跨组件的模型实例复用
        
        验证多个组件获取同一模型时，返回相同的实例
        """
        manager = ModelManager.get_instance()
        
        # 模拟已初始化的模型
        class MockModel:
            def __init__(self, name):
                self.name = name
        
        manager._models = {
            'page_detector_integrated': MockModel('integrated'),
            'page_detector_hybrid': MockModel('hybrid'),
            'ocr_thread_pool': MockModel('ocr')
        }
        
        # 模拟多个组件获取模型
        component1_integrated = manager.get_page_detector_integrated()
        component2_integrated = manager.get_page_detector_integrated()
        component3_integrated = manager.get_page_detector_integrated()
        
        component1_hybrid = manager.get_page_detector_hybrid()
        component2_hybrid = manager.get_page_detector_hybrid()
        
        component1_ocr = manager.get_ocr_thread_pool()
        component2_ocr = manager.get_ocr_thread_pool()
        component3_ocr = manager.get_ocr_thread_pool()
        
        # 验证：同一类型的模型返回相同实例
        assert component1_integrated is component2_integrated
        assert component2_integrated is component3_integrated
        
        assert component1_hybrid is component2_hybrid
        
        assert component1_ocr is component2_ocr
        assert component2_ocr is component3_ocr
        
        # 验证：不同类型的模型返回不同实例
        assert component1_integrated is not component1_hybrid
        assert component1_integrated is not component1_ocr
        assert component1_hybrid is not component1_ocr
    
    def test_model_instance_reuse_memory_efficiency(self):
        """
        单元测试：模型实例复用的内存效率
        
        验证多次获取模型不会创建新的实例，从而节省内存
        """
        manager = ModelManager.get_instance()
        
        # 模拟已初始化的模型
        class MockModel:
            def __init__(self, name):
                self.name = name
                self.data = [0] * 1000  # 模拟占用内存
        
        manager._models = {
            'page_detector_integrated': MockModel('integrated'),
            'page_detector_hybrid': MockModel('hybrid'),
            'ocr_thread_pool': MockModel('ocr')
        }
        
        # 记录初始实例ID
        initial_integrated_id = id(manager.get_page_detector_integrated())
        initial_hybrid_id = id(manager.get_page_detector_hybrid())
        initial_ocr_id = id(manager.get_ocr_thread_pool())
        
        # 多次获取模型（模拟30个账号的场景）
        for _ in range(30):
            integrated = manager.get_page_detector_integrated()
            hybrid = manager.get_page_detector_hybrid()
            ocr = manager.get_ocr_thread_pool()
            
            # 验证：每次获取的都是同一个实例
            assert id(integrated) == initial_integrated_id
            assert id(hybrid) == initial_hybrid_id
            assert id(ocr) == initial_ocr_id
        
        # 验证：模型字典中只有3个实例
        assert len(manager._models) == 3
    
    def test_model_instance_reuse_with_error_handling(self):
        """
        单元测试：错误处理不影响实例复用
        
        验证即使某些请求失败，成功的请求仍然返回相同的实例
        """
        manager = ModelManager.get_instance()
        
        # 模拟已初始化的模型
        class MockModel:
            def __init__(self, name):
                self.name = name
        
        # 只初始化部分模型
        manager._models = {
            'page_detector_integrated': MockModel('integrated'),
            'page_detector_hybrid': MockModel('hybrid')
            # 故意不初始化 ocr_thread_pool
        }
        
        # 成功的请求应该返回相同实例
        instance1 = manager.get_page_detector_integrated()
        instance2 = manager.get_page_detector_integrated()
        assert instance1 is instance2
        
        # 失败的请求应该抛出异常
        with pytest.raises(RuntimeError, match="未初始化"):
            manager.get_ocr_thread_pool()
        
        # 失败后，成功的请求仍然返回相同实例
        instance3 = manager.get_page_detector_integrated()
        assert instance1 is instance3


if __name__ == '__main__':
    # 运行属性测试
    pytest.main([__file__, '-v', '--tb=short'])
