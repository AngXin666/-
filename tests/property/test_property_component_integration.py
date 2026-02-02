"""
属性测试：组件集成正确性

Feature: model-singleton-optimization
Property 7: 组件集成正确性

For any 程序执行过程，在ModelManager初始化完成后，系统不应该创建新的模型实例
（除了ModelManager管理的实例）。

Validates: Requirements 5.4
"""

import pytest
from hypothesis import given, strategies as st, settings
import threading
import sys
import os
from typing import List, Dict, Any
from unittest.mock import Mock, patch

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.model_manager import ModelManager


class MockModel:
    """模拟模型类，用于测试"""
    _instance_count = 0  # 类变量，跟踪创建的实例数量
    
    def __init__(self, name: str):
        self.name = name
        MockModel._instance_count += 1
        self.instance_id = MockModel._instance_count
    
    @classmethod
    def reset_count(cls):
        """重置实例计数器"""
        cls._instance_count = 0


class TestPropertyComponentIntegration:
    """组件集成正确性属性测试"""
    
    def setup_method(self):
        """每个测试前重置单例"""
        # 重置单例状态
        ModelManager._instance = None
        ModelManager._initialized = False
        
        # 重置模拟模型计数器
        MockModel.reset_count()
    
    @given(st.integers(min_value=1, max_value=50))
    @settings(max_examples=100, deadline=None)
    def test_property_no_new_instances_after_init(self, num_components):
        """
        属性测试：初始化后不创建新实例
        
        For any 数量的组件请求，在ModelManager初始化后，
        不应该创建新的模型实例
        
        测试策略：
        1. 初始化ModelManager并加载模型
        2. 记录初始的模型实例ID
        3. 模拟多个组件（1-50个）请求模型
        4. 验证所有组件获取的都是初始实例
        5. 验证没有创建新的模型实例
        """
        manager = ModelManager.get_instance()
        
        # 模拟初始化完成，加载模型
        manager._models = {
            'page_detector_integrated': MockModel('integrated'),
            'page_detector_hybrid': MockModel('hybrid'),
            'ocr_thread_pool': MockModel('ocr')
        }
        
        # 记录初始实例数量和ID
        initial_instance_count = MockModel._instance_count
        initial_integrated_id = id(manager._models['page_detector_integrated'])
        initial_hybrid_id = id(manager._models['page_detector_hybrid'])
        initial_ocr_id = id(manager._models['ocr_thread_pool'])
        
        # 模拟多个组件请求模型
        for component_id in range(num_components):
            # 每个组件随机请求不同的模型
            model_choice = component_id % 3
            
            if model_choice == 0:
                model = manager.get_page_detector_integrated()
                expected_id = initial_integrated_id
                model_name = 'integrated'
            elif model_choice == 1:
                model = manager.get_page_detector_hybrid()
                expected_id = initial_hybrid_id
                model_name = 'hybrid'
            else:
                model = manager.get_ocr_thread_pool()
                expected_id = initial_ocr_id
                model_name = 'ocr'
            
            # 验证：返回的是初始实例
            assert id(model) == expected_id, \
                f"组件 {component_id} 请求 {model_name} 时应该返回初始实例，" \
                f"但得到了不同的实例"
        
        # 验证：没有创建新的模型实例
        final_instance_count = MockModel._instance_count
        assert final_instance_count == initial_instance_count, \
            f"初始化后不应该创建新实例，但实例数从 {initial_instance_count} " \
            f"增加到了 {final_instance_count}"
    
    @given(
        st.integers(min_value=2, max_value=30),
        st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_concurrent_components_no_new_instances(
        self, num_components, accesses_per_component
    ):
        """
        属性测试：并发组件不创建新实例
        
        For any 并发的组件访问，在ModelManager初始化后，
        不应该创建新的模型实例
        
        测试策略：
        1. 初始化ModelManager并加载模型
        2. 记录初始的模型实例数量
        3. 创建多个并发组件（2-30个）
        4. 每个组件进行多次模型访问（1-10次）
        5. 验证没有创建新的模型实例
        """
        manager = ModelManager.get_instance()
        
        # 模拟初始化完成
        manager._models = {
            'page_detector_integrated': MockModel('integrated'),
            'page_detector_hybrid': MockModel('hybrid'),
            'ocr_thread_pool': MockModel('ocr')
        }
        
        # 记录初始实例数量
        initial_instance_count = MockModel._instance_count
        
        # 记录初始实例ID
        initial_ids = {
            'integrated': id(manager._models['page_detector_integrated']),
            'hybrid': id(manager._models['page_detector_hybrid']),
            'ocr': id(manager._models['ocr_thread_pool'])
        }
        
        errors = []
        results = []
        
        def component_thread(component_id: int):
            """模拟组件线程"""
            try:
                component_results = []
                
                for access_num in range(accesses_per_component):
                    # 循环访问三种模型
                    model_choice = (component_id + access_num) % 3
                    
                    if model_choice == 0:
                        model = manager.get_page_detector_integrated()
                        model_name = 'integrated'
                    elif model_choice == 1:
                        model = manager.get_page_detector_hybrid()
                        model_name = 'hybrid'
                    else:
                        model = manager.get_ocr_thread_pool()
                        model_name = 'ocr'
                    
                    component_results.append({
                        'model_name': model_name,
                        'model_id': id(model),
                        'expected_id': initial_ids[model_name]
                    })
                
                results.append({
                    'component_id': component_id,
                    'accesses': component_results
                })
            except Exception as e:
                errors.append({
                    'component_id': component_id,
                    'error': str(e)
                })
        
        # 创建并启动多个组件线程
        threads = [
            threading.Thread(target=component_thread, args=(i,))
            for i in range(num_components)
        ]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        # 验证：没有错误
        assert len(errors) == 0, \
            f"并发组件访问时出现错误: {errors}"
        
        # 验证：所有访问返回的都是初始实例
        for result in results:
            for access in result['accesses']:
                assert access['model_id'] == access['expected_id'], \
                    f"组件 {result['component_id']} 访问 {access['model_name']} " \
                    f"时应该返回初始实例"
        
        # 验证：没有创建新的模型实例
        final_instance_count = MockModel._instance_count
        assert final_instance_count == initial_instance_count, \
            f"并发访问后不应该创建新实例，但实例数从 {initial_instance_count} " \
            f"增加到了 {final_instance_count}"
    
    @given(
        st.lists(
            st.sampled_from(['page_detector_integrated', 'page_detector_hybrid', 'ocr_thread_pool']),
            min_size=1,
            max_size=100
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_component_access_pattern_no_new_instances(self, access_pattern):
        """
        属性测试：任意访问模式不创建新实例
        
        For any 组件访问模式，在ModelManager初始化后，
        不应该创建新的模型实例
        
        测试策略：
        1. 初始化ModelManager并加载模型
        2. 生成随机的访问模式（1-100次访问）
        3. 按照访问模式请求模型
        4. 验证没有创建新的模型实例
        5. 验证所有访问返回的都是初始实例
        """
        manager = ModelManager.get_instance()
        
        # 模拟初始化完成
        manager._models = {
            'page_detector_integrated': MockModel('integrated'),
            'page_detector_hybrid': MockModel('hybrid'),
            'ocr_thread_pool': MockModel('ocr')
        }
        
        # 记录初始状态
        initial_instance_count = MockModel._instance_count
        initial_ids = {
            'page_detector_integrated': id(manager._models['page_detector_integrated']),
            'page_detector_hybrid': id(manager._models['page_detector_hybrid']),
            'ocr_thread_pool': id(manager._models['ocr_thread_pool'])
        }
        
        # 按照访问模式请求模型
        for model_name in access_pattern:
            if model_name == 'page_detector_integrated':
                model = manager.get_page_detector_integrated()
            elif model_name == 'page_detector_hybrid':
                model = manager.get_page_detector_hybrid()
            else:  # ocr_thread_pool
                model = manager.get_ocr_thread_pool()
            
            # 验证：返回的是初始实例
            expected_id = initial_ids[model_name]
            assert id(model) == expected_id, \
                f"访问 {model_name} 时应该返回初始实例"
        
        # 验证：没有创建新的模型实例
        final_instance_count = MockModel._instance_count
        assert final_instance_count == initial_instance_count, \
            f"访问模式执行后不应该创建新实例，但实例数从 {initial_instance_count} " \
            f"增加到了 {final_instance_count}"
    
    def test_component_integration_no_direct_instantiation(self):
        """
        单元测试：组件不直接实例化模型
        
        验证组件通过ModelManager获取模型，而不是直接创建新实例
        """
        manager = ModelManager.get_instance()
        
        # 模拟初始化完成
        manager._models = {
            'page_detector_integrated': MockModel('integrated'),
            'page_detector_hybrid': MockModel('hybrid'),
            'ocr_thread_pool': MockModel('ocr')
        }
        
        # 记录初始实例数量
        initial_count = MockModel._instance_count
        
        # 模拟多个组件获取模型（正确方式）
        component1_integrated = manager.get_page_detector_integrated()
        component2_integrated = manager.get_page_detector_integrated()
        component3_hybrid = manager.get_page_detector_hybrid()
        component4_ocr = manager.get_ocr_thread_pool()
        
        # 验证：没有创建新实例
        assert MockModel._instance_count == initial_count, \
            "通过ModelManager获取模型不应该创建新实例"
        
        # 验证：相同类型的模型返回相同实例
        assert component1_integrated is component2_integrated, \
            "不同组件获取的相同类型模型应该是同一个实例"
    
    def test_component_integration_lifecycle(self):
        """
        单元测试：组件生命周期中的模型使用
        
        验证在组件的整个生命周期中，使用的都是同一个模型实例
        """
        manager = ModelManager.get_instance()
        
        # 模拟初始化完成
        manager._models = {
            'page_detector_integrated': MockModel('integrated'),
            'page_detector_hybrid': MockModel('hybrid'),
            'ocr_thread_pool': MockModel('ocr')
        }
        
        # 记录初始实例
        initial_integrated = manager.get_page_detector_integrated()
        initial_hybrid = manager.get_page_detector_hybrid()
        initial_ocr = manager.get_ocr_thread_pool()
        
        initial_count = MockModel._instance_count
        
        # 模拟组件在不同阶段获取模型
        # 阶段1：初始化
        init_integrated = manager.get_page_detector_integrated()
        
        # 阶段2：执行任务
        task_hybrid = manager.get_page_detector_hybrid()
        task_ocr = manager.get_ocr_thread_pool()
        
        # 阶段3：清理
        cleanup_integrated = manager.get_page_detector_integrated()
        
        # 验证：所有阶段使用的都是同一个实例
        assert init_integrated is initial_integrated
        assert task_hybrid is initial_hybrid
        assert task_ocr is initial_ocr
        assert cleanup_integrated is initial_integrated
        
        # 验证：没有创建新实例
        assert MockModel._instance_count == initial_count
    
    def test_multiple_automation_instances_share_models(self):
        """
        单元测试：多个自动化实例共享模型
        
        验证创建多个XimengAutomation实例时，它们共享相同的模型实例
        （模拟Orchestrator创建多个自动化实例的场景）
        """
        manager = ModelManager.get_instance()
        
        # 模拟初始化完成
        manager._models = {
            'page_detector_integrated': MockModel('integrated'),
            'page_detector_hybrid': MockModel('hybrid'),
            'ocr_thread_pool': MockModel('ocr')
        }
        
        # 记录初始实例
        initial_integrated = manager.get_page_detector_integrated()
        initial_hybrid = manager.get_page_detector_hybrid()
        initial_ocr = manager.get_ocr_thread_pool()
        
        initial_count = MockModel._instance_count
        
        # 模拟创建多个自动化实例（如30个账号）
        num_instances = 30
        automation_instances = []
        
        for i in range(num_instances):
            # 每个自动化实例获取模型
            instance_models = {
                'integrated': manager.get_page_detector_integrated(),
                'hybrid': manager.get_page_detector_hybrid(),
                'ocr': manager.get_ocr_thread_pool()
            }
            automation_instances.append(instance_models)
        
        # 验证：所有实例使用相同的模型
        for i, instance_models in enumerate(automation_instances):
            assert instance_models['integrated'] is initial_integrated, \
                f"自动化实例 {i} 应该使用共享的 integrated 模型"
            assert instance_models['hybrid'] is initial_hybrid, \
                f"自动化实例 {i} 应该使用共享的 hybrid 模型"
            assert instance_models['ocr'] is initial_ocr, \
                f"自动化实例 {i} 应该使用共享的 ocr 模型"
        
        # 验证：没有创建新实例
        assert MockModel._instance_count == initial_count, \
            f"创建 {num_instances} 个自动化实例不应该创建新的模型实例"
    
    def test_component_integration_with_error_recovery(self):
        """
        单元测试：错误恢复后仍使用相同实例
        
        验证即使某些操作失败，组件仍然使用相同的模型实例
        """
        manager = ModelManager.get_instance()
        
        # 模拟初始化完成
        manager._models = {
            'page_detector_integrated': MockModel('integrated'),
            'page_detector_hybrid': MockModel('hybrid'),
            'ocr_thread_pool': MockModel('ocr')
        }
        
        # 记录初始实例
        initial_integrated = manager.get_page_detector_integrated()
        initial_count = MockModel._instance_count
        
        # 模拟操作失败（但不影响模型实例）
        try:
            # 尝试访问不存在的模型
            manager.get_page_detector_integrated()
            # 模拟某些操作...
            pass
        except Exception:
            pass
        
        # 错误恢复后，再次获取模型
        recovered_integrated = manager.get_page_detector_integrated()
        
        # 验证：仍然是同一个实例
        assert recovered_integrated is initial_integrated, \
            "错误恢复后应该仍然使用相同的模型实例"
        
        # 验证：没有创建新实例
        assert MockModel._instance_count == initial_count
    
    def test_component_integration_memory_efficiency(self):
        """
        单元测试：组件集成的内存效率
        
        验证多个组件共享模型实例确实节省了内存
        """
        manager = ModelManager.get_instance()
        
        # 模拟初始化完成
        manager._models = {
            'page_detector_integrated': MockModel('integrated'),
            'page_detector_hybrid': MockModel('hybrid'),
            'ocr_thread_pool': MockModel('ocr')
        }
        
        initial_count = MockModel._instance_count
        
        # 模拟30个账号，每个账号访问所有模型
        num_accounts = 30
        for account_id in range(num_accounts):
            # 每个账号获取所有模型
            integrated = manager.get_page_detector_integrated()
            hybrid = manager.get_page_detector_hybrid()
            ocr = manager.get_ocr_thread_pool()
            
            # 模拟使用模型...
            pass
        
        # 验证：仍然只有3个模型实例
        assert MockModel._instance_count == initial_count, \
            f"30个账号共享模型，应该只有 {initial_count} 个实例，" \
            f"但实际有 {MockModel._instance_count} 个"
        
        # 如果不共享，应该有 30 * 3 = 90 个实例
        # 共享后只有 3 个实例，节省了 87 个实例的内存
        expected_without_sharing = num_accounts * 3
        memory_saved_instances = expected_without_sharing - initial_count
        
        print(f"\n内存效率统计:")
        print(f"  账号数量: {num_accounts}")
        print(f"  实际模型实例数: {initial_count}")
        print(f"  不共享时的实例数: {expected_without_sharing}")
        print(f"  节省的实例数: {memory_saved_instances}")
        print(f"  内存节省率: {memory_saved_instances/expected_without_sharing*100:.1f}%")


if __name__ == '__main__':
    # 运行属性测试
    pytest.main([__file__, '-v', '--tb=short', '-s'])
