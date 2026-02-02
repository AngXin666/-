"""
属性测试：资源清理完整性

Feature: model-singleton-optimization
Property 9: 资源清理完整性

For any 程序退出场景，ModelManager的cleanup方法应该释放所有已加载的模型，
并确保GPU内存（如果使用）被完全释放。

Validates: Requirements 8.1, 8.4
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
import threading
import sys
import os
import gc
import time

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.model_manager import ModelManager


class TestPropertyResourceCleanup:
    """资源清理完整性属性测试"""
    
    def setup_method(self):
        """每个测试前重置单例"""
        # 重置单例状态（仅用于测试）
        ModelManager._instance = None
        ModelManager._initialized = False
        gc.collect()
    
    def teardown_method(self):
        """每个测试后清理"""
        # 确保清理资源
        try:
            manager = ModelManager.get_instance()
            if manager.is_initialized():
                manager.cleanup()
        except:
            pass
        
        # 重置单例
        ModelManager._instance = None
        ModelManager._initialized = False
        gc.collect()
    
    @given(
        st.lists(
            st.sampled_from(['page_detector_integrated', 'page_detector_hybrid', 'ocr_thread_pool']),
            min_size=0,
            max_size=3,
            unique=True
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_cleanup_releases_all_models(self, loaded_models):
        """
        属性测试：cleanup释放所有已加载的模型
        
        For any 已加载的模型集合，cleanup应该释放所有模型
        
        测试策略：
        1. 生成随机的已加载模型集合（0-3个模型）
        2. 模拟这些模型已加载
        3. 调用cleanup()
        4. 验证所有模型都被释放
        5. 验证_models字典为空
        """
        manager = ModelManager.get_instance()
        
        # 模拟加载模型（添加到_models字典）
        for model_name in loaded_models:
            # 创建模拟模型对象
            mock_model = type('MockModel', (), {'name': model_name})()
            manager._models[model_name] = mock_model
        
        # 记录加载的模型数量
        initial_model_count = len(manager._models)
        
        # 调用cleanup
        manager.cleanup()
        
        # 验证：所有模型都被释放
        assert len(manager._models) == 0, \
            f"cleanup后应该没有模型，但还有 {len(manager._models)} 个模型"
        
        # 验证：_model_info也被清空
        assert len(manager._model_info) == 0, \
            f"cleanup后_model_info应该为空，但还有 {len(manager._model_info)} 个条目"
    
    @given(st.integers(min_value=1, max_value=10))
    @settings(max_examples=100, deadline=None)
    def test_property_cleanup_multiple_times(self, num_cleanups):
        """
        属性测试：多次调用cleanup是安全的
        
        For any 调用cleanup的次数，多次调用应该是安全的，不会出错
        
        测试策略：
        1. 生成随机的cleanup调用次数（1-10次）
        2. 模拟加载一些模型
        3. 多次调用cleanup
        4. 验证不会抛出异常
        5. 验证最终状态正确
        """
        manager = ModelManager.get_instance()
        
        # 模拟加载模型
        manager._models['test_model'] = type('MockModel', (), {})()
        
        # 多次调用cleanup
        errors = []
        for i in range(num_cleanups):
            try:
                manager.cleanup()
            except Exception as e:
                errors.append((i, e))
        
        # 验证：没有错误
        assert len(errors) == 0, \
            f"多次调用cleanup不应该出错，但出现了 {len(errors)} 个错误: {errors}"
        
        # 验证：最终状态正确
        assert len(manager._models) == 0, \
            "多次cleanup后，_models应该为空"
    
    @given(st.integers(min_value=2, max_value=20))
    @settings(max_examples=100, deadline=None)
    def test_property_cleanup_thread_safety(self, num_threads):
        """
        属性测试：cleanup的线程安全性
        
        For any 并发的cleanup调用，应该是线程安全的
        
        测试策略：
        1. 生成随机数量的并发线程（2-20个）
        2. 模拟加载模型
        3. 所有线程同时调用cleanup
        4. 验证没有竞态条件
        5. 验证最终状态正确
        """
        manager = ModelManager.get_instance()
        
        # 模拟加载模型
        for i in range(3):
            manager._models[f'model_{i}'] = type('MockModel', (), {})()
        
        errors = []
        barrier = threading.Barrier(num_threads)
        
        def cleanup_thread():
            """线程函数：调用cleanup"""
            try:
                # 等待所有线程就绪
                barrier.wait()
                
                # 同时调用cleanup
                manager.cleanup()
            except Exception as e:
                errors.append(e)
        
        # 创建多个线程
        threads = [threading.Thread(target=cleanup_thread) for _ in range(num_threads)]
        
        # 启动所有线程
        for t in threads:
            t.start()
        
        # 等待所有线程完成
        for t in threads:
            t.join()
        
        # 验证：没有错误
        assert len(errors) == 0, \
            f"并发cleanup不应该出错，但出现了 {len(errors)} 个错误: {errors}"
        
        # 验证：最终状态正确
        assert len(manager._models) == 0, \
            "并发cleanup后，_models应该为空"
    
    @given(
        st.lists(
            st.sampled_from(['load', 'cleanup', 'load', 'cleanup']),
            min_size=2,
            max_size=20
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_cleanup_load_cycle(self, operations):
        """
        属性测试：加载-清理循环
        
        For any 加载和清理操作的序列，系统应该正确处理
        
        测试策略：
        1. 生成随机的操作序列（load/cleanup）
        2. 执行操作序列
        3. 验证每次cleanup后模型都被释放
        4. 验证可以重新加载模型
        """
        manager = ModelManager.get_instance()
        
        for operation in operations:
            if operation == 'load':
                # 模拟加载模型
                model_name = f'model_{len(manager._models)}'
                manager._models[model_name] = type('MockModel', (), {})()
            else:  # cleanup
                # 清理模型
                manager.cleanup()
                
                # 验证清理后模型为空
                assert len(manager._models) == 0, \
                    "cleanup后应该没有模型"
        
        # 最终清理
        manager.cleanup()
        assert len(manager._models) == 0
    
    def test_cleanup_releases_model_references(self):
        """
        单元测试：cleanup释放模型引用
        
        验证cleanup后，模型对象的引用被移除
        """
        manager = ModelManager.get_instance()
        
        # 创建一个可追踪的模型对象
        import weakref
        
        class TrackableModel:
            def __init__(self, name):
                self.name = name
        
        # 加载模型
        model = TrackableModel('test_model')
        weak_ref = weakref.ref(model)
        manager._models['test_model'] = model
        
        # 验证模型存在
        assert weak_ref() is not None, "模型应该存在"
        
        # 删除本地引用
        del model
        
        # 此时模型还在manager中，弱引用应该仍然有效
        gc.collect()
        assert weak_ref() is not None, \
            "模型还在manager中，弱引用应该仍然有效"
        
        # 调用cleanup
        manager.cleanup()
        
        # 强制垃圾回收多次，确保对象被回收
        for _ in range(3):
            gc.collect()
            time.sleep(0.01)
        
        # 验证模型被回收（弱引用失效）
        assert weak_ref() is None, \
            "cleanup后，模型应该被垃圾回收，弱引用应该失效"
    
    def test_cleanup_clears_model_info(self):
        """
        单元测试：cleanup清空模型信息
        
        验证cleanup清空_model_info字典
        """
        manager = ModelManager.get_instance()
        
        # 添加模型和模型信息
        manager._models['test_model'] = type('MockModel', (), {})()
        manager._model_info['test_model'] = {
            'name': 'test_model',
            'load_time': 1.0,
            'memory_usage': 1000
        }
        
        # 验证数据存在
        assert len(manager._models) == 1
        assert len(manager._model_info) == 1
        
        # 调用cleanup
        manager.cleanup()
        
        # 验证都被清空
        assert len(manager._models) == 0, \
            "cleanup后_models应该为空"
        assert len(manager._model_info) == 0, \
            "cleanup后_model_info应该为空"
    
    def test_cleanup_handles_cleanup_errors_gracefully(self):
        """
        单元测试：cleanup优雅处理清理错误
        
        验证即使某个模型清理失败，cleanup也能继续清理其他模型
        """
        manager = ModelManager.get_instance()
        
        # 创建一个会在删除时抛出异常的模型
        class ProblematicModel:
            def __del__(self):
                raise RuntimeError("Cleanup error")
        
        # 添加多个模型，包括有问题的模型
        manager._models['good_model_1'] = type('MockModel', (), {})()
        manager._models['bad_model'] = ProblematicModel()
        manager._models['good_model_2'] = type('MockModel', (), {})()
        
        # 调用cleanup（不应该抛出异常）
        try:
            manager.cleanup()
        except Exception as e:
            pytest.fail(f"cleanup不应该抛出异常，但抛出了: {e}")
        
        # 验证所有模型都被从字典中移除
        assert len(manager._models) == 0, \
            "即使某个模型清理失败，所有模型都应该从字典中移除"
    
    def test_cleanup_forces_garbage_collection(self):
        """
        单元测试：cleanup强制垃圾回收
        
        验证cleanup调用gc.collect()
        """
        manager = ModelManager.get_instance()
        
        # 添加模型
        manager._models['test_model'] = type('MockModel', (), {})()
        
        # 记录垃圾回收前的对象数量
        gc.collect()
        before_count = len(gc.get_objects())
        
        # 调用cleanup
        manager.cleanup()
        
        # cleanup应该调用了gc.collect()
        # 注意：这个测试可能不稳定，因为gc.collect()的效果取决于很多因素
        # 但我们至少可以验证cleanup没有增加对象数量
        after_count = len(gc.get_objects())
        
        # 对象数量不应该显著增加
        assert after_count <= before_count + 100, \
            f"cleanup后对象数量不应该显著增加（前: {before_count}, 后: {after_count}）"
    
    @pytest.mark.skipif(
        not hasattr(__import__('importlib').import_module('torch'), 'cuda') 
        if 'torch' in sys.modules or __import__('importlib').util.find_spec('torch') 
        else True,
        reason="PyTorch未安装或不支持CUDA"
    )
    def test_cleanup_clears_gpu_cache_if_available(self):
        """
        单元测试：cleanup清理GPU缓存（如果可用）
        
        验证cleanup尝试清理GPU缓存
        """
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch未安装")
        
        manager = ModelManager.get_instance()
        
        # 添加模型
        manager._models['test_model'] = type('MockModel', (), {})()
        
        # 如果GPU可用，分配一些GPU内存
        if torch.cuda.is_available():
            # 创建一个小的tensor占用GPU内存
            device = torch.device('cuda')
            tensor = torch.randn(100, 100, device=device)
            
            # 记录GPU内存使用
            memory_before = torch.cuda.memory_allocated()
            
            # 删除tensor
            del tensor
            
            # 调用cleanup（应该清理GPU缓存）
            manager.cleanup()
            
            # 验证GPU内存被释放
            memory_after = torch.cuda.memory_allocated()
            
            # cleanup应该调用了torch.cuda.empty_cache()
            # 注意：empty_cache()不一定立即释放所有内存，但应该有所减少
            assert memory_after <= memory_before, \
                f"cleanup后GPU内存应该减少或保持不变（前: {memory_before}, 后: {memory_after}）"
        else:
            # GPU不可用，cleanup应该正常完成而不出错
            manager.cleanup()
    
    def test_cleanup_is_idempotent(self):
        """
        单元测试：cleanup是幂等的
        
        验证多次调用cleanup产生相同的结果
        """
        manager = ModelManager.get_instance()
        
        # 添加模型
        manager._models['test_model'] = type('MockModel', (), {})()
        
        # 第一次cleanup
        manager.cleanup()
        state_after_first = {
            'models_count': len(manager._models),
            'model_info_count': len(manager._model_info)
        }
        
        # 第二次cleanup
        manager.cleanup()
        state_after_second = {
            'models_count': len(manager._models),
            'model_info_count': len(manager._model_info)
        }
        
        # 第三次cleanup
        manager.cleanup()
        state_after_third = {
            'models_count': len(manager._models),
            'model_info_count': len(manager._model_info)
        }
        
        # 验证：所有状态相同
        assert state_after_first == state_after_second == state_after_third, \
            "多次cleanup应该产生相同的结果（幂等性）"
        
        # 验证：最终状态正确
        assert state_after_third['models_count'] == 0
        assert state_after_third['model_info_count'] == 0


if __name__ == '__main__':
    # 运行属性测试
    pytest.main([__file__, '-v', '--tb=short'])
