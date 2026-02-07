"""
属性测试：初始化顺序保证

Feature: model-singleton-optimization, Property 6: 初始化顺序保证

验证所有需要的模型在任务开始前已经完成加载并处于可用状态。
"""

import sys
import os
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from hypothesis import given, strategies as st, settings
import pytest

from model_manager import ModelManager


# Feature: model-singleton-optimization, Property 6: 初始化顺序保证
@given(
    st.lists(
        st.sampled_from(['page_detector_integrated', 'page_detector_hybrid', 'ocr_thread_pool']),
        min_size=1,
        max_size=20
    )
)
@settings(max_examples=100)
def test_property_initialization_order_guarantee(model_access_sequence):
    """
    属性测试：初始化顺序保证
    
    **Validates: Requirements 2.4, 4.4, 5.2**
    
    For any 自动化任务的执行，所有需要的模型应该在任务开始前已经完成加载并处于可用状态。
    
    测试策略：
    1. 模拟模型已初始化的状态
    2. 按随机顺序访问模型
    3. 验证所有访问都成功（不抛出未初始化异常）
    4. 验证is_initialized()返回True
    """
    # 重置单例
    ModelManager._instance = None
    ModelManager._initialized = False
    
    # 创建新实例
    manager = ModelManager.get_instance()
    
    # 模拟模型已初始化
    class MockModel:
        def __init__(self, name):
            self.name = name
            self.device = 'cpu'
    
    with manager._lock:
        manager._models['page_detector_integrated'] = MockModel('integrated')
        manager._models['page_detector_hybrid'] = MockModel('hybrid')
        manager._models['ocr_thread_pool'] = MockModel('ocr')
    
    # 验证：is_initialized返回True
    assert manager.is_initialized() is True, "模型已加载后，is_initialized应该返回True"
    
    # 验证：按任意顺序访问模型都成功
    for model_name in model_access_sequence:
        try:
            if model_name == 'page_detector_integrated':
                model = manager.get_page_detector_integrated()
            elif model_name == 'page_detector_hybrid':
                model = manager.get_page_detector_hybrid()
            else:  # ocr_thread_pool
                model = manager.get_ocr_thread_pool()
            
            # 验证返回的是有效对象
            assert model is not None, f"模型 {model_name} 不应该为None"
            assert hasattr(model, 'name'), f"模型 {model_name} 应该有name属性"
            
        except RuntimeError as e:
            pytest.fail(f"访问已初始化的模型 {model_name} 时不应该抛出异常: {e}")


@given(
    st.lists(
        st.sampled_from(['page_detector_integrated', 'page_detector_hybrid', 'ocr_thread_pool']),
        min_size=1,
        max_size=20
    )
)
@settings(max_examples=100)
def test_property_uninitialized_access_fails(model_access_sequence):
    """
    属性测试：未初始化访问失败
    
    **Validates: Requirements 2.4, 5.2**
    
    For any 模型访问请求，如果模型未初始化，应该抛出RuntimeError。
    
    测试策略：
    1. 确保模型未初始化
    2. 尝试访问模型
    3. 验证抛出RuntimeError
    """
    # 重置单例
    ModelManager._instance = None
    ModelManager._initialized = False
    
    # 创建新实例（但不初始化模型）
    manager = ModelManager.get_instance()
    
    # 验证：is_initialized返回False
    assert manager.is_initialized() is False, "模型未加载时，is_initialized应该返回False"
    
    # 验证：访问未初始化的模型抛出异常
    for model_name in model_access_sequence:
        with pytest.raises(RuntimeError, match="未初始化"):
            if model_name == 'page_detector_integrated':
                manager.get_page_detector_integrated()
            elif model_name == 'page_detector_hybrid':
                manager.get_page_detector_hybrid()
            else:  # ocr_thread_pool
                manager.get_ocr_thread_pool()


def test_initialization_order_single_model():
    """单元测试：单个模型初始化"""
    ModelManager._instance = None
    ModelManager._initialized = False
    
    manager = ModelManager.get_instance()
    
    # 初始状态：未初始化
    assert manager.is_initialized() is False
    
    # 添加一个模型
    class MockModel:
        pass
    
    with manager._lock:
        manager._models['page_detector_integrated'] = MockModel()
    
    # 现在应该是已初始化
    assert manager.is_initialized() is True
    
    # 可以访问模型
    model = manager.get_page_detector_integrated()
    assert model is not None


def test_initialization_order_all_models():
    """单元测试：所有模型初始化"""
    ModelManager._instance = None
    ModelManager._initialized = False
    
    manager = ModelManager.get_instance()
    
    # 初始状态：未初始化
    assert manager.is_initialized() is False
    
    # 添加所有模型
    class MockModel:
        def __init__(self, name):
            self.name = name
    
    with manager._lock:
        manager._models['page_detector_integrated'] = MockModel('integrated')
        manager._models['page_detector_hybrid'] = MockModel('hybrid')
        manager._models['ocr_thread_pool'] = MockModel('ocr')
    
    # 现在应该是已初始化
    assert manager.is_initialized() is True
    
    # 可以访问所有模型
    integrated = manager.get_page_detector_integrated()
    hybrid = manager.get_page_detector_hybrid()
    ocr = manager.get_ocr_thread_pool()
    
    assert integrated.name == 'integrated'
    assert hybrid.name == 'hybrid'
    assert ocr.name == 'ocr'


def test_initialization_order_partial_initialization():
    """单元测试：部分初始化"""
    ModelManager._instance = None
    ModelManager._initialized = False
    
    manager = ModelManager.get_instance()
    
    # 只添加部分模型
    class MockModel:
        pass
    
    with manager._lock:
        manager._models['page_detector_integrated'] = MockModel()
    
    # 应该是已初始化（因为有至少一个模型）
    assert manager.is_initialized() is True
    
    # 可以访问已加载的模型
    integrated = manager.get_page_detector_integrated()
    assert integrated is not None
    
    # 访问未加载的模型应该失败
    with pytest.raises(RuntimeError):
        manager.get_page_detector_hybrid()


def test_initialization_order_empty():
    """单元测试：空初始化"""
    ModelManager._instance = None
    ModelManager._initialized = False
    
    manager = ModelManager.get_instance()
    
    # 没有添加任何模型
    assert manager.is_initialized() is False
    
    # 访问任何模型都应该失败
    with pytest.raises(RuntimeError):
        manager.get_page_detector_integrated()
    
    with pytest.raises(RuntimeError):
        manager.get_page_detector_hybrid()
    
    with pytest.raises(RuntimeError):
        manager.get_ocr_thread_pool()


if __name__ == '__main__':
    print("=" * 60)
    print("属性测试：初始化顺序保证")
    print("=" * 60)
    print("\n运行属性测试（100次迭代）...")
    
    # 运行属性测试
    pytest.main([__file__, '-v', '--tb=short'])
