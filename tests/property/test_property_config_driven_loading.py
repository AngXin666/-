"""
属性测试：配置驱动加载

Feature: model-singleton-optimization, Property 4: 配置驱动加载

验证ModelManager根据配置正确加载或跳过相应的模型。
"""

import sys
import os
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from hypothesis import given, strategies as st, settings
import pytest

from model_manager import ModelManager


# Feature: model-singleton-optimization, Property 4: 配置驱动加载
@given(
    st.dictionaries(
        keys=st.sampled_from(['page_detector_integrated', 'page_detector_hybrid', 'ocr_thread_pool']),
        values=st.booleans(),
        min_size=1,
        max_size=3
    )
)
@settings(max_examples=100)
def test_property_config_driven_loading(enabled_models):
    """
    属性测试：配置驱动加载
    
    **Validates: Requirements 3.2, 10.1, 10.2**
    
    For any 有效的配置文件，ModelManager应该根据配置正确加载或跳过相应的模型，
    且加载的模型列表应该与配置中启用的模型列表一致。
    
    测试策略：
    1. 生成随机的模型启用/禁用配置
    2. 应用配置到ModelManager
    3. 验证_is_model_enabled方法返回正确的结果
    4. 验证配置被正确合并和应用
    """
    # 重置单例（测试用）
    ModelManager._instance = None
    ModelManager._initialized = False
    
    # 创建新实例
    manager = ModelManager.get_instance()
    
    # 创建测试配置
    test_config = {
        'models': {},
        'startup': {
            'show_progress': True,
            'log_loading_time': True,
            'log_memory_usage': True
        }
    }
    
    # 填充模型配置
    for model_name in ['page_detector_integrated', 'page_detector_hybrid', 'ocr_thread_pool']:
        if model_name in enabled_models:
            test_config['models'][model_name] = {
                'enabled': enabled_models[model_name]
            }
        else:
            # 如果没有在enabled_models中，使用默认值True
            test_config['models'][model_name] = {
                'enabled': True
            }
    
    # 应用配置
    with manager._lock:
        manager._config = test_config
    
    # 验证：_is_model_enabled返回正确的结果
    for model_name in ['page_detector_integrated', 'page_detector_hybrid', 'ocr_thread_pool']:
        expected_enabled = test_config['models'][model_name]['enabled']
        actual_enabled = manager._is_model_enabled(model_name)
        
        assert actual_enabled == expected_enabled, (
            f"模型 {model_name} 的启用状态不正确: "
            f"期望={expected_enabled}, 实际={actual_enabled}"
        )
    
    # 验证：配置结构完整
    assert 'models' in manager._config
    assert 'startup' in manager._config
    
    # 验证：所有模型都有配置
    for model_name in ['page_detector_integrated', 'page_detector_hybrid', 'ocr_thread_pool']:
        assert model_name in manager._config['models'], f"缺少模型配置: {model_name}"


def test_config_driven_loading_all_enabled():
    """单元测试：所有模型启用"""
    ModelManager._instance = None
    ModelManager._initialized = False
    
    manager = ModelManager.get_instance()
    
    # 所有模型启用
    test_config = {
        'models': {
            'page_detector_integrated': {'enabled': True},
            'page_detector_hybrid': {'enabled': True},
            'ocr_thread_pool': {'enabled': True}
        }
    }
    
    with manager._lock:
        manager._config = test_config
    
    # 验证所有模型都启用
    assert manager._is_model_enabled('page_detector_integrated') is True
    assert manager._is_model_enabled('page_detector_hybrid') is True
    assert manager._is_model_enabled('ocr_thread_pool') is True


def test_config_driven_loading_all_disabled():
    """单元测试：所有模型禁用"""
    ModelManager._instance = None
    ModelManager._initialized = False
    
    manager = ModelManager.get_instance()
    
    # 所有模型禁用
    test_config = {
        'models': {
            'page_detector_integrated': {'enabled': False},
            'page_detector_hybrid': {'enabled': False},
            'ocr_thread_pool': {'enabled': False}
        }
    }
    
    with manager._lock:
        manager._config = test_config
    
    # 验证所有模型都禁用
    assert manager._is_model_enabled('page_detector_integrated') is False
    assert manager._is_model_enabled('page_detector_hybrid') is False
    assert manager._is_model_enabled('ocr_thread_pool') is False


def test_config_driven_loading_mixed():
    """单元测试：混合启用/禁用"""
    ModelManager._instance = None
    ModelManager._initialized = False
    
    manager = ModelManager.get_instance()
    
    # 混合配置
    test_config = {
        'models': {
            'page_detector_integrated': {'enabled': True},
            'page_detector_hybrid': {'enabled': False},
            'ocr_thread_pool': {'enabled': True}
        }
    }
    
    with manager._lock:
        manager._config = test_config
    
    # 验证配置正确应用
    assert manager._is_model_enabled('page_detector_integrated') is True
    assert manager._is_model_enabled('page_detector_hybrid') is False
    assert manager._is_model_enabled('ocr_thread_pool') is True


def test_config_driven_loading_nonexistent_model():
    """单元测试：不存在的模型"""
    ModelManager._instance = None
    ModelManager._initialized = False
    
    manager = ModelManager.get_instance()
    
    # 查询不存在的模型
    assert manager._is_model_enabled('nonexistent_model') is False


if __name__ == '__main__':
    print("=" * 60)
    print("属性测试：配置驱动加载")
    print("=" * 60)
    print("\n运行属性测试（100次迭代）...")
    
    # 运行属性测试
    pytest.main([__file__, '-v', '--tb=short'])
