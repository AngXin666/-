"""
属性测试：性能监控完整性

Feature: model-singleton-optimization
Property 10: 性能监控完整性

For any 模型加载操作，ModelManager应该记录该模型的加载时间，并在所有模型加载完成后
提供总加载时间和各模型加载时间的统计信息。

Validates: Requirements 9.1
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
import sys
import os
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.model_manager import ModelManager, LoadingStats


class TestPropertyPerformanceMonitoring:
    """性能监控完整性属性测试"""
    
    def setup_method(self):
        """每个测试前重置单例"""
        # 重置单例状态（仅用于测试）
        ModelManager._instance = None
        ModelManager._initialized = False
    
    @given(
        st.lists(
            st.tuples(
                st.sampled_from(['page_detector_integrated', 'page_detector_hybrid', 'ocr_thread_pool']),
                st.floats(min_value=0.1, max_value=10.0)  # 模拟加载时间（秒）
            ),
            min_size=1,
            max_size=3,
            unique_by=lambda x: x[0]  # 确保模型名称唯一
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_records_individual_model_loading_time(self, model_load_times):
        """
        属性测试：记录每个模型的加载时间
        
        For any 模型加载操作，ModelManager应该记录该模型的加载时间
        
        测试策略：
        1. 生成随机的模型加载时间列表
        2. 模拟模型加载过程
        3. 验证每个模型的加载时间都被记录
        4. 验证记录的时间与实际加载时间一致（允许小误差）
        """
        manager = ModelManager.get_instance()
        
        # 模拟加载统计
        manager._loading_stats = LoadingStats(
            total_models=len(model_load_times),
            loaded_models=0,
            failed_models=0,
            total_time=0.0,
            memory_before=0,
            memory_after=0,
            errors=[],
            model_times={}
        )
        
        # 模拟每个模型的加载
        for model_name, load_time in model_load_times:
            # 记录加载时间
            manager._loading_stats.model_times[model_name] = load_time
            manager._loading_stats.loaded_models += 1
        
        # 获取加载统计
        stats = manager.get_loading_stats()
        
        # 验证：每个模型的加载时间都被记录
        for model_name, expected_time in model_load_times:
            assert model_name in stats['model_times'], \
                f"模型 {model_name} 的加载时间应该被记录"
            
            recorded_time = stats['model_times'][model_name]
            assert abs(recorded_time - expected_time) < 0.01, \
                f"模型 {model_name} 的记录时间 {recorded_time} 应该接近实际时间 {expected_time}"
    
    @given(
        st.lists(
            st.floats(min_value=0.1, max_value=10.0),
            min_size=1,
            max_size=10
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_calculates_total_loading_time(self, individual_times):
        """
        属性测试：计算总加载时间
        
        For any 模型加载操作集合，ModelManager应该提供总加载时间
        
        测试策略：
        1. 生成随机的个体加载时间列表
        2. 模拟加载过程
        3. 验证总加载时间等于所有个体时间之和
        """
        manager = ModelManager.get_instance()
        
        # 计算预期的总时间
        expected_total_time = sum(individual_times)
        
        # 模拟加载统计
        manager._loading_stats = LoadingStats(
            total_models=len(individual_times),
            loaded_models=len(individual_times),
            failed_models=0,
            total_time=expected_total_time,
            memory_before=0,
            memory_after=0,
            errors=[],
            model_times={f'model_{i}': t for i, t in enumerate(individual_times)}
        )
        
        # 获取加载统计
        stats = manager.get_loading_stats()
        
        # 验证：总加载时间正确
        assert 'total_time' in stats, "统计信息应该包含总加载时间"
        assert abs(stats['total_time'] - expected_total_time) < 0.01, \
            f"总加载时间 {stats['total_time']} 应该接近预期时间 {expected_total_time}"
        
        # 验证：平均加载时间正确
        expected_avg_time = expected_total_time / len(individual_times)
        assert abs(stats['average_load_time'] - expected_avg_time) < 0.01, \
            f"平均加载时间 {stats['average_load_time']} 应该接近预期 {expected_avg_time}"
    
    @given(
        st.tuples(
            st.integers(min_value=100_000_000, max_value=1_000_000_000),  # 加载前内存（字节）
            st.integers(min_value=100_000_000, max_value=2_000_000_000)   # 加载后内存（字节）
        ).filter(lambda x: x[1] >= x[0])  # 确保加载后内存 >= 加载前内存
    )
    @settings(max_examples=100, deadline=None)
    def test_property_records_memory_usage(self, memory_values):
        """
        属性测试：记录内存使用情况
        
        For any 模型加载操作，ModelManager应该记录加载前后的内存使用情况
        
        测试策略：
        1. 生成随机的内存使用值（加载前和加载后）
        2. 模拟加载过程
        3. 验证内存统计信息正确
        4. 验证内存增量计算正确
        """
        memory_before, memory_after = memory_values
        
        manager = ModelManager.get_instance()
        
        # 模拟加载统计
        manager._loading_stats = LoadingStats(
            total_models=1,
            loaded_models=1,
            failed_models=0,
            total_time=1.0,
            memory_before=memory_before,
            memory_after=memory_after,
            errors=[],
            model_times={'test_model': 1.0}
        )
        
        # 获取加载统计
        stats = manager.get_loading_stats()
        
        # 验证：内存统计信息存在
        assert 'memory_before' in stats, "统计信息应该包含加载前内存"
        assert 'memory_after' in stats, "统计信息应该包含加载后内存"
        assert 'memory_delta' in stats, "统计信息应该包含内存增量"
        
        # 验证：内存值正确
        assert stats['memory_before'] == memory_before, \
            f"加载前内存 {stats['memory_before']} 应该等于 {memory_before}"
        assert stats['memory_after'] == memory_after, \
            f"加载后内存 {stats['memory_after']} 应该等于 {memory_after}"
        
        # 验证：内存增量计算正确
        expected_delta = memory_after - memory_before
        assert stats['memory_delta'] == expected_delta, \
            f"内存增量 {stats['memory_delta']} 应该等于 {expected_delta}"
        
        # 验证：MB单位转换正确
        expected_delta_mb = expected_delta / 1024 / 1024
        assert abs(stats['memory_delta_mb'] - expected_delta_mb) < 0.01, \
            f"内存增量(MB) {stats['memory_delta_mb']} 应该接近 {expected_delta_mb}"
    
    @given(
        st.lists(
            st.text(min_size=10, max_size=100),  # 错误消息
            min_size=0,
            max_size=10
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_records_errors(self, error_messages):
        """
        属性测试：记录错误信息
        
        For any 模型加载过程中的错误，ModelManager应该记录错误信息
        
        测试策略：
        1. 生成随机的错误消息列表
        2. 模拟加载过程中的错误
        3. 验证所有错误都被记录
        4. 验证错误标志正确设置
        """
        manager = ModelManager.get_instance()
        
        # 模拟加载统计
        manager._loading_stats = LoadingStats(
            total_models=len(error_messages) + 1,
            loaded_models=1,
            failed_models=len(error_messages),
            total_time=1.0,
            memory_before=0,
            memory_after=0,
            errors=list(error_messages),
            model_times={'success_model': 1.0}
        )
        
        # 获取加载统计
        stats = manager.get_loading_stats()
        
        # 验证：错误列表存在
        assert 'errors' in stats, "统计信息应该包含错误列表"
        assert 'has_errors' in stats, "统计信息应该包含错误标志"
        
        # 验证：错误数量正确
        assert len(stats['errors']) == len(error_messages), \
            f"错误数量 {len(stats['errors'])} 应该等于 {len(error_messages)}"
        
        # 验证：错误标志正确
        expected_has_errors = len(error_messages) > 0
        assert stats['has_errors'] == expected_has_errors, \
            f"错误标志 {stats['has_errors']} 应该等于 {expected_has_errors}"
        
        # 验证：所有错误消息都被记录
        for error_msg in error_messages:
            assert error_msg in stats['errors'], \
                f"错误消息 '{error_msg}' 应该被记录"
    
    @given(
        st.tuples(
            st.integers(min_value=1, max_value=10),  # 总模型数
            st.integers(min_value=0, max_value=10),  # 已加载模型数
            st.integers(min_value=0, max_value=10)   # 失败模型数
        ).filter(lambda x: x[1] + x[2] <= x[0])  # 确保已加载+失败 <= 总数
    )
    @settings(max_examples=100, deadline=None)
    def test_property_calculates_success_rate(self, model_counts):
        """
        属性测试：计算成功率
        
        For any 模型加载操作，ModelManager应该计算并提供加载成功率
        
        测试策略：
        1. 生成随机的模型数量（总数、已加载、失败）
        2. 模拟加载统计
        3. 验证成功率计算正确
        """
        total_models, loaded_models, failed_models = model_counts
        
        manager = ModelManager.get_instance()
        
        # 模拟加载统计
        manager._loading_stats = LoadingStats(
            total_models=total_models,
            loaded_models=loaded_models,
            failed_models=failed_models,
            total_time=1.0,
            memory_before=0,
            memory_after=0,
            errors=[],
            model_times={}
        )
        
        # 获取加载统计
        stats = manager.get_loading_stats()
        
        # 验证：成功率存在
        assert 'success_rate' in stats, "统计信息应该包含成功率"
        
        # 计算预期成功率
        if total_models > 0:
            expected_success_rate = (loaded_models / total_models) * 100
        else:
            expected_success_rate = 0.0
        
        # 验证：成功率计算正确
        assert abs(stats['success_rate'] - expected_success_rate) < 0.01, \
            f"成功率 {stats['success_rate']} 应该接近 {expected_success_rate}"
        
        # 验证：成功率在合理范围内
        assert 0.0 <= stats['success_rate'] <= 100.0, \
            f"成功率 {stats['success_rate']} 应该在 0-100 之间"
    
    def test_property_statistics_completeness(self):
        """
        单元测试：统计信息完整性
        
        验证get_loading_stats()返回所有必需的统计字段
        """
        manager = ModelManager.get_instance()
        
        # 模拟加载统计
        manager._loading_stats = LoadingStats(
            total_models=3,
            loaded_models=2,
            failed_models=1,
            total_time=5.5,
            memory_before=100_000_000,
            memory_after=150_000_000,
            errors=['Error loading model X'],
            model_times={
                'model_a': 2.0,
                'model_b': 3.5
            }
        )
        
        # 获取加载统计
        stats = manager.get_loading_stats()
        
        # 验证：所有必需字段都存在
        required_fields = [
            # 模型数量统计
            'total_models', 'loaded_models', 'failed_models', 'success_rate',
            # 时间统计
            'total_time', 'average_load_time', 'model_times',
            # 内存统计
            'memory_before', 'memory_after', 'memory_delta',
            'memory_before_mb', 'memory_after_mb', 'memory_delta_mb',
            # 错误信息
            'errors', 'has_errors'
        ]
        
        for field in required_fields:
            assert field in stats, f"统计信息应该包含字段: {field}"
        
        # 验证：字段类型正确
        assert isinstance(stats['total_models'], int)
        assert isinstance(stats['loaded_models'], int)
        assert isinstance(stats['failed_models'], int)
        assert isinstance(stats['success_rate'], (int, float))
        assert isinstance(stats['total_time'], (int, float))
        assert isinstance(stats['average_load_time'], (int, float))
        assert isinstance(stats['model_times'], dict)
        assert isinstance(stats['memory_before'], int)
        assert isinstance(stats['memory_after'], int)
        assert isinstance(stats['memory_delta'], int)
        assert isinstance(stats['memory_before_mb'], (int, float))
        assert isinstance(stats['memory_after_mb'], (int, float))
        assert isinstance(stats['memory_delta_mb'], (int, float))
        assert isinstance(stats['errors'], list)
        assert isinstance(stats['has_errors'], bool)
    
    def test_property_statistics_immutability(self):
        """
        单元测试：统计信息不可变性
        
        验证get_loading_stats()返回的是副本，修改不会影响内部状态
        """
        manager = ModelManager.get_instance()
        
        # 模拟加载统计
        manager._loading_stats = LoadingStats(
            total_models=1,
            loaded_models=1,
            failed_models=0,
            total_time=1.0,
            memory_before=0,
            memory_after=0,
            errors=['original_error'],
            model_times={'model_a': 1.0}
        )
        
        # 获取加载统计
        stats1 = manager.get_loading_stats()
        
        # 修改返回的统计信息
        stats1['errors'].append('new_error')
        stats1['model_times']['model_b'] = 2.0
        
        # 再次获取加载统计
        stats2 = manager.get_loading_stats()
        
        # 验证：内部状态没有被修改
        assert len(stats2['errors']) == 1, \
            "修改返回的统计信息不应该影响内部状态"
        assert 'original_error' in stats2['errors']
        assert 'new_error' not in stats2['errors']
        
        assert len(stats2['model_times']) == 1, \
            "修改返回的model_times不应该影响内部状态"
        assert 'model_a' in stats2['model_times']
        assert 'model_b' not in stats2['model_times']
    
    @given(
        st.lists(
            st.tuples(
                st.text(min_size=5, max_size=30),  # 模型名称
                st.floats(min_value=0.1, max_value=10.0)  # 加载时间
            ),
            min_size=1,
            max_size=5,
            unique_by=lambda x: x[0]  # 确保模型名称唯一
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_model_times_consistency(self, model_data):
        """
        属性测试：模型时间一致性
        
        For any 模型加载操作，model_times中的时间总和应该接近total_time
        
        测试策略：
        1. 生成随机的模型加载数据
        2. 模拟加载统计
        3. 验证model_times中的时间总和与total_time一致
        """
        manager = ModelManager.get_instance()
        
        # 计算总时间
        total_time = sum(load_time for _, load_time in model_data)
        
        # 模拟加载统计
        manager._loading_stats = LoadingStats(
            total_models=len(model_data),
            loaded_models=len(model_data),
            failed_models=0,
            total_time=total_time,
            memory_before=0,
            memory_after=0,
            errors=[],
            model_times={name: time for name, time in model_data}
        )
        
        # 获取加载统计
        stats = manager.get_loading_stats()
        
        # 计算model_times中的时间总和
        model_times_sum = sum(stats['model_times'].values())
        
        # 验证：时间总和一致（允许小误差）
        assert abs(model_times_sum - stats['total_time']) < 0.01, \
            f"model_times总和 {model_times_sum} 应该接近total_time {stats['total_time']}"
    
    def test_property_zero_models_edge_case(self):
        """
        边界测试：零模型情况
        
        验证当没有模型加载时，统计信息仍然正确
        """
        manager = ModelManager.get_instance()
        
        # 模拟零模型加载
        manager._loading_stats = LoadingStats(
            total_models=0,
            loaded_models=0,
            failed_models=0,
            total_time=0.0,
            memory_before=0,
            memory_after=0,
            errors=[],
            model_times={}
        )
        
        # 获取加载统计
        stats = manager.get_loading_stats()
        
        # 验证：统计信息正确
        assert stats['total_models'] == 0
        assert stats['loaded_models'] == 0
        assert stats['failed_models'] == 0
        assert stats['success_rate'] == 0.0
        assert stats['total_time'] == 0.0
        assert stats['average_load_time'] == 0.0
        assert len(stats['model_times']) == 0
        assert len(stats['errors']) == 0
        assert stats['has_errors'] is False


if __name__ == '__main__':
    # 运行属性测试
    pytest.main([__file__, '-v', '--tb=short'])
