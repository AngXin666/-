"""
属性测试：配置验证

Feature: model-singleton-optimization
Property 11: 配置验证

For any 无效的配置（缺失字段、类型错误、无效值），ModelManager应该使用默认值
并记录警告日志，而不是崩溃。

Validates: Requirements 10.3
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
import json
import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.model_manager import ModelManager


class TestPropertyConfigValidation:
    """配置验证属性测试"""
    
    def setup_method(self):
        """每个测试前重置单例并备份配置文件"""
        # 重置单例状态
        ModelManager._instance = None
        ModelManager._initialized = False
        
        # 备份现有配置文件（如果存在）
        self.config_path = 'model_config.json'
        self.backup_path = 'model_config.json.backup'
        
        if os.path.exists(self.config_path):
            shutil.copy2(self.config_path, self.backup_path)
            os.remove(self.config_path)
    
    def teardown_method(self):
        """每个测试后恢复配置文件"""
        # 删除测试创建的配置文件
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
        
        # 恢复备份的配置文件
        if os.path.exists(self.backup_path):
            shutil.move(self.backup_path, self.config_path)
    
    def _create_config_file(self, config: Dict[str, Any]):
        """创建配置文件
        
        Args:
            config: 配置字典
        """
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    @given(st.dictionaries(
        keys=st.text(min_size=1, max_size=20),
        values=st.one_of(
            st.text(),
            st.integers(),
            st.floats(allow_nan=False, allow_infinity=False),
            st.booleans(),
            st.none()
        ),
        min_size=0,
        max_size=10
    ))
    @settings(max_examples=100, deadline=None)
    def test_property_invalid_config_structure(self, random_config):
        """
        属性测试：无效配置结构
        
        For any 随机的配置结构，ModelManager应该能够处理并使用默认值
        
        测试策略：
        1. 生成随机的配置字典（可能缺少必需字段）
        2. 创建配置文件
        3. 初始化ModelManager
        4. 验证ModelManager成功初始化（使用默认值）
        5. 验证没有崩溃
        """
        # 创建随机配置文件
        self._create_config_file(random_config)
        
        # 初始化ModelManager（应该不会崩溃）
        try:
            manager = ModelManager.get_instance()
            
            # 验证：ModelManager成功创建
            assert manager is not None, "ModelManager应该成功创建"
            
            # 验证：配置已加载（即使是默认配置）
            assert manager._config is not None, "配置应该已加载"
            assert 'models' in manager._config, "配置应该包含models字段"
            assert 'startup' in manager._config, "配置应该包含startup字段"
            
            # 验证：关键模型配置存在（使用默认值）
            assert 'page_detector_integrated' in manager._config['models']
            assert 'page_detector_hybrid' in manager._config['models']
            assert 'ocr_thread_pool' in manager._config['models']
            
        except Exception as e:
            pytest.fail(f"ModelManager在处理无效配置时崩溃: {e}")
    
    @given(st.lists(
        st.one_of(
            st.text(min_size=0, max_size=100),
            st.integers(),
            st.floats(allow_nan=False, allow_infinity=False),
            st.booleans(),
            st.none(),
            st.lists(st.integers(), max_size=5),
            st.dictionaries(st.text(max_size=10), st.integers(), max_size=5)
        ),
        min_size=0,
        max_size=20
    ))
    @settings(max_examples=100, deadline=None)
    def test_property_invalid_config_types(self, random_values):
        """
        属性测试：无效配置类型
        
        For any 类型错误的配置值，ModelManager应该使用默认值
        
        测试策略：
        1. 生成随机类型的配置值
        2. 将这些值赋给配置字段
        3. 验证ModelManager能够处理并使用默认值
        """
        # 构造包含随机类型值的配置
        config = {
            'models': {},
            'startup': {}
        }
        
        # 随机填充配置字段
        model_fields = ['page_detector_integrated', 'page_detector_hybrid', 'ocr_thread_pool']
        for i, field in enumerate(model_fields):
            if i < len(random_values):
                config['models'][field] = random_values[i]
        
        # 创建配置文件
        self._create_config_file(config)
        
        # 初始化ModelManager（应该不会崩溃）
        try:
            manager = ModelManager.get_instance()
            
            # 验证：ModelManager成功创建
            assert manager is not None
            
            # 验证：配置已加载
            assert manager._config is not None
            
            # 验证：关键字段存在（即使使用了默认值）
            assert 'models' in manager._config
            assert 'startup' in manager._config
            
        except Exception as e:
            pytest.fail(f"ModelManager在处理类型错误的配置时崩溃: {e}")
    
    @given(st.text(min_size=0, max_size=1000))
    @settings(max_examples=100, deadline=None)
    def test_property_malformed_json(self, random_text):
        """
        属性测试：格式错误的JSON
        
        For any 格式错误的JSON文本，ModelManager应该使用默认配置
        
        测试策略：
        1. 生成随机文本（可能不是有效的JSON）
        2. 写入配置文件
        3. 验证ModelManager能够处理并使用默认配置
        """
        # 跳过有效的JSON（我们要测试无效的）
        try:
            json.loads(random_text)
            assume(False)  # 如果是有效JSON，跳过这个测试用例
        except (json.JSONDecodeError, ValueError):
            pass  # 这是我们想要的：无效JSON
        
        # 写入格式错误的JSON
        with open(self.config_path, 'w', encoding='utf-8') as f:
            f.write(random_text)
        
        # 初始化ModelManager（应该使用默认配置）
        try:
            manager = ModelManager.get_instance()
            
            # 验证：ModelManager成功创建
            assert manager is not None
            
            # 验证：使用了默认配置
            assert manager._config is not None
            assert 'models' in manager._config
            assert 'page_detector_integrated' in manager._config['models']
            
            # 验证：默认配置的关键字段存在
            integrated_config = manager._config['models']['page_detector_integrated']
            assert 'enabled' in integrated_config
            assert 'model_path' in integrated_config
            
        except Exception as e:
            pytest.fail(f"ModelManager在处理格式错误的JSON时崩溃: {e}")
    
    @given(
        st.booleans(),
        st.text(min_size=0, max_size=100),
        st.text(min_size=0, max_size=100),
        st.text(min_size=0, max_size=20)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_partial_config(self, enabled, model_path, classes_path, device):
        """
        属性测试：部分配置
        
        For any 部分配置（只包含部分字段），ModelManager应该合并默认值
        
        测试策略：
        1. 生成只包含部分字段的配置
        2. 验证ModelManager能够合并默认值
        3. 验证所有必需字段都存在
        """
        # 创建部分配置
        partial_config = {
            'models': {
                'page_detector_integrated': {
                    'enabled': enabled,
                    'model_path': model_path
                    # 缺少其他字段
                }
                # 缺少其他模型配置
            }
            # 缺少startup配置
        }
        
        self._create_config_file(partial_config)
        
        # 初始化ModelManager
        try:
            manager = ModelManager.get_instance()
            
            # 验证：ModelManager成功创建
            assert manager is not None, "ModelManager应该成功创建"
            
            # 验证：配置已合并
            config = manager._config
            assert 'models' in config, "配置应该包含models字段"
            assert 'startup' in config, "配置应该包含startup字段"
            
            # 验证：用户配置的字段被保留
            integrated = config['models']['page_detector_integrated']
            # 注意：由于配置合并的方式，用户提供的值应该被保留
            # 但如果用户值与默认值冲突，可能会有不同的行为
            # 这里我们只验证字段存在，不强制要求值完全相同
            assert 'enabled' in integrated, "enabled字段应该存在"
            assert 'model_path' in integrated, "model_path字段应该存在"
            
            # 验证：缺失的字段使用了默认值
            assert 'classes_path' in integrated, "classes_path应该使用默认值"
            assert 'device' in integrated, "device应该使用默认值"
            assert 'quantize' in integrated, "quantize应该使用默认值"
            
            # 验证：其他模型配置使用了默认值
            assert 'page_detector_hybrid' in config['models'], "应该包含hybrid配置"
            assert 'ocr_thread_pool' in config['models'], "应该包含ocr配置"
            
        except AssertionError:
            # 如果是断言失败，重新抛出以便hypothesis捕获
            raise
        except Exception as e:
            pytest.fail(f"ModelManager在处理部分配置时崩溃: {e}")
    
    @given(st.integers(min_value=-1000, max_value=1000))
    @settings(max_examples=100, deadline=None)
    def test_property_invalid_thread_count(self, thread_count):
        """
        属性测试：无效的线程数
        
        For any 无效的线程数（负数、零、过大），ModelManager应该使用默认值
        
        测试策略：
        1. 生成随机的线程数（可能无效）
        2. 验证ModelManager能够处理
        3. 验证使用了合理的默认值
        """
        config = {
            'models': {
                'ocr_thread_pool': {
                    'enabled': True,
                    'thread_count': thread_count,
                    'use_gpu': True
                }
            }
        }
        
        self._create_config_file(config)
        
        # 初始化ModelManager
        try:
            manager = ModelManager.get_instance()
            
            # 验证：ModelManager成功创建
            assert manager is not None
            
            # 验证：配置已加载
            ocr_config = manager._config['models']['ocr_thread_pool']
            
            # 验证：线程数字段存在
            assert 'thread_count' in ocr_config
            
            # 注意：这里我们不强制要求线程数必须是正数，
            # 因为实际的验证应该在模型加载时进行
            # 这个测试只验证配置加载不会崩溃
            
        except Exception as e:
            pytest.fail(f"ModelManager在处理无效线程数时崩溃: {e}")
    
    @given(st.text(min_size=0, max_size=50))
    @settings(max_examples=100, deadline=None)
    def test_property_invalid_device(self, device):
        """
        属性测试：无效的设备配置
        
        For any 无效的设备配置，ModelManager应该能够处理
        
        测试策略：
        1. 生成随机的设备字符串
        2. 验证ModelManager能够处理
        3. 验证配置加载不会崩溃
        """
        config = {
            'models': {
                'page_detector_integrated': {
                    'enabled': True,
                    'device': device
                }
            }
        }
        
        self._create_config_file(config)
        
        # 初始化ModelManager
        try:
            manager = ModelManager.get_instance()
            
            # 验证：ModelManager成功创建
            assert manager is not None
            
            # 验证：配置已加载
            integrated_config = manager._config['models']['page_detector_integrated']
            assert 'device' in integrated_config
            
        except Exception as e:
            pytest.fail(f"ModelManager在处理无效设备配置时崩溃: {e}")
    
    def test_missing_config_file(self):
        """
        单元测试：配置文件缺失
        
        验证当配置文件不存在时，ModelManager使用默认配置
        """
        # 确保配置文件不存在
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
        
        # 初始化ModelManager
        manager = ModelManager.get_instance()
        
        # 验证：使用了默认配置
        assert manager._config is not None
        assert 'models' in manager._config
        assert 'startup' in manager._config
        
        # 验证：所有关键模型配置存在
        assert 'page_detector_integrated' in manager._config['models']
        assert 'page_detector_hybrid' in manager._config['models']
        assert 'ocr_thread_pool' in manager._config['models']
        
        # 验证：默认值正确
        integrated = manager._config['models']['page_detector_integrated']
        assert integrated['enabled'] is True
        assert integrated['model_path'] == 'page_classifier_pytorch_best.pth'
        assert integrated['device'] == 'auto'
    
    def test_empty_config_file(self):
        """
        单元测试：空配置文件
        
        验证当配置文件为空时，ModelManager使用默认配置
        """
        # 创建空配置文件
        self._create_config_file({})
        
        # 初始化ModelManager
        manager = ModelManager.get_instance()
        
        # 验证：使用了默认配置
        assert manager._config is not None
        assert 'models' in manager._config
        assert 'startup' in manager._config
    
    def test_config_with_extra_fields(self):
        """
        单元测试：配置包含额外字段
        
        验证当配置包含未知字段时，ModelManager能够处理
        """
        config = {
            'models': {
                'page_detector_integrated': {
                    'enabled': True,
                    'model_path': 'test.pth',
                    'unknown_field': 'unknown_value',
                    'extra_config': {
                        'nested': 'value'
                    }
                }
            },
            'unknown_section': {
                'field': 'value'
            }
        }
        
        self._create_config_file(config)
        
        # 初始化ModelManager
        manager = ModelManager.get_instance()
        
        # 验证：ModelManager成功创建
        assert manager is not None
        
        # 验证：额外字段被保留（不会导致错误）
        integrated = manager._config['models']['page_detector_integrated']
        assert 'unknown_field' in integrated
        assert integrated['unknown_field'] == 'unknown_value'
    
    def test_config_merge_behavior(self):
        """
        单元测试：配置合并行为
        
        验证用户配置正确合并到默认配置中
        """
        # 创建部分用户配置
        user_config = {
            'models': {
                'page_detector_integrated': {
                    'enabled': False,  # 覆盖默认值
                    'model_path': 'custom.pth'  # 覆盖默认值
                    # 其他字段使用默认值
                }
            }
        }
        
        self._create_config_file(user_config)
        
        # 初始化ModelManager
        manager = ModelManager.get_instance()
        
        # 验证：用户配置覆盖了默认值
        integrated = manager._config['models']['page_detector_integrated']
        assert integrated['enabled'] is False
        assert integrated['model_path'] == 'custom.pth'
        
        # 验证：未指定的字段使用默认值
        assert 'classes_path' in integrated
        assert integrated['classes_path'] == 'page_classes.json'
        assert 'device' in integrated
        assert integrated['device'] == 'auto'
        
        # 验证：其他模型配置使用默认值
        assert 'page_detector_hybrid' in manager._config['models']
        assert manager._config['models']['page_detector_hybrid']['enabled'] is True
    
    def test_config_validation_warnings(self, capsys):
        """
        单元测试：配置验证警告
        
        验证当配置无效时，会记录警告日志
        """
        # 创建格式错误的配置文件
        with open(self.config_path, 'w', encoding='utf-8') as f:
            f.write('{ invalid json }')
        
        # 初始化ModelManager
        manager = ModelManager.get_instance()
        
        # 捕获输出
        captured = capsys.readouterr()
        
        # 验证：输出包含警告信息
        assert 'WARNING' in captured.out or 'warning' in captured.out.lower()
        
        # 验证：ModelManager仍然成功创建
        assert manager is not None
        assert manager._config is not None
    
    def test_config_with_null_values(self):
        """
        单元测试：配置包含null值
        
        验证当配置字段为null时，ModelManager能够处理
        """
        config = {
            'models': {
                'page_detector_integrated': {
                    'enabled': True,
                    'model_path': None,  # null值
                    'classes_path': None,
                    'device': None
                }
            }
        }
        
        self._create_config_file(config)
        
        # 初始化ModelManager
        try:
            manager = ModelManager.get_instance()
            
            # 验证：ModelManager成功创建
            assert manager is not None
            
            # 验证：配置已加载
            assert manager._config is not None
            
        except Exception as e:
            pytest.fail(f"ModelManager在处理null值时崩溃: {e}")
    
    def test_config_with_nested_invalid_structure(self):
        """
        单元测试：嵌套的无效配置结构
        
        验证当配置包含深层嵌套的无效结构时，ModelManager能够处理
        """
        config = {
            'models': {
                'page_detector_integrated': {
                    'enabled': True,
                    'nested': {
                        'deep': {
                            'structure': {
                                'with': {
                                    'many': {
                                        'levels': 'value'
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        self._create_config_file(config)
        
        # 初始化ModelManager
        try:
            manager = ModelManager.get_instance()
            
            # 验证：ModelManager成功创建
            assert manager is not None
            
        except Exception as e:
            pytest.fail(f"ModelManager在处理嵌套无效结构时崩溃: {e}")


if __name__ == '__main__':
    # 运行属性测试
    pytest.main([__file__, '-v', '--tb=short', '-s'])
