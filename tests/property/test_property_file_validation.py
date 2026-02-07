"""
属性测试：文件验证

Feature: model-singleton-optimization
Property 5: 文件验证

For any 配置中指定的模型文件路径，如果文件不存在，ModelManager应该在加载前
抛出包含文件路径的FileNotFoundError。

Validates: Requirements 3.4
"""

import pytest
from hypothesis import given, strategies as st, settings
import sys
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.model_manager import ModelManager


class TestPropertyFileValidation:
    """文件验证属性测试"""
    
    def setup_method(self):
        """每个测试前重置单例"""
        # 重置单例状态（仅用于测试）
        ModelManager._instance = None
        ModelManager._initialized = False
    
    @given(
        st.lists(
            st.sampled_from([
                'page_classifier_pytorch_best.pth',
                'page_classes.json',
                'page_yolo_mapping.json'
            ]),
            min_size=1,
            max_size=3,
            unique=True
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_property_missing_files_detected(self, missing_file_list):
        """
        属性测试：缺失文件检测
        
        For any 配置中指定的模型文件，如果文件不存在，
        _validate_model_files应该检测到并返回缺失文件列表
        
        测试策略：
        1. 生成随机的缺失文件列表（1-3个文件）
        2. 配置ModelManager使用这些不存在的文件
        3. 验证_validate_model_files返回所有缺失的文件
        """
        manager = ModelManager.get_instance()
        
        # 创建临时目录用于测试
        with tempfile.TemporaryDirectory() as temp_dir:
            # 配置使用不存在的文件路径
            temp_config = {
                'models': {
                    'page_detector_integrated': {
                        'enabled': True,
                        'model_path': os.path.join(temp_dir, 'nonexistent_model.pth'),
                        'classes_path': os.path.join(temp_dir, 'nonexistent_classes.json'),
                        'device': 'cpu'
                    },
                    'page_detector_hybrid': {
                        'enabled': True,
                        'mapping_path': os.path.join(temp_dir, 'nonexistent_mapping.json'),
                        'device': 'cpu'
                    },
                    'ocr_thread_pool': {
                        'enabled': False
                    }
                }
            }
            
            # 根据missing_file_list创建部分文件（其他保持不存在）
            all_files = {
                'page_classifier_pytorch_best.pth': temp_config['models']['page_detector_integrated']['model_path'],
                'page_classes.json': temp_config['models']['page_detector_integrated']['classes_path'],
                'page_yolo_mapping.json': temp_config['models']['page_detector_hybrid']['mapping_path']
            }
            
            # 创建不在missing_file_list中的文件
            for file_key, file_path in all_files.items():
                if file_key not in missing_file_list:
                    # 创建空文件
                    Path(file_path).touch()
            
            # 设置配置
            manager._config = temp_config
            
            # 调用验证方法
            missing_files = manager._validate_model_files()
            
            # 验证：返回的缺失文件数量应该等于missing_file_list的长度
            assert len(missing_files) == len(missing_file_list), \
                f"期望检测到 {len(missing_file_list)} 个缺失文件，但检测到 {len(missing_files)} 个"
            
            # 验证：所有缺失的文件都被检测到
            expected_missing_paths = [all_files[f] for f in missing_file_list]
            for expected_path in expected_missing_paths:
                assert expected_path in missing_files, \
                    f"缺失文件 {expected_path} 应该被检测到"
    
    @given(
        st.booleans(),  # page_detector_integrated enabled
        st.booleans(),  # page_detector_hybrid enabled
    )
    @settings(max_examples=100, deadline=None)
    def test_property_disabled_models_not_validated(self, pd_integrated_enabled, pd_hybrid_enabled):
        """
        属性测试：禁用的模型不验证
        
        For any 配置中禁用的模型，即使其文件不存在，也不应该被验证
        
        测试策略：
        1. 生成随机的模型启用/禁用配置
        2. 所有模型文件都不存在
        3. 验证只有启用的模型的文件被检测为缺失
        """
        manager = ModelManager.get_instance()
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            # 配置：所有文件都不存在
            temp_config = {
                'models': {
                    'page_detector_integrated': {
                        'enabled': pd_integrated_enabled,
                        'model_path': os.path.join(temp_dir, 'nonexistent_model.pth'),
                        'classes_path': os.path.join(temp_dir, 'nonexistent_classes.json'),
                        'device': 'cpu'
                    },
                    'page_detector_hybrid': {
                        'enabled': pd_hybrid_enabled,
                        'mapping_path': os.path.join(temp_dir, 'nonexistent_mapping.json'),
                        'device': 'cpu'
                    },
                    'ocr_thread_pool': {
                        'enabled': False  # OCR不需要文件
                    }
                }
            }
            
            manager._config = temp_config
            
            # 调用验证方法
            missing_files = manager._validate_model_files()
            
            # 计算期望的缺失文件数量
            expected_missing_count = 0
            if pd_integrated_enabled:
                expected_missing_count += 2  # model_path + classes_path
            if pd_hybrid_enabled:
                expected_missing_count += 1  # mapping_path
            
            # 验证：缺失文件数量应该等于启用的模型的文件数量
            assert len(missing_files) == expected_missing_count, \
                f"期望检测到 {expected_missing_count} 个缺失文件，但检测到 {len(missing_files)} 个"
    
    @given(
        st.integers(min_value=0, max_value=3)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_initialize_fails_with_missing_files(self, num_missing_files):
        """
        属性测试：缺失文件时初始化失败
        
        For any 数量的缺失文件（0-3个），如果有文件缺失，
        initialize_all_models应该抛出FileNotFoundError
        
        测试策略：
        1. 生成随机数量的缺失文件（0-3个）
        2. 调用initialize_all_models
        3. 验证：如果有缺失文件，应该抛出FileNotFoundError
        4. 验证：如果没有缺失文件，不应该抛出异常
        """
        manager = ModelManager.get_instance()
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建配置
            temp_config = {
                'models': {
                    'page_detector_integrated': {
                        'enabled': True,
                        'model_path': os.path.join(temp_dir, 'model.pth'),
                        'classes_path': os.path.join(temp_dir, 'classes.json'),
                        'device': 'cpu'
                    },
                    'page_detector_hybrid': {
                        'enabled': True,
                        'mapping_path': os.path.join(temp_dir, 'mapping.json'),
                        'device': 'cpu'
                    },
                    'ocr_thread_pool': {
                        'enabled': False
                    }
                }
            }
            
            # 根据num_missing_files创建文件
            all_files = [
                temp_config['models']['page_detector_integrated']['model_path'],
                temp_config['models']['page_detector_integrated']['classes_path'],
                temp_config['models']['page_detector_hybrid']['mapping_path']
            ]
            
            # 创建部分文件（留下num_missing_files个文件不创建）
            files_to_create = all_files[num_missing_files:]
            for file_path in files_to_create:
                Path(file_path).touch()
            
            manager._config = temp_config
            
            # Mock ADB桥接器
            mock_adb = MagicMock()
            
            if num_missing_files > 0:
                # 应该抛出FileNotFoundError
                with pytest.raises(FileNotFoundError) as exc_info:
                    manager.initialize_all_models(mock_adb)
                
                # 验证：异常消息包含"缺失"或"不存在"
                error_message = str(exc_info.value)
                assert '缺失' in error_message or '不存在' in error_message, \
                    f"错误消息应该包含'缺失'或'不存在'，但得到: {error_message}"
                
                # 验证：异常消息包含至少一个缺失文件的路径
                missing_files = all_files[:num_missing_files]
                assert any(missing_file in error_message for missing_file in missing_files), \
                    f"错误消息应该包含缺失文件的路径"
            else:
                # 没有缺失文件，但由于我们没有真实的模型文件，
                # 会在实际加载时失败，这里我们只验证文件验证通过
                # 使用patch来避免实际加载模型
                with patch.object(manager, '_load_page_detector_integrated'), \
                     patch.object(manager, '_load_page_detector_hybrid'), \
                     patch.object(manager, '_load_ocr_thread_pool'):
                    
                    # 不应该在文件验证阶段抛出异常
                    try:
                        manager.initialize_all_models(mock_adb)
                        # 如果成功，说明文件验证通过
                    except FileNotFoundError:
                        pytest.fail("当所有文件都存在时，不应该抛出FileNotFoundError")
    
    def test_file_validation_error_message_contains_path(self):
        """
        单元测试：文件验证错误消息包含文件路径
        
        验证当文件缺失时，错误消息包含具体的文件路径
        """
        manager = ModelManager.get_instance()
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_model_path = os.path.join(temp_dir, 'missing_model.pth')
            missing_classes_path = os.path.join(temp_dir, 'missing_classes.json')
            
            # 配置使用不存在的文件
            temp_config = {
                'models': {
                    'page_detector_integrated': {
                        'enabled': True,
                        'model_path': missing_model_path,
                        'classes_path': missing_classes_path,
                        'device': 'cpu'
                    },
                    'page_detector_hybrid': {
                        'enabled': False
                    },
                    'ocr_thread_pool': {
                        'enabled': False
                    }
                }
            }
            
            manager._config = temp_config
            
            # Mock ADB
            mock_adb = MagicMock()
            
            # 调用initialize_all_models应该抛出FileNotFoundError
            with pytest.raises(FileNotFoundError) as exc_info:
                manager.initialize_all_models(mock_adb)
            
            error_message = str(exc_info.value)
            
            # 验证：错误消息包含缺失的文件路径
            assert missing_model_path in error_message or missing_classes_path in error_message, \
                f"错误消息应该包含缺失文件的路径，但得到: {error_message}"
    
    def test_file_validation_checks_all_enabled_models(self):
        """
        单元测试：文件验证检查所有启用的模型
        
        验证_validate_model_files检查所有启用的模型的文件
        """
        manager = ModelManager.get_instance()
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            # 配置：所有模型都启用，所有文件都不存在
            temp_config = {
                'models': {
                    'page_detector_integrated': {
                        'enabled': True,
                        'model_path': os.path.join(temp_dir, 'model.pth'),
                        'classes_path': os.path.join(temp_dir, 'classes.json'),
                        'device': 'cpu'
                    },
                    'page_detector_hybrid': {
                        'enabled': True,
                        'mapping_path': os.path.join(temp_dir, 'mapping.json'),
                        'device': 'cpu'
                    },
                    'ocr_thread_pool': {
                        'enabled': True  # OCR不需要文件验证
                    }
                }
            }
            
            manager._config = temp_config
            
            # 调用验证方法
            missing_files = manager._validate_model_files()
            
            # 验证：应该检测到3个缺失文件
            # (model.pth, classes.json, mapping.json)
            assert len(missing_files) == 3, \
                f"期望检测到3个缺失文件，但检测到 {len(missing_files)} 个"
            
            # 验证：包含所有预期的文件
            expected_files = [
                os.path.join(temp_dir, 'model.pth'),
                os.path.join(temp_dir, 'classes.json'),
                os.path.join(temp_dir, 'mapping.json')
            ]
            
            for expected_file in expected_files:
                assert expected_file in missing_files, \
                    f"缺失文件列表应该包含 {expected_file}"
    
    def test_file_validation_returns_empty_list_when_all_exist(self):
        """
        单元测试：所有文件存在时返回空列表
        
        验证当所有必需的文件都存在时，_validate_model_files返回空列表
        """
        manager = ModelManager.get_instance()
        
        # 创建临时目录和文件
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'model.pth')
            classes_path = os.path.join(temp_dir, 'classes.json')
            mapping_path = os.path.join(temp_dir, 'mapping.json')
            
            # 创建所有文件
            Path(model_path).touch()
            Path(classes_path).touch()
            Path(mapping_path).touch()
            
            # 配置
            temp_config = {
                'models': {
                    'page_detector_integrated': {
                        'enabled': True,
                        'model_path': model_path,
                        'classes_path': classes_path,
                        'device': 'cpu'
                    },
                    'page_detector_hybrid': {
                        'enabled': True,
                        'mapping_path': mapping_path,
                        'device': 'cpu'
                    },
                    'ocr_thread_pool': {
                        'enabled': False
                    }
                }
            }
            
            manager._config = temp_config
            
            # 调用验证方法
            missing_files = manager._validate_model_files()
            
            # 验证：应该返回空列表
            assert len(missing_files) == 0, \
                f"当所有文件都存在时，应该返回空列表，但得到 {len(missing_files)} 个缺失文件"
            assert isinstance(missing_files, list), \
                "_validate_model_files应该返回列表类型"


if __name__ == '__main__':
    # 运行属性测试
    pytest.main([__file__, '-v', '--tb=short'])
