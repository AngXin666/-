"""
Task 1.4: 配置加载单元测试

测试ModelManager的配置加载功能，包括：
- 默认配置加载
- 用户配置覆盖
- 配置文件缺失
- 配置格式错误

Requirements: 10.3
"""

import sys
import os
import json
import tempfile
import shutil
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from model_manager import ModelManager


class TestConfigLoading:
    """配置加载测试类"""
    
    def setup_method(self):
        """每个测试前的设置"""
        # 重置单例
        ModelManager._instance = None
        ModelManager._initialized = False
        
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
    
    def teardown_method(self):
        """每个测试后的清理"""
        # 恢复工作目录
        os.chdir(self.original_cwd)
        
        # 删除临时目录
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        
        # 重置单例
        ModelManager._instance = None
        ModelManager._initialized = False
    
    def test_default_config_loading(self):
        """测试默认配置加载
        
        当配置文件不存在时，应该使用默认配置
        """
        print("\n" + "=" * 60)
        print("测试1: 默认配置加载")
        print("=" * 60)
        
        # 确保配置文件不存在
        config_path = 'model_config.json'
        assert not os.path.exists(config_path), "配置文件应该不存在"
        
        # 创建ModelManager实例
        manager = ModelManager.get_instance()
        
        # 验证配置结构
        assert 'models' in manager._config, "配置中应该有models字段"
        assert 'startup' in manager._config, "配置中应该有startup字段"
        
        # 验证模型配置
        models_config = manager._config['models']
        assert 'page_detector_integrated' in models_config, "应该有page_detector_integrated配置"
        assert 'page_detector_hybrid' in models_config, "应该有page_detector_hybrid配置"
        assert 'ocr_thread_pool' in models_config, "应该有ocr_thread_pool配置"
        
        # 验证默认值
        integrated_config = models_config['page_detector_integrated']
        assert integrated_config['enabled'] is True, "page_detector_integrated默认应该启用"
        assert integrated_config['device'] == 'auto', "device默认应该是auto"
        assert integrated_config['model_path'] == 'page_classifier_pytorch_best.pth', "model_path应该是默认值"
        assert integrated_config['classes_path'] == 'page_classes.json', "classes_path应该是默认值"
        assert integrated_config['quantize'] is False, "quantize默认应该是False"
        
        hybrid_config = models_config['page_detector_hybrid']
        assert hybrid_config['enabled'] is True, "page_detector_hybrid默认应该启用"
        assert hybrid_config['device'] == 'auto', "device默认应该是auto"
        
        ocr_config = models_config['ocr_thread_pool']
        assert ocr_config['enabled'] is True, "ocr_thread_pool默认应该启用"
        assert ocr_config['thread_count'] == 4, "thread_count默认应该是4"
        assert ocr_config['use_gpu'] is True, "use_gpu默认应该是True"
        
        # 验证启动配置
        startup_config = manager._config['startup']
        assert startup_config['show_progress'] is True, "show_progress默认应该是True"
        assert startup_config['log_loading_time'] is True, "log_loading_time默认应该是True"
        assert startup_config['log_memory_usage'] is True, "log_memory_usage默认应该是True"
        
        print("[OK] 默认配置加载测试通过")
        print(f"  - 配置文件存在: {os.path.exists(config_path)}")
        print(f"  - 模型配置数量: {len(models_config)}")
        print(f"  - 所有默认值正确")
    
    def test_user_config_override(self):
        """测试用户配置覆盖
        
        用户配置应该覆盖默认配置
        """
        print("\n" + "=" * 60)
        print("测试2: 用户配置覆盖")
        print("=" * 60)
        
        # 创建用户配置文件
        user_config = {
            'models': {
                'page_detector_integrated': {
                    'enabled': False,  # 覆盖默认值
                    'device': 'cpu',   # 覆盖默认值
                    'quantize': True   # 覆盖默认值
                },
                'ocr_thread_pool': {
                    'thread_count': 8  # 覆盖默认值
                }
            },
            'startup': {
                'show_progress': False  # 覆盖默认值
            }
        }
        
        config_path = 'model_config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(user_config, f, indent=2)
        
        # 创建ModelManager实例
        manager = ModelManager.get_instance()
        
        # 验证用户配置覆盖了默认值
        models_config = manager._config['models']
        
        integrated_config = models_config['page_detector_integrated']
        assert integrated_config['enabled'] is False, "enabled应该被覆盖为False"
        assert integrated_config['device'] == 'cpu', "device应该被覆盖为cpu"
        assert integrated_config['quantize'] is True, "quantize应该被覆盖为True"
        # 验证未覆盖的默认值仍然存在
        assert integrated_config['model_path'] == 'page_classifier_pytorch_best.pth', "未覆盖的默认值应该保留"
        assert integrated_config['classes_path'] == 'page_classes.json', "未覆盖的默认值应该保留"
        
        ocr_config = models_config['ocr_thread_pool']
        assert ocr_config['thread_count'] == 8, "thread_count应该被覆盖为8"
        # 验证未覆盖的默认值仍然存在
        assert ocr_config['enabled'] is True, "未覆盖的默认值应该保留"
        assert ocr_config['use_gpu'] is True, "未覆盖的默认值应该保留"
        
        # 验证未修改的模型配置仍然使用默认值
        hybrid_config = models_config['page_detector_hybrid']
        assert hybrid_config['enabled'] is True, "未修改的配置应该使用默认值"
        assert hybrid_config['device'] == 'auto', "未修改的配置应该使用默认值"
        
        # 验证启动配置
        startup_config = manager._config['startup']
        assert startup_config['show_progress'] is False, "show_progress应该被覆盖为False"
        assert startup_config['log_loading_time'] is True, "未覆盖的默认值应该保留"
        assert startup_config['log_memory_usage'] is True, "未覆盖的默认值应该保留"
        
        print("[OK] 用户配置覆盖测试通过")
        print(f"  - 用户配置正确覆盖默认值")
        print(f"  - 未覆盖的默认值正确保留")
        print(f"  - 嵌套配置正确合并")
    
    def test_missing_config_file(self):
        """测试配置文件缺失
        
        当配置文件不存在时，应该使用默认配置并记录警告
        """
        print("\n" + "=" * 60)
        print("测试3: 配置文件缺失")
        print("=" * 60)
        
        # 确保配置文件不存在
        config_path = 'model_config.json'
        if os.path.exists(config_path):
            os.remove(config_path)
        
        # 创建ModelManager实例
        manager = ModelManager.get_instance()
        
        # 验证使用了默认配置
        assert 'models' in manager._config, "应该使用默认配置"
        assert 'startup' in manager._config, "应该使用默认配置"
        
        models_config = manager._config['models']
        assert len(models_config) == 3, "应该有3个模型配置"
        
        # 验证所有默认值
        integrated_config = models_config['page_detector_integrated']
        assert integrated_config['enabled'] is True
        assert integrated_config['device'] == 'auto'
        assert integrated_config['model_path'] == 'page_classifier_pytorch_best.pth'
        
        print("[OK] 配置文件缺失测试通过")
        print(f"  - 配置文件不存在时使用默认配置")
        print(f"  - 程序正常运行")
    
    def test_invalid_json_format(self):
        """测试配置格式错误 - 无效的JSON
        
        当配置文件包含无效的JSON时，应该使用默认配置并记录警告
        """
        print("\n" + "=" * 60)
        print("测试4: 配置格式错误 - 无效JSON")
        print("=" * 60)
        
        # 创建无效的JSON配置文件
        config_path = 'model_config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write('{ invalid json content }')
        
        # 创建ModelManager实例
        manager = ModelManager.get_instance()
        
        # 验证使用了默认配置（因为JSON解析失败）
        assert 'models' in manager._config, "应该回退到默认配置"
        assert 'startup' in manager._config, "应该回退到默认配置"
        
        models_config = manager._config['models']
        assert len(models_config) == 3, "应该有3个模型配置"
        
        # 验证默认值
        integrated_config = models_config['page_detector_integrated']
        assert integrated_config['enabled'] is True, "应该使用默认值"
        assert integrated_config['device'] == 'auto', "应该使用默认值"
        
        print("[OK] 无效JSON格式测试通过")
        print(f"  - JSON解析失败时回退到默认配置")
        print(f"  - 程序正常运行")
    
    def test_empty_config_file(self):
        """测试配置格式错误 - 空配置文件
        
        当配置文件为空时，应该使用默认配置
        """
        print("\n" + "=" * 60)
        print("测试5: 配置格式错误 - 空文件")
        print("=" * 60)
        
        # 创建空配置文件
        config_path = 'model_config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write('')
        
        # 创建ModelManager实例
        manager = ModelManager.get_instance()
        
        # 验证使用了默认配置
        assert 'models' in manager._config, "应该回退到默认配置"
        assert 'startup' in manager._config, "应该回退到默认配置"
        
        print("[OK] 空配置文件测试通过")
        print(f"  - 空文件时回退到默认配置")
    
    def test_partial_config(self):
        """测试部分配置
        
        当配置文件只包含部分配置时，应该与默认配置合并
        """
        print("\n" + "=" * 60)
        print("测试6: 部分配置")
        print("=" * 60)
        
        # 创建只包含部分配置的文件
        user_config = {
            'models': {
                'page_detector_integrated': {
                    'enabled': False
                }
            }
        }
        
        config_path = 'model_config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(user_config, f, indent=2)
        
        # 创建ModelManager实例
        manager = ModelManager.get_instance()
        
        # 验证部分配置被应用
        models_config = manager._config['models']
        integrated_config = models_config['page_detector_integrated']
        assert integrated_config['enabled'] is False, "用户配置应该被应用"
        
        # 验证其他默认值仍然存在
        assert integrated_config['device'] == 'auto', "未指定的值应该使用默认值"
        assert integrated_config['model_path'] == 'page_classifier_pytorch_best.pth', "未指定的值应该使用默认值"
        
        # 验证其他模型配置使用默认值
        assert 'page_detector_hybrid' in models_config, "未指定的模型应该使用默认配置"
        assert 'ocr_thread_pool' in models_config, "未指定的模型应该使用默认配置"
        
        # 验证startup配置使用默认值
        assert 'startup' in manager._config, "未指定的配置节应该使用默认值"
        
        print("[OK] 部分配置测试通过")
        print(f"  - 部分配置正确应用")
        print(f"  - 未指定的配置使用默认值")
    
    def test_config_with_extra_fields(self):
        """测试配置包含额外字段
        
        当配置文件包含额外字段时，应该保留这些字段
        """
        print("\n" + "=" * 60)
        print("测试7: 配置包含额外字段")
        print("=" * 60)
        
        # 创建包含额外字段的配置
        user_config = {
            'models': {
                'page_detector_integrated': {
                    'enabled': True,
                    'custom_field': 'custom_value'  # 额外字段
                },
                'custom_model': {  # 额外模型
                    'enabled': True,
                    'path': 'custom_model.pth'
                }
            },
            'custom_section': {  # 额外配置节
                'custom_key': 'custom_value'
            }
        }
        
        config_path = 'model_config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(user_config, f, indent=2)
        
        # 创建ModelManager实例
        manager = ModelManager.get_instance()
        
        # 验证额外字段被保留
        models_config = manager._config['models']
        integrated_config = models_config['page_detector_integrated']
        assert 'custom_field' in integrated_config, "额外字段应该被保留"
        assert integrated_config['custom_field'] == 'custom_value', "额外字段值应该正确"
        
        # 验证额外模型被保留
        assert 'custom_model' in models_config, "额外模型应该被保留"
        assert models_config['custom_model']['path'] == 'custom_model.pth', "额外模型配置应该正确"
        
        # 验证额外配置节被保留
        assert 'custom_section' in manager._config, "额外配置节应该被保留"
        assert manager._config['custom_section']['custom_key'] == 'custom_value', "额外配置节值应该正确"
        
        # 验证默认配置仍然存在
        assert 'page_detector_hybrid' in models_config, "默认模型配置应该保留"
        assert 'startup' in manager._config, "默认配置节应该保留"
        
        print("[OK] 额外字段测试通过")
        print(f"  - 额外字段正确保留")
        print(f"  - 默认配置正确保留")
    
    def test_config_merge_deep_nesting(self):
        """测试深层嵌套配置合并
        
        验证_merge_config方法能正确处理深层嵌套的配置
        """
        print("\n" + "=" * 60)
        print("测试8: 深层嵌套配置合并")
        print("=" * 60)
        
        manager = ModelManager.get_instance()
        
        # 测试深层嵌套合并
        default = {
            'level1': {
                'level2': {
                    'level3': {
                        'a': 1,
                        'b': 2
                    },
                    'c': 3
                },
                'd': 4
            },
            'e': 5
        }
        
        user = {
            'level1': {
                'level2': {
                    'level3': {
                        'b': 20  # 覆盖深层值
                    },
                    'f': 30  # 添加新值
                }
            },
            'g': 50  # 添加顶层新值
        }
        
        merged = manager._merge_config(default, user)
        
        # 验证深层覆盖
        assert merged['level1']['level2']['level3']['a'] == 1, "未覆盖的深层值应该保留"
        assert merged['level1']['level2']['level3']['b'] == 20, "深层值应该被覆盖"
        
        # 验证新增值
        assert merged['level1']['level2']['f'] == 30, "新增的中层值应该添加"
        assert merged['g'] == 50, "新增的顶层值应该添加"
        
        # 验证其他值保留
        assert merged['level1']['level2']['c'] == 3, "其他中层值应该保留"
        assert merged['level1']['d'] == 4, "其他中层值应该保留"
        assert merged['e'] == 5, "其他顶层值应该保留"
        
        print("[OK] 深层嵌套配置合并测试通过")
        print(f"  - 深层值正确覆盖")
        print(f"  - 新增值正确添加")
        print(f"  - 未覆盖值正确保留")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("Task 1.4: 配置加载单元测试")
    print("=" * 60)
    
    test_suite = TestConfigLoading()
    
    tests = [
        ('test_default_config_loading', '默认配置加载'),
        ('test_user_config_override', '用户配置覆盖'),
        ('test_missing_config_file', '配置文件缺失'),
        ('test_invalid_json_format', '无效JSON格式'),
        ('test_empty_config_file', '空配置文件'),
        ('test_partial_config', '部分配置'),
        ('test_config_with_extra_fields', '额外字段'),
        ('test_config_merge_deep_nesting', '深层嵌套合并')
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_desc in tests:
        try:
            test_suite.setup_method()
            test_method = getattr(test_suite, test_name)
            test_method()
            test_suite.teardown_method()
            passed += 1
        except AssertionError as e:
            print(f"\n[FAILED] {test_desc}: {e}")
            failed += 1
        except Exception as e:
            print(f"\n[ERROR] {test_desc}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    exit(run_all_tests())
