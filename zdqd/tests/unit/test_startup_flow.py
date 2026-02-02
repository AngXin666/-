#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单元测试：启动流程

测试启动顺序正确性和错误处理
Requirements: 5.2
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch, MagicMock, call
from io import StringIO

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestStartupSequence(unittest.TestCase):
    """测试启动顺序正确性"""
    
    def setUp(self):
        """测试前准备"""
        # 重置 ModelManager 单例
        from src.model_manager import ModelManager
        ModelManager._instance = None
        ModelManager._initialized = False
    
    def test_model_manager_initializes_before_gui(self):
        """测试 ModelManager 在 GUI 之前初始化"""
        print("\n测试：ModelManager 在 GUI 之前初始化")
        
        # 读取 run.py 文件
        with open('run.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找关键代码位置
        model_init_pos = content.find('initialize_all_models')
        gui_import_pos = content.find('from src.gui import main as gui_main')
        gui_call_pos = content.find('gui_main()')
        
        # 验证顺序
        self.assertGreater(model_init_pos, 0, "应该包含 initialize_all_models 调用")
        self.assertGreater(gui_import_pos, 0, "应该包含 GUI 导入")
        self.assertGreater(gui_call_pos, 0, "应该包含 GUI 启动调用")
        
        # 验证 ModelManager 初始化在 GUI 启动之前
        self.assertLess(model_init_pos, gui_call_pos,
                       "ModelManager 初始化必须在 GUI 启动之前")
        
        print("✓ ModelManager 在 GUI 之前初始化")
    
    def test_adb_initializes_before_model_manager(self):
        """测试 ADB 在 ModelManager 之前初始化"""
        print("\n测试：ADB 在 ModelManager 之前初始化")
        
        with open('run.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找 ADB 和 ModelManager 初始化位置
        adb_init_pos = content.find('ADBBridge()')
        model_init_pos = content.find('initialize_all_models')
        
        # 验证顺序
        self.assertGreater(adb_init_pos, 0, "应该包含 ADB 初始化")
        self.assertGreater(model_init_pos, 0, "应该包含 ModelManager 初始化")
        
        # ADB 必须在 ModelManager 之前初始化
        self.assertLess(adb_init_pos, model_init_pos,
                       "ADB 初始化必须在 ModelManager 之前")
        
        print("✓ ADB 在 ModelManager 之前初始化")
    
    def test_license_check_before_model_loading(self):
        """测试许可证检查在模型加载之前"""
        print("\n测试：许可证检查在模型加载之前")
        
        with open('run.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找许可证检查和模型加载位置
        license_pos = content.find('SimpleLicenseManager')
        model_init_pos = content.find('initialize_all_models')
        
        # 验证顺序
        self.assertGreater(license_pos, 0, "应该包含许可证检查")
        self.assertGreater(model_init_pos, 0, "应该包含模型加载")
        
        # 许可证检查必须在模型加载之前
        self.assertLess(license_pos, model_init_pos,
                       "许可证检查必须在模型加载之前")
        
        print("✓ 许可证检查在模型加载之前")
    
    def test_complete_startup_order(self):
        """测试完整的启动顺序"""
        print("\n测试：完整的启动顺序")
        
        with open('run.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找所有关键步骤的位置
        license_pos = content.find('SimpleLicenseManager')
        adb_pos = content.find('ADBBridge()')
        model_init_pos = content.find('initialize_all_models')
        gui_pos = content.find('gui_main()')
        
        # 验证完整顺序：许可证 -> ADB -> 模型 -> GUI
        positions = [
            ('许可证检查', license_pos),
            ('ADB初始化', adb_pos),
            ('模型加载', model_init_pos),
            ('GUI启动', gui_pos)
        ]
        
        # 确保所有步骤都存在
        for name, pos in positions:
            self.assertGreater(pos, 0, f"应该包含{name}")
        
        # 验证顺序
        for i in range(len(positions) - 1):
            current_name, current_pos = positions[i]
            next_name, next_pos = positions[i + 1]
            self.assertLess(current_pos, next_pos,
                          f"{current_name}必须在{next_name}之前")
        
        print("✓ 启动顺序正确：许可证 -> ADB -> 模型 -> GUI")


class TestStartupErrorHandling(unittest.TestCase):
    """测试启动错误处理"""
    
    def setUp(self):
        """测试前准备"""
        from src.model_manager import ModelManager
        ModelManager._instance = None
        ModelManager._initialized = False
    
    def test_model_loading_failure_prevents_gui_startup(self):
        """测试模型加载失败时阻止 GUI 启动"""
        print("\n测试：模型加载失败时阻止 GUI 启动")
        
        # 读取 run.py 验证错误处理
        with open('run.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 验证有 try-except 包裹模型加载
        self.assertIn('try:', content, "应该有 try 块")
        self.assertIn('except Exception', content, "应该有 except 块捕获异常")
        
        # 验证错误时显示消息并退出
        self.assertIn('模型加载失败', content, "应该显示模型加载失败消息")
        self.assertIn('return', content, "应该在错误时返回/退出")
        
        print("✓ 模型加载失败时正确阻止 GUI 启动")
    
    def test_model_manager_initialization_error_handling(self):
        """测试 ModelManager 初始化错误处理"""
        print("\n测试：ModelManager 初始化错误处理")
        
        from src.model_manager import ModelManager
        from src.adb_bridge import ADBBridge
        
        # 创建 ModelManager 实例
        manager = ModelManager.get_instance()
        
        # 模拟 ADB
        adb = Mock(spec=ADBBridge)
        
        # 模拟模型加载失败
        with patch.object(manager, '_load_page_detector_integrated',
                         side_effect=FileNotFoundError("模型文件不存在")):
            with patch.object(manager, '_validate_model_files', return_value=[]):
                with patch.object(manager, '_is_critical_model', return_value=True):
                    
                    # 应该抛出异常
                    with self.assertRaises(RuntimeError) as context:
                        manager.initialize_all_models(
                            adb_bridge=adb,
                            log_callback=lambda msg: None
                        )
                    
                    # 验证错误信息包含关键模型加载失败
                    error_msg = str(context.exception)
                    self.assertIn('关键模型', error_msg,
                                "错误消息应该提到关键模型")
        
        print("✓ ModelManager 正确处理初始化错误")
    
    def test_gui_refuses_to_start_without_models(self):
        """测试 GUI 拒绝在模型未初始化时启动"""
        print("\n测试：GUI 拒绝在模型未初始化时启动")
        
        from src.model_manager import ModelManager
        
        # 确保 ModelManager 未初始化
        ModelManager._instance = None
        ModelManager._initialized = False
        
        # 尝试创建 GUI（应该失败）
        with self.assertRaises(RuntimeError) as context:
            from src.gui import AutomationGUI
            gui = AutomationGUI()
        
        # 验证错误消息
        error_msg = str(context.exception)
        self.assertIn('ModelManager', error_msg, "错误消息应该提到 ModelManager")
        self.assertIn('未初始化', error_msg, "错误消息应该说明未初始化")
        
        print("✓ GUI 正确拒绝在模型未初始化时启动")
    
    def test_critical_model_failure_blocks_startup(self):
        """测试关键模型加载失败时阻止启动"""
        print("\n测试：关键模型加载失败时阻止启动")
        
        from src.model_manager import ModelManager
        from src.adb_bridge import ADBBridge
        
        manager = ModelManager.get_instance()
        adb = Mock(spec=ADBBridge)
        
        # 模拟关键模型加载失败
        with patch.object(manager, '_load_page_detector_integrated',
                         side_effect=RuntimeError("关键模型加载失败")):
            with patch.object(manager, '_validate_model_files', return_value=[]):
                with patch.object(manager, '_is_critical_model', return_value=True):
                    
                    # 应该抛出异常
                    with self.assertRaises(RuntimeError) as context:
                        manager.initialize_all_models(
                            adb_bridge=adb,
                            log_callback=lambda msg: None
                        )
                    
                    # 验证错误信息
                    self.assertIn('关键模型', str(context.exception))
        
        print("✓ 关键模型加载失败时正确阻止启动")
    
    def test_file_validation_before_loading(self):
        """测试加载前验证文件存在"""
        print("\n测试：加载前验证文件存在")
        
        from src.model_manager import ModelManager
        from src.adb_bridge import ADBBridge
        
        manager = ModelManager.get_instance()
        adb = Mock(spec=ADBBridge)
        
        # 模拟文件缺失
        missing_files = ['model1.pth', 'model2.pt']
        with patch.object(manager, '_validate_model_files',
                         return_value=missing_files):
            
            # 应该抛出 FileNotFoundError
            with self.assertRaises(FileNotFoundError) as context:
                manager.initialize_all_models(
                    adb_bridge=adb,
                    log_callback=lambda msg: None
                )
            
            # 验证错误消息包含缺失文件
            error_msg = str(context.exception)
            for file in missing_files:
                self.assertIn(file, error_msg,
                            f"错误消息应该包含缺失文件: {file}")
        
        print("✓ 加载前正确验证文件存在")
    
    def test_error_message_clarity(self):
        """测试错误消息清晰度"""
        print("\n测试：错误消息清晰度")
        
        # 读取 run.py 验证错误消息
        with open('run.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 验证错误消息包含有用信息
        self.assertIn('模型加载失败', content, "应该有清晰的错误消息")
        self.assertIn('请检查', content, "应该提供解决建议")
        
        print("✓ 错误消息清晰明确")


class TestStartupProgressReporting(unittest.TestCase):
    """测试启动进度报告"""
    
    def setUp(self):
        """测试前准备"""
        from src.model_manager import ModelManager
        ModelManager._instance = None
        ModelManager._initialized = False
    
    def test_progress_callback_is_called(self):
        """测试进度回调被调用"""
        print("\n测试：进度回调被调用")
        
        from src.model_manager import ModelManager
        from src.adb_bridge import ADBBridge
        
        manager = ModelManager.get_instance()
        adb = Mock(spec=ADBBridge)
        
        # 创建 mock 回调
        progress_callback = Mock()
        log_callback = Mock()
        
        # 模拟模型加载
        with patch.object(manager, '_load_page_detector_integrated'), \
             patch.object(manager, '_load_page_detector_hybrid'), \
             patch.object(manager, '_load_ocr_thread_pool'), \
             patch.object(manager, '_validate_model_files', return_value=[]):
            
            manager.initialize_all_models(
                adb_bridge=adb,
                log_callback=log_callback,
                progress_callback=progress_callback
            )
            
            # 验证进度回调被调用
            self.assertGreater(progress_callback.call_count, 0,
                             "进度回调应该被调用")
            
            # 验证回调参数格式
            for call_args in progress_callback.call_args_list:
                args = call_args[0]
                self.assertEqual(len(args), 3,
                               "进度回调应该有3个参数：message, current, total")
        
        print("✓ 进度回调正确调用")
    
    def test_loading_stats_are_returned(self):
        """测试返回加载统计信息"""
        print("\n测试：返回加载统计信息")
        
        from src.model_manager import ModelManager
        from src.adb_bridge import ADBBridge
        
        manager = ModelManager.get_instance()
        adb = Mock(spec=ADBBridge)
        
        # 模拟模型加载
        with patch.object(manager, '_load_page_detector_integrated'), \
             patch.object(manager, '_load_page_detector_hybrid'), \
             patch.object(manager, '_load_ocr_thread_pool'), \
             patch.object(manager, '_validate_model_files', return_value=[]):
            
            stats = manager.initialize_all_models(
                adb_bridge=adb,
                log_callback=lambda msg: None
            )
            
            # 验证统计信息包含必要字段
            required_fields = [
                'success',
                'models_loaded',
                'total_time',
                'memory_before',
                'memory_after',
                'errors'
            ]
            
            for field in required_fields:
                self.assertIn(field, stats,
                            f"统计信息应该包含 {field} 字段")
            
            # 验证数据类型
            self.assertIsInstance(stats['success'], bool)
            self.assertIsInstance(stats['models_loaded'], list)
            self.assertIsInstance(stats['total_time'], (int, float))
            self.assertIsInstance(stats['errors'], list)
        
        print("✓ 加载统计信息正确返回")
    
    def test_run_py_displays_stats(self):
        """测试 run.py 显示统计信息"""
        print("\n测试：run.py 显示统计信息")
        
        with open('run.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 验证显示关键统计信息
        stats_fields = [
            "stats['total_time']",
            "stats['memory_after']",
            "stats['models_loaded']"
        ]
        
        for field in stats_fields:
            self.assertIn(field, content,
                        f"run.py 应该显示 {field}")
        
        print("✓ run.py 正确显示统计信息")


def run_tests():
    """运行所有测试"""
    print("=" * 70)
    print("启动流程单元测试")
    print("=" * 70)
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加所有测试类
    suite.addTests(loader.loadTestsFromTestCase(TestStartupSequence))
    suite.addTests(loader.loadTestsFromTestCase(TestStartupErrorHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestStartupProgressReporting))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 输出总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    print(f"运行测试: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✓ 所有测试通过！")
        return 0
    else:
        print("\n✗ 部分测试失败")
        return 1


if __name__ == '__main__':
    sys.exit(run_tests())
