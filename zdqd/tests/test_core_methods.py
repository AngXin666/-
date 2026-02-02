"""
核心方法测试 - 确保关键方法存在且可调用
Core Methods Tests - Ensure critical methods exist and are callable

这个测试文件用于防止核心方法被误删除。
"""

import pytest
import sys
import os
import inspect

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestXimengAutomationMethods:
    """测试 XimengAutomation 类的核心方法"""
    
    def test_class_exists(self):
        """测试 XimengAutomation 类存在"""
        from src.ximeng_automation import XimengAutomation
        assert XimengAutomation is not None
    
    def test_run_full_workflow_exists(self):
        """测试 run_full_workflow 方法存在"""
        from src.ximeng_automation import XimengAutomation
        assert hasattr(XimengAutomation, 'run_full_workflow'), \
            "XimengAutomation 必须有 run_full_workflow 方法"
    
    def test_run_full_workflow_is_async(self):
        """测试 run_full_workflow 是异步方法"""
        from src.ximeng_automation import XimengAutomation
        method = getattr(XimengAutomation, 'run_full_workflow')
        assert inspect.iscoroutinefunction(method), \
            "run_full_workflow 必须是异步方法"
    
    def test_run_full_workflow_signature(self):
        """测试 run_full_workflow 方法签名"""
        from src.ximeng_automation import XimengAutomation
        method = getattr(XimengAutomation, 'run_full_workflow')
        sig = inspect.signature(method)
        
        # 检查必需的参数
        params = list(sig.parameters.keys())
        assert 'self' in params, "方法必须有 self 参数"
        assert 'device_id' in params, "方法必须有 device_id 参数"
        assert 'account' in params, "方法必须有 account 参数"
    
    def test_handle_startup_flow_integrated_exists(self):
        """测试 handle_startup_flow_integrated 方法存在"""
        from src.ximeng_automation import XimengAutomation
        assert hasattr(XimengAutomation, 'handle_startup_flow_integrated'), \
            "XimengAutomation 必须有 handle_startup_flow_integrated 方法"
    
    def test_navigate_to_profile_with_ad_handling_exists(self):
        """测试 _navigate_to_profile_with_ad_handling 方法存在"""
        from src.ximeng_automation import XimengAutomation
        assert hasattr(XimengAutomation, '_navigate_to_profile_with_ad_handling'), \
            "XimengAutomation 必须有 _navigate_to_profile_with_ad_handling 方法"
    
    def test_no_deprecated_startup_methods(self):
        """测试已删除的废弃启动方法不存在"""
        from src.ximeng_automation import XimengAutomation
        
        # 这些方法应该已被删除
        deprecated_methods = [
            'handle_startup_flow',  # 已删除的旧方法
            'handle_startup_flow_optimized'  # 已删除的优化方法
        ]
        
        for method_name in deprecated_methods:
            assert not hasattr(XimengAutomation, method_name), \
                f"废弃方法 {method_name} 不应该存在"


class TestGUIMethods:
    """测试 GUI 类的核心方法"""
    
    def test_class_exists(self):
        """测试 AutomationGUI 类存在"""
        from src.gui import AutomationGUI
        assert AutomationGUI is not None
    
    def test_process_account_with_instance_exists(self):
        """测试 _process_account_with_instance 方法存在"""
        from src.gui import AutomationGUI
        assert hasattr(AutomationGUI, '_process_account_with_instance'), \
            "AutomationGUI 必须有 _process_account_with_instance 方法"
    
    def test_process_account_with_instance_has_time_import(self):
        """测试 _process_account_with_instance 方法正确导入 time 模块"""
        # 读取 gui.py 文件
        gui_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'gui.py')
        with open(gui_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找方法定义
        method_start = content.find('async def _process_account_with_instance(')
        assert method_start != -1, "_process_account_with_instance 方法不存在"
        
        # 查找下一个方法定义（方法结束）
        next_method = content.find('\n    async def ', method_start + 1)
        if next_method == -1:
            next_method = content.find('\n    def ', method_start + 1)
        
        method_content = content[method_start:next_method] if next_method != -1 else content[method_start:]
        
        # 检查是否导入了 time 模块
        assert 'import time' in method_content, \
            "_process_account_with_instance 方法必须导入 time 模块"
        
        # 检查是否初始化了 start_time
        assert 'start_time = time.time()' in method_content, \
            "_process_account_with_instance 方法必须初始化 start_time 变量"
    
    def test_no_deprecated_monitored_method(self):
        """测试已删除的废弃测试方法不存在"""
        from src.gui import AutomationGUI
        
        # 这个方法应该已被删除
        assert not hasattr(AutomationGUI, '_process_single_account_monitored'), \
            "废弃方法 _process_single_account_monitored 不应该存在"


class TestModelManagerMethods:
    """测试 ModelManager 类的核心方法"""
    
    def test_class_exists(self):
        """测试 ModelManager 类存在"""
        from src.model_manager import ModelManager
        assert ModelManager is not None
    
    def test_get_instance_exists(self):
        """测试 get_instance 方法存在"""
        from src.model_manager import ModelManager
        assert hasattr(ModelManager, 'get_instance'), \
            "ModelManager 必须有 get_instance 方法（单例模式）"
    
    def test_get_page_detector_integrated_exists(self):
        """测试 get_page_detector_integrated 方法存在"""
        from src.model_manager import ModelManager
        assert hasattr(ModelManager, 'get_page_detector_integrated'), \
            "ModelManager 必须有 get_page_detector_integrated 方法"
    
    def test_get_page_detector_hybrid_exists(self):
        """测试 get_page_detector_hybrid 方法存在"""
        from src.model_manager import ModelManager
        assert hasattr(ModelManager, 'get_page_detector_hybrid'), \
            "ModelManager 必须有 get_page_detector_hybrid 方法"


class TestAutoLoginMethods:
    """测试 AutoLogin 类的核心方法"""
    
    def test_class_exists(self):
        """测试 AutoLogin 类存在"""
        from src.auto_login import AutoLogin
        assert AutoLogin is not None
    
    def test_login_method_exists(self):
        """测试 login 方法存在"""
        from src.auto_login import AutoLogin
        assert hasattr(AutoLogin, 'login'), \
            "AutoLogin 必须有 login 方法"
    
    def test_logout_method_exists(self):
        """测试 logout 方法存在"""
        from src.auto_login import AutoLogin
        assert hasattr(AutoLogin, 'logout'), \
            "AutoLogin 必须有 logout 方法"


class TestMethodDocumentation:
    """测试核心方法是否有文档字符串"""
    
    def test_run_full_workflow_has_docstring(self):
        """测试 run_full_workflow 有文档字符串"""
        from src.ximeng_automation import XimengAutomation
        method = getattr(XimengAutomation, 'run_full_workflow')
        assert method.__doc__ is not None, \
            "run_full_workflow 方法必须有文档字符串"
        assert len(method.__doc__.strip()) > 0, \
            "run_full_workflow 方法的文档字符串不能为空"
    
    def test_handle_startup_flow_integrated_has_docstring(self):
        """测试 handle_startup_flow_integrated 有文档字符串"""
        from src.ximeng_automation import XimengAutomation
        method = getattr(XimengAutomation, 'handle_startup_flow_integrated')
        assert method.__doc__ is not None, \
            "handle_startup_flow_integrated 方法必须有文档字符串"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
