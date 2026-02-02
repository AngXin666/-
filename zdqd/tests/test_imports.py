"""
导入测试 - 确保所有模块可以正确导入
Import Tests - Ensure all modules can be imported correctly

这个测试文件用于防止导入错误，特别是在代码清理后。
"""

import pytest
import sys
import os

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestCoreImports:
    """测试核心模块的导入"""
    
    def test_import_ximeng_automation(self):
        """测试 XimengAutomation 模块可以导入"""
        try:
            from src.ximeng_automation import XimengAutomation
            assert XimengAutomation is not None
        except ImportError as e:
            pytest.fail(f"无法导入 XimengAutomation: {e}")
    
    def test_import_auto_login(self):
        """测试 AutoLogin 模块可以导入"""
        try:
            from src.auto_login import AutoLogin
            assert AutoLogin is not None
        except ImportError as e:
            pytest.fail(f"无法导入 AutoLogin: {e}")
    
    def test_import_gui(self):
        """测试 GUI 模块可以导入"""
        try:
            from src.gui import AutomationGUI
            assert AutomationGUI is not None
        except ImportError as e:
            pytest.fail(f"无法导入 AutomationGUI: {e}")
    
    def test_import_model_manager(self):
        """测试 ModelManager 模块可以导入"""
        try:
            from src.model_manager import ModelManager
            assert ModelManager is not None
        except ImportError as e:
            pytest.fail(f"无法导入 ModelManager: {e}")
    
    def test_import_page_detector(self):
        """测试 PageDetector 模块可以导入"""
        try:
            from src.page_detector import PageDetector, PageState
            assert PageDetector is not None
            assert PageState is not None
        except ImportError as e:
            pytest.fail(f"无法导入 PageDetector: {e}")
    
    def test_import_page_detector_hybrid(self):
        """测试 PageDetectorHybrid 模块可以导入"""
        try:
            from src.page_detector_hybrid import PageDetectorHybrid
            assert PageDetectorHybrid is not None
        except ImportError as e:
            pytest.fail(f"无法导入 PageDetectorHybrid: {e}")
    
    def test_import_page_detector_integrated(self):
        """测试 PageDetectorIntegrated 模块可以导入"""
        try:
            from src.page_detector_integrated import PageDetectorIntegrated
            assert PageDetectorIntegrated is not None
        except ImportError as e:
            pytest.fail(f"无法导入 PageDetectorIntegrated: {e}")


class TestPageStateImports:
    """测试 PageState 可以从正确的模块导入"""
    
    def test_pagestate_from_page_detector(self):
        """测试可以从 page_detector 导入 PageState"""
        try:
            from src.page_detector import PageState
            assert PageState is not None
            # 验证 PageState 有必要的属性
            assert hasattr(PageState, 'HOME')
            assert hasattr(PageState, 'PROFILE')
            assert hasattr(PageState, 'LOGIN')
        except ImportError as e:
            pytest.fail(f"无法从 page_detector 导入 PageState: {e}")
    
    def test_pagestate_not_from_hybrid_optimized(self):
        """测试不应该从 page_detector_hybrid_optimized 导入 PageState（该模块已删除）"""
        with pytest.raises(ImportError):
            from src.page_detector_hybrid_optimized import PageState


class TestCrossModuleImports:
    """测试跨模块导入的一致性"""
    
    def test_auto_login_imports_pagestate_correctly(self):
        """测试 auto_login 正确导入 PageState"""
        try:
            # 读取 auto_login.py 文件
            auto_login_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'auto_login.py')
            with open(auto_login_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查不应该有 page_detector_hybrid_optimized 的导入
            assert 'page_detector_hybrid_optimized' not in content, \
                "auto_login.py 不应该导入 page_detector_hybrid_optimized"
            
            # 检查应该有正确的导入
            assert 'from .page_detector import PageState' in content or \
                   'from src.page_detector import PageState' in content, \
                "auto_login.py 应该从 page_detector 导入 PageState"
        except Exception as e:
            pytest.fail(f"检查 auto_login.py 导入失败: {e}")
    
    def test_gui_imports_pagestate_correctly(self):
        """测试 gui 正确导入 PageState"""
        try:
            # 读取 gui.py 文件
            gui_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'gui.py')
            with open(gui_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查不应该有 page_detector_hybrid_optimized 的导入
            assert 'page_detector_hybrid_optimized' not in content, \
                "gui.py 不应该导入 page_detector_hybrid_optimized"
        except Exception as e:
            pytest.fail(f"检查 gui.py 导入失败: {e}")
    
    def test_model_manager_imports_correctly(self):
        """测试 model_manager 正确导入检测器"""
        try:
            # 读取 model_manager.py 文件
            model_manager_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'model_manager.py')
            with open(model_manager_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查不应该有 PageDetectorHybridOptimized 的导入
            assert 'PageDetectorHybridOptimized' not in content, \
                "model_manager.py 不应该导入 PageDetectorHybridOptimized"
            
            # 检查应该有正确的导入
            assert 'PageDetectorHybrid' in content, \
                "model_manager.py 应该导入 PageDetectorHybrid"
        except Exception as e:
            pytest.fail(f"检查 model_manager.py 导入失败: {e}")


class TestDeprecatedModules:
    """测试已删除的模块不应该被导入"""
    
    def test_no_hybrid_optimized_module(self):
        """测试 page_detector_hybrid_optimized 模块不存在"""
        with pytest.raises(ImportError):
            import src.page_detector_hybrid_optimized
    
    def test_no_references_to_deleted_modules(self):
        """测试源代码中没有引用已删除的模块"""
        src_dir = os.path.join(os.path.dirname(__file__), '..', 'src')
        
        # 要检查的已删除模块
        deleted_modules = [
            'page_detector_hybrid_optimized',
            'PageDetectorHybridOptimized'
        ]
        
        # 扫描所有 Python 文件
        for root, dirs, files in os.walk(src_dir):
            # 跳过 __pycache__ 目录
            if '__pycache__' in root:
                continue
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    for module in deleted_modules:
                        if module in content:
                            # 检查是否在注释中
                            lines = content.split('\n')
                            for i, line in enumerate(lines, 1):
                                if module in line and not line.strip().startswith('#'):
                                    pytest.fail(
                                        f"在 {file_path}:{i} 发现对已删除模块 '{module}' 的引用:\n{line}"
                                    )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
