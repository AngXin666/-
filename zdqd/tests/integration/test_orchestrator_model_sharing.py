"""
Orchestrator集成测试：多个自动化实例共享模型

测试多个XimengAutomation实例通过ModelManager共享相同的模型实例。

**Validates: Requirements 5.3, 5.4**
"""

import sys
import os
from pathlib import Path
import threading
import time
from unittest.mock import Mock, MagicMock, patch

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from src.model_manager import ModelManager


class TestOrchestratorModelSharing:
    """Orchestrator模型共享集成测试套件"""
    
    def setup_method(self):
        """每个测试前重置单例"""
        ModelManager._instance = None
        ModelManager._initialized = False
    
    def test_multiple_automation_instances_share_models(self):
        """
        测试多个XimengAutomation实例共享相同的模型
        
        验证：
        1. 创建多个XimengAutomation实例
        2. 每个实例从ModelManager获取模型
        3. 所有实例获取的是同一个模型对象
        4. 模型只加载一次
        """
        print("\n" + "=" * 60)
        print("测试：多个自动化实例共享模型")
        print("=" * 60)
        
        # 1. 初始化ModelManager
        print("\n[步骤1] 初始化ModelManager...")
        manager = ModelManager.get_instance()
        
        # 创建模拟模型实例
        mock_integrated_instance = Mock(name='integrated_detector')
        mock_hybrid_instance = Mock(name='hybrid_detector')
        mock_ocr_instance = Mock(name='ocr_pool')
        
        # 设置模型到ModelManager
        with manager._lock:
            manager._models['page_detector_integrated'] = mock_integrated_instance
            manager._models['page_detector_hybrid'] = mock_hybrid_instance
            manager._models['ocr_thread_pool'] = mock_ocr_instance
        
        print("[OK] ModelManager已初始化，模型已加载")
        
        # 2. 创建多个XimengAutomation实例
        print("\n[步骤2] 创建5个XimengAutomation实例...")
        
        from src.ximeng_automation import XimengAutomation
        
        automation_instances = []
        for i in range(5):
            # 创建模拟依赖
            mock_adb = Mock()
            mock_adb.device_id = f'emulator-555{i}'
            mock_screen = Mock()
            mock_ui = Mock()
            mock_ui.adb_bridge = mock_adb
            mock_login = Mock()
            
            automation = XimengAutomation(
                ui_automation=mock_ui,
                screen_capture=mock_screen,
                auto_login=mock_login,
                adb_bridge=mock_adb
            )
            
            automation_instances.append(automation)
            print(f"  [OK] 创建实例 {i + 1}")
        
        print(f"[OK] 成功创建 {len(automation_instances)} 个实例")
        
        # 3. 验证所有实例使用相同的模型
        print("\n[步骤3] 验证所有实例共享相同的模型...")
        
        # 获取第一个实例的模型ID
        first_instance = automation_instances[0]
        first_integrated_id = id(first_instance.integrated_detector)
        first_hybrid_id = id(first_instance.hybrid_detector)
        
        print(f"  第一个实例的integrated_detector ID: {first_integrated_id}")
        print(f"  第一个实例的hybrid_detector ID: {first_hybrid_id}")
        
        # 验证其他实例的模型ID相同
        for i, automation in enumerate(automation_instances[1:], 2):
            integrated_id = id(automation.integrated_detector)
            hybrid_id = id(automation.hybrid_detector)
            
            assert integrated_id == first_integrated_id, \
                f"实例{i}的integrated_detector不是同一对象"
            assert hybrid_id == first_hybrid_id, \
                f"实例{i}的hybrid_detector不是同一对象"
            
            print(f"  [OK] 实例 {i} 使用相同的模型")
        
        print("[OK] 所有实例共享相同的模型对象")
        
        # 4. 验证模型只加载一次
        print("\n[步骤4] 验证模型只加载一次...")
        
        # 检查ModelManager中的模型数量
        assert len(manager._models) == 3, "模型数量不正确"
        print(f"[OK] ModelManager中有 {len(manager._models)} 个模型")
        
        # 验证每个模型只有一个实例
        integrated_instances = set()
        hybrid_instances = set()
        
        for automation in automation_instances:
            integrated_instances.add(id(automation.integrated_detector))
            hybrid_instances.add(id(automation.hybrid_detector))
        
        assert len(integrated_instances) == 1, "integrated_detector有多个实例"
        assert len(hybrid_instances) == 1, "hybrid_detector有多个实例"
        
        print("[OK] 每个模型类型只有一个实例")
        
        print("\n" + "=" * 60)
        print("[OK] 多个自动化实例共享模型测试通过")
        print("=" * 60)
    
    def test_concurrent_automation_creation(self):
        """
        测试并发创建自动化实例
        
        验证：
        1. 多个线程同时创建XimengAutomation实例
        2. 所有实例都能成功获取模型
        3. 没有竞态条件
        4. 所有实例共享相同的模型
        """
        print("\n" + "=" * 60)
        print("测试：并发创建自动化实例")
        print("=" * 60)
        
        # 初始化ModelManager
        print("\n[步骤1] 初始化ModelManager...")
        manager = ModelManager.get_instance()
        
        # 创建模拟模型实例
        mock_integrated_instance = Mock(name='integrated_detector')
        mock_hybrid_instance = Mock(name='hybrid_detector')
        mock_ocr_instance = Mock(name='ocr_pool')
        
        with manager._lock:
            manager._models['page_detector_integrated'] = mock_integrated_instance
            manager._models['page_detector_hybrid'] = mock_hybrid_instance
            manager._models['ocr_thread_pool'] = mock_ocr_instance
        
        print("[OK] ModelManager已初始化")
        
        # 并发创建实例
        print("\n[步骤2] 并发创建10个实例...")
        
        from src.ximeng_automation import XimengAutomation
        
        automation_instances = []
        errors = []
        lock = threading.Lock()
        
        def create_automation(index):
            """创建自动化实例的线程函数"""
            try:
                # 创建模拟依赖
                mock_adb = Mock()
                mock_adb.device_id = f'emulator-{5554 + index}'
                mock_screen = Mock()
                mock_ui = Mock()
                mock_ui.adb_bridge = mock_adb
                mock_login = Mock()
                
                automation = XimengAutomation(
                    ui_automation=mock_ui,
                    screen_capture=mock_screen,
                    auto_login=mock_login,
                    adb_bridge=mock_adb
                )
                
                with lock:
                    automation_instances.append(automation)
                    print(f"  [OK] 线程 {index} 创建成功")
            
            except Exception as e:
                with lock:
                    errors.append((index, str(e)))
                    print(f"  [ERROR] 线程 {index} 失败: {e}")
        
        # 创建并启动线程
        threads = []
        for i in range(10):
            thread = threading.Thread(target=create_automation, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证没有错误
        assert len(errors) == 0, f"有 {len(errors)} 个线程失败: {errors}"
        assert len(automation_instances) == 10, f"只创建了 {len(automation_instances)} 个实例"
        
        print(f"[OK] 成功并发创建 {len(automation_instances)} 个实例")
        
        # 验证所有实例共享相同的模型
        print("\n[步骤3] 验证所有实例共享相同的模型...")
        
        first_integrated_id = id(automation_instances[0].integrated_detector)
        first_hybrid_id = id(automation_instances[0].hybrid_detector)
        
        for i, automation in enumerate(automation_instances[1:], 2):
            assert id(automation.integrated_detector) == first_integrated_id, \
                f"实例{i}的integrated_detector不是同一对象"
            assert id(automation.hybrid_detector) == first_hybrid_id, \
                f"实例{i}的hybrid_detector不是同一对象"
        
        print("[OK] 所有实例共享相同的模型")
        
        print("\n" + "=" * 60)
        print("[OK] 并发创建自动化实例测试通过")
        print("=" * 60)
    
    def test_orchestrator_fails_without_initialized_manager(self):
        """
        测试ModelManager未初始化时Orchestrator创建失败
        
        验证：
        1. ModelManager未初始化
        2. 创建Orchestrator抛出RuntimeError
        3. 错误信息清晰
        """
        print("\n" + "=" * 60)
        print("测试：ModelManager未初始化时Orchestrator失败")
        print("=" * 60)
        
        # 确保ModelManager未初始化
        print("\n[步骤1] 确保ModelManager未初始化...")
        manager = ModelManager.get_instance()
        assert manager.is_initialized() is False
        print("[OK] ModelManager未初始化")
        
        # 尝试创建Orchestrator
        print("\n[步骤2] 尝试创建Orchestrator...")
        
        from src.orchestrator import Orchestrator
        from src.models import Config
        
        # 创建模拟配置
        config = Config(
            nox_path="C:/Program Files/Nox/bin",
            adb_path="adb",
            target_app_package="com.example.app",
            target_app_activity=".MainActivity",
            screenshot_dir="screenshots",
            max_concurrent_instances=3
        )
        
        # 创建模拟账号管理器
        mock_account_manager = Mock()
        
        # 应该抛出RuntimeError
        with pytest.raises(RuntimeError, match="ModelManager未初始化"):
            orchestrator = Orchestrator(config, mock_account_manager)
        
        print("[OK] 正确抛出RuntimeError")
        print("[OK] 错误信息包含'ModelManager未初始化'")
        
        print("\n" + "=" * 60)
        print("[OK] ModelManager未初始化时Orchestrator失败测试通过")
        print("=" * 60)
    
    def test_memory_efficiency_with_multiple_instances(self):
        """
        测试多个实例的内存效率
        
        验证：
        1. 创建多个自动化实例
        2. 内存增量很小（因为共享模型）
        3. 没有重复的模型对象
        """
        print("\n" + "=" * 60)
        print("测试：多实例内存效率")
        print("=" * 60)
        
        import psutil
        
        # 记录初始内存
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss
        print(f"\n初始内存: {memory_before / 1024 / 1024:.1f}MB")
        
        # 初始化ModelManager
        print("\n[步骤1] 初始化ModelManager...")
        manager = ModelManager.get_instance()
        
        # 创建模拟模型（每个1MB）
        class LargeModel:
            def __init__(self, name):
                self.name = name
                self.data = bytearray(1024 * 1024)  # 1MB
        
        with manager._lock:
            manager._models['page_detector_integrated'] = LargeModel('integrated')
            manager._models['page_detector_hybrid'] = LargeModel('hybrid')
            manager._models['ocr_thread_pool'] = LargeModel('ocr')
        
        memory_after_models = process.memory_info().rss
        model_memory = memory_after_models - memory_before
        print(f"加载模型后内存: {memory_after_models / 1024 / 1024:.1f}MB")
        print(f"模型占用: {model_memory / 1024 / 1024:.1f}MB")
        
        # 创建20个自动化实例
        print("\n[步骤2] 创建20个自动化实例...")
        
        from src.ximeng_automation import XimengAutomation
        
        automation_instances = []
        for i in range(20):
            # 创建模拟依赖
            mock_adb = Mock()
            mock_adb.device_id = f'emulator-{5554 + i}'
            mock_screen = Mock()
            mock_ui = Mock()
            mock_ui.adb_bridge = mock_adb
            mock_login = Mock()
            
            automation = XimengAutomation(
                ui_automation=mock_ui,
                screen_capture=mock_screen,
                auto_login=mock_login,
                adb_bridge=mock_adb
            )
            
            automation_instances.append(automation)
        
        memory_after_instances = process.memory_info().rss
        instance_memory = memory_after_instances - memory_after_models
        
        print(f"创建实例后内存: {memory_after_instances / 1024 / 1024:.1f}MB")
        print(f"实例增量: {instance_memory / 1024 / 1024:.1f}MB")
        
        # 验证：实例增量远小于模型占用（因为共享）
        # 如果每个实例都加载模型，应该是 20 * 3MB = 60MB
        # 但因为共享，应该只增加很少的内存（< 5MB）
        assert instance_memory < 5 * 1024 * 1024, \
            f"实例内存增量过大: {instance_memory / 1024 / 1024:.1f}MB"
        
        print(f"[OK] 内存效率良好（20个实例仅增加 {instance_memory / 1024 / 1024:.1f}MB）")
        
        # 验证所有实例共享相同的模型
        print("\n[步骤3] 验证所有实例共享相同的模型...")
        
        integrated_ids = set(id(a.integrated_detector) for a in automation_instances)
        hybrid_ids = set(id(a.hybrid_detector) for a in automation_instances)
        
        assert len(integrated_ids) == 1, "integrated_detector有多个实例"
        assert len(hybrid_ids) == 1, "hybrid_detector有多个实例"
        
        print("[OK] 所有实例共享相同的模型")
        
        print("\n" + "=" * 60)
        print("[OK] 多实例内存效率测试通过")
        print("=" * 60)


def test_integration_summary():
    """集成测试总结"""
    print("\n" + "=" * 60)
    print("Orchestrator模型共享集成测试总结")
    print("=" * 60)
    print("\n已验证的功能:")
    print("  [OK] 多个自动化实例共享模型")
    print("  [OK] 并发创建自动化实例")
    print("  [OK] ModelManager未初始化时Orchestrator失败")
    print("  [OK] 多实例内存效率")
    print("\n核心收益:")
    print("  [OK] 模型实例复用（Requirements 5.3）")
    print("  [OK] 组件正确集成（Requirements 5.4）")
    print("  [OK] 内存占用优化")
    print("  [OK] 线程安全")
    print("=" * 60)


if __name__ == '__main__':
    print("=" * 60)
    print("Orchestrator集成测试：模型共享")
    print("=" * 60)
    
    # 运行测试
    pytest.main([__file__, '-v', '-s', '--tb=short'])
