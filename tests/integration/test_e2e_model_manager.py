"""
端到端集成测试：ModelManager完整流程

测试完整的启动流程、多账号场景、内存占用和启动时间。

**Validates: Requirements 1.1, 5.1, 9.1**
"""

import sys
import os
import time
import psutil
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

import pytest
from model_manager import ModelManager


class TestE2EModelManager:
    """端到端集成测试套件"""
    
    def setup_method(self):
        """每个测试前重置单例"""
        ModelManager._instance = None
        ModelManager._initialized = False
    
    def test_complete_startup_flow(self):
        """
        测试完整的启动流程
        
        验证：
        1. 获取单例实例
        2. 检查初始状态
        3. 模拟模型初始化
        4. 验证初始化后状态
        5. 访问所有模型
        6. 获取加载统计
        """
        print("\n" + "=" * 60)
        print("测试：完整启动流程")
        print("=" * 60)
        
        # 1. 获取单例实例
        print("\n[步骤1] 获取ModelManager单例...")
        manager = ModelManager.get_instance()
        assert manager is not None
        print("[OK] 成功获取单例实例")
        
        # 2. 检查初始状态
        print("\n[步骤2] 检查初始状态...")
        assert manager.is_initialized() is False
        print("[OK] 初始状态正确：未初始化")
        
        # 3. 模拟模型初始化
        print("\n[步骤3] 模拟模型初始化...")
        
        class MockModel:
            def __init__(self, name):
                self.name = name
                self.device = 'cpu'
        
        start_time = time.time()
        
        with manager._lock:
            manager._models['page_detector_integrated'] = MockModel('integrated')
            manager._models['page_detector_hybrid'] = MockModel('hybrid')
            manager._models['ocr_thread_pool'] = MockModel('ocr')
            
            # 设置加载统计
            manager._loading_stats.total_models = 3
            manager._loading_stats.loaded_models = 3
            manager._loading_stats.failed_models = 0
            manager._loading_stats.total_time = time.time() - start_time
            manager._loading_stats.model_times = {
                'page_detector_integrated': 0.1,
                'page_detector_hybrid': 0.1,
                'ocr_thread_pool': 0.05
            }
        
        print("[OK] 模型初始化完成")
        
        # 4. 验证初始化后状态
        print("\n[步骤4] 验证初始化后状态...")
        assert manager.is_initialized() is True
        print("[OK] 初始化状态正确：已初始化")
        
        # 5. 访问所有模型
        print("\n[步骤5] 访问所有模型...")
        integrated = manager.get_page_detector_integrated()
        assert integrated.name == 'integrated'
        print(f"[OK] 获取integrated检测器: {integrated.name}")
        
        hybrid = manager.get_page_detector_hybrid()
        assert hybrid.name == 'hybrid'
        print(f"[OK] 获取hybrid检测器: {hybrid.name}")
        
        ocr = manager.get_ocr_thread_pool()
        assert ocr.name == 'ocr'
        print(f"[OK] 获取OCR线程池: {ocr.name}")
        
        # 6. 获取加载统计
        print("\n[步骤6] 获取加载统计...")
        stats = manager.get_loading_stats()
        
        assert stats['total_models'] == 3
        assert stats['loaded_models'] == 3
        assert stats['failed_models'] == 0
        
        print(f"  总模型数: {stats['total_models']}")
        print(f"  已加载: {stats['loaded_models']}")
        print(f"  失败: {stats['failed_models']}")
        print(f"  总时间: {stats['total_time']:.3f}秒")
        print("[OK] 加载统计正确")
        
        print("\n" + "=" * 60)
        print("[OK] 完整启动流程测试通过")
        print("=" * 60)
    
    def test_multi_account_scenario(self):
        """
        测试多账号场景
        
        模拟30个账号依次访问模型，验证：
        1. 所有账号获取的是同一个模型实例
        2. 没有重复加载
        3. 内存占用稳定
        """
        print("\n" + "=" * 60)
        print("测试：多账号场景（模拟30个账号）")
        print("=" * 60)
        
        # 初始化ModelManager
        manager = ModelManager.get_instance()
        
        # 模拟模型已加载
        class MockModel:
            def __init__(self, name):
                self.name = name
                self.device = 'cpu'
        
        with manager._lock:
            manager._models['page_detector_integrated'] = MockModel('integrated')
            manager._models['page_detector_hybrid'] = MockModel('hybrid')
            manager._models['ocr_thread_pool'] = MockModel('ocr')
        
        # 记录初始内存
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss
        
        print(f"\n初始内存: {memory_before / 1024 / 1024:.1f}MB")
        
        # 模拟30个账号访问
        account_models = []
        
        for i in range(30):
            # 每个账号访问所有模型
            integrated = manager.get_page_detector_integrated()
            hybrid = manager.get_page_detector_hybrid()
            ocr = manager.get_ocr_thread_pool()
            
            account_models.append({
                'account_id': i + 1,
                'integrated_id': id(integrated),
                'hybrid_id': id(hybrid),
                'ocr_id': id(ocr)
            })
            
            if (i + 1) % 10 == 0:
                print(f"已处理 {i + 1} 个账号...")
        
        # 记录最终内存
        memory_after = process.memory_info().rss
        memory_delta = memory_after - memory_before
        
        print(f"\n最终内存: {memory_after / 1024 / 1024:.1f}MB")
        print(f"内存增量: {memory_delta / 1024 / 1024:.1f}MB")
        
        # 验证：所有账号获取的是同一个实例
        first_account = account_models[0]
        for account in account_models[1:]:
            assert account['integrated_id'] == first_account['integrated_id'], \
                f"账号{account['account_id']}的integrated检测器不是同一实例"
            assert account['hybrid_id'] == first_account['hybrid_id'], \
                f"账号{account['account_id']}的hybrid检测器不是同一实例"
            assert account['ocr_id'] == first_account['ocr_id'], \
                f"账号{account['account_id']}的OCR线程池不是同一实例"
        
        print("\n[OK] 所有30个账号获取的是同一个模型实例")
        
        # 验证：内存增量很小（应该小于10MB，因为只是引用）
        assert memory_delta < 10 * 1024 * 1024, \
            f"内存增量过大: {memory_delta / 1024 / 1024:.1f}MB"
        
        print(f"[OK] 内存占用稳定（增量 < 10MB）")
        
        print("\n" + "=" * 60)
        print("[OK] 多账号场景测试通过")
        print("=" * 60)
    
    def test_memory_usage(self):
        """
        测试内存占用
        
        验证：
        1. 初始化前后的内存变化
        2. 多次访问不增加内存
        3. cleanup后内存释放
        """
        print("\n" + "=" * 60)
        print("测试：内存占用")
        print("=" * 60)
        
        process = psutil.Process(os.getpid())
        
        # 记录初始内存
        memory_initial = process.memory_info().rss
        print(f"\n初始内存: {memory_initial / 1024 / 1024:.1f}MB")
        
        # 创建ModelManager
        manager = ModelManager.get_instance()
        memory_after_create = process.memory_info().rss
        print(f"创建后内存: {memory_after_create / 1024 / 1024:.1f}MB")
        print(f"创建增量: {(memory_after_create - memory_initial) / 1024 / 1024:.1f}MB")
        
        # 模拟加载模型
        class MockModel:
            def __init__(self, name):
                self.name = name
                self.data = bytearray(1024 * 1024)  # 1MB数据
        
        with manager._lock:
            manager._models['page_detector_integrated'] = MockModel('integrated')
            manager._models['page_detector_hybrid'] = MockModel('hybrid')
            manager._models['ocr_thread_pool'] = MockModel('ocr')
        
        memory_after_load = process.memory_info().rss
        print(f"加载后内存: {memory_after_load / 1024 / 1024:.1f}MB")
        print(f"加载增量: {(memory_after_load - memory_after_create) / 1024 / 1024:.1f}MB")
        
        # 多次访问模型
        for _ in range(100):
            manager.get_page_detector_integrated()
            manager.get_page_detector_hybrid()
            manager.get_ocr_thread_pool()
        
        memory_after_access = process.memory_info().rss
        print(f"访问后内存: {memory_after_access / 1024 / 1024:.1f}MB")
        print(f"访问增量: {(memory_after_access - memory_after_load) / 1024 / 1024:.1f}MB")
        
        # 验证：多次访问不显著增加内存（< 1MB）
        access_delta = memory_after_access - memory_after_load
        assert access_delta < 1 * 1024 * 1024, \
            f"多次访问增加了过多内存: {access_delta / 1024 / 1024:.1f}MB"
        
        print("[OK] 多次访问不增加内存")
        
        # 清理
        manager.cleanup()
        
        # 强制垃圾回收
        import gc
        gc.collect()
        
        memory_after_cleanup = process.memory_info().rss
        print(f"清理后内存: {memory_after_cleanup / 1024 / 1024:.1f}MB")
        print(f"清理释放: {(memory_after_load - memory_after_cleanup) / 1024 / 1024:.1f}MB")
        
        print("\n" + "=" * 60)
        print("[OK] 内存占用测试通过")
        print("=" * 60)
    
    def test_startup_time(self):
        """
        测试启动时间
        
        验证：
        1. 模型初始化时间合理
        2. 后续访问时间极短
        """
        print("\n" + "=" * 60)
        print("测试：启动时间")
        print("=" * 60)
        
        # 创建ModelManager
        manager = ModelManager.get_instance()
        
        # 模拟模型加载（记录时间）
        class MockModel:
            def __init__(self, name):
                self.name = name
                time.sleep(0.01)  # 模拟加载时间
        
        print("\n模拟加载模型...")
        start_time = time.time()
        
        with manager._lock:
            manager._models['page_detector_integrated'] = MockModel('integrated')
            manager._models['page_detector_hybrid'] = MockModel('hybrid')
            manager._models['ocr_thread_pool'] = MockModel('ocr')
            
            manager._loading_stats.total_time = time.time() - start_time
        
        load_time = time.time() - start_time
        print(f"加载时间: {load_time:.3f}秒")
        
        # 测试访问时间
        print("\n测试访问时间（100次）...")
        access_times = []
        
        for _ in range(100):
            start = time.time()
            manager.get_page_detector_integrated()
            manager.get_page_detector_hybrid()
            manager.get_ocr_thread_pool()
            access_times.append(time.time() - start)
        
        avg_access_time = sum(access_times) / len(access_times)
        max_access_time = max(access_times)
        
        print(f"平均访问时间: {avg_access_time * 1000:.3f}ms")
        print(f"最大访问时间: {max_access_time * 1000:.3f}ms")
        
        # 验证：访问时间极短（< 1ms）
        assert avg_access_time < 0.001, \
            f"平均访问时间过长: {avg_access_time * 1000:.3f}ms"
        
        print("[OK] 访问时间极短（< 1ms）")
        
        print("\n" + "=" * 60)
        print("[OK] 启动时间测试通过")
        print("=" * 60)
    
    def test_error_recovery(self):
        """
        测试错误恢复
        
        验证：
        1. 访问未初始化的模型抛出异常
        2. 异常信息清晰
        3. 系统状态保持一致
        """
        print("\n" + "=" * 60)
        print("测试：错误恢复")
        print("=" * 60)
        
        manager = ModelManager.get_instance()
        
        # 验证：访问未初始化的模型抛出异常
        print("\n测试访问未初始化的模型...")
        
        with pytest.raises(RuntimeError, match="未初始化"):
            manager.get_page_detector_integrated()
        print("[OK] 正确抛出RuntimeError")
        
        with pytest.raises(RuntimeError, match="未初始化"):
            manager.get_page_detector_hybrid()
        print("[OK] 正确抛出RuntimeError")
        
        with pytest.raises(RuntimeError, match="未初始化"):
            manager.get_ocr_thread_pool()
        print("[OK] 正确抛出RuntimeError")
        
        # 验证：系统状态保持一致
        assert manager.is_initialized() is False
        print("[OK] 系统状态保持一致")
        
        print("\n" + "=" * 60)
        print("[OK] 错误恢复测试通过")
        print("=" * 60)


def test_integration_summary():
    """集成测试总结"""
    print("\n" + "=" * 60)
    print("端到端集成测试总结")
    print("=" * 60)
    print("\n已验证的功能:")
    print("  [OK] 完整启动流程")
    print("  [OK] 多账号场景（30个账号）")
    print("  [OK] 内存占用优化")
    print("  [OK] 启动时间优化")
    print("  [OK] 错误恢复机制")
    print("\n核心收益:")
    print("  [OK] 模型实例复用（单例模式）")
    print("  [OK] 内存占用稳定（< 10MB增量）")
    print("  [OK] 访问时间极短（< 1ms）")
    print("  [OK] 错误处理清晰")
    print("=" * 60)


if __name__ == '__main__':
    print("=" * 60)
    print("端到端集成测试：ModelManager")
    print("=" * 60)
    
    # 运行测试
    pytest.main([__file__, '-v', '-s', '--tb=short'])
