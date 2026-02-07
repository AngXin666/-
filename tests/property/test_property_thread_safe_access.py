"""
属性测试：线程安全访问

Feature: model-singleton-optimization
Property 2: 线程安全访问

For any 并发的模型访问请求集合，所有请求都应该成功获取模型实例，
且不会出现竞态条件或死锁。

Validates: Requirements 1.3
"""

import pytest
from hypothesis import given, strategies as st, settings
import threading
import time
import sys
import os
from typing import List, Dict, Any

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.model_manager import ModelManager


class MockModel:
    """模拟模型类，用于测试"""
    def __init__(self, name: str):
        self.name = name
        self.access_count = 0
    
    def access(self):
        """模拟模型访问"""
        self.access_count += 1
        return self.access_count


class TestPropertyThreadSafeAccess:
    """线程安全访问属性测试"""
    
    def setup_method(self):
        """每个测试前重置单例并初始化模拟模型"""
        # 重置单例状态
        ModelManager._instance = None
        ModelManager._initialized = False
        
        # 获取实例并初始化模拟模型
        manager = ModelManager.get_instance()
        
        # 注入模拟模型（用于测试）
        with manager._lock:
            manager._models['page_detector_integrated'] = MockModel('integrated')
            manager._models['page_detector_hybrid'] = MockModel('hybrid')
            manager._models['ocr_thread_pool'] = MockModel('ocr')
    
    @given(st.integers(min_value=2, max_value=50))
    @settings(max_examples=100, deadline=None)
    def test_property_concurrent_model_access(self, num_threads):
        """
        属性测试：并发模型访问
        
        For any 并发的模型访问请求，所有请求都应该成功获取模型实例
        
        测试策略：
        1. 生成随机数量的并发线程（2-50个）
        2. 每个线程随机访问不同的模型
        3. 验证所有访问都成功
        4. 验证没有竞态条件或死锁
        """
        manager = ModelManager.get_instance()
        results = []
        errors = []
        
        def access_random_model(thread_id: int):
            """线程函数：随机访问模型"""
            try:
                # 随机选择要访问的模型
                model_choice = thread_id % 3
                
                if model_choice == 0:
                    model = manager.get_page_detector_integrated()
                elif model_choice == 1:
                    model = manager.get_page_detector_hybrid()
                else:
                    model = manager.get_ocr_thread_pool()
                
                # 记录成功访问
                results.append({
                    'thread_id': thread_id,
                    'model_name': model.name,
                    'model_id': id(model)
                })
            except Exception as e:
                errors.append({
                    'thread_id': thread_id,
                    'error': str(e)
                })
        
        # 创建多个线程并发访问
        threads = [
            threading.Thread(target=access_random_model, args=(i,))
            for i in range(num_threads)
        ]
        
        # 启动所有线程
        for t in threads:
            t.start()
        
        # 等待所有线程完成
        for t in threads:
            t.join()
        
        # 验证：没有错误
        assert len(errors) == 0, \
            f"并发访问模型时出现错误: {errors}"
        
        # 验证：所有线程都成功访问
        assert len(results) == num_threads, \
            f"期望 {num_threads} 个成功访问，但得到 {len(results)} 个"
        
        # 验证：相同模型的所有访问返回同一个实例
        model_instances = {}
        for result in results:
            model_name = result['model_name']
            model_id = result['model_id']
            
            if model_name not in model_instances:
                model_instances[model_name] = model_id
            else:
                assert model_instances[model_name] == model_id, \
                    f"模型 {model_name} 在不同线程中返回了不同的实例"
    
    @given(
        st.integers(min_value=2, max_value=30),
        st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_concurrent_mixed_access(self, num_threads, accesses_per_thread):
        """
        属性测试：并发混合访问
        
        For any 并发的混合模型访问（多次访问多个模型），应该都成功
        
        测试策略：
        1. 生成随机数量的线程（2-30个）
        2. 每个线程进行多次随机模型访问（1-10次）
        3. 验证所有访问都成功
        4. 验证模型实例的一致性
        """
        manager = ModelManager.get_instance()
        results = []
        errors = []
        
        def mixed_access_thread(thread_id: int):
            """线程函数：混合访问多个模型"""
            try:
                thread_results = []
                
                for access_num in range(accesses_per_thread):
                    # 循环访问三种模型
                    model_choice = (thread_id + access_num) % 3
                    
                    if model_choice == 0:
                        model = manager.get_page_detector_integrated()
                        model_name = 'integrated'
                    elif model_choice == 1:
                        model = manager.get_page_detector_hybrid()
                        model_name = 'hybrid'
                    else:
                        model = manager.get_ocr_thread_pool()
                        model_name = 'ocr'
                    
                    thread_results.append({
                        'model_name': model_name,
                        'model_id': id(model)
                    })
                
                results.append({
                    'thread_id': thread_id,
                    'accesses': thread_results
                })
            except Exception as e:
                errors.append({
                    'thread_id': thread_id,
                    'error': str(e)
                })
        
        # 创建多个线程
        threads = [
            threading.Thread(target=mixed_access_thread, args=(i,))
            for i in range(num_threads)
        ]
        
        # 启动所有线程
        for t in threads:
            t.start()
        
        # 等待所有线程完成
        for t in threads:
            t.join()
        
        # 验证：没有错误
        assert len(errors) == 0, \
            f"并发混合访问时出现错误: {errors}"
        
        # 验证：所有线程都完成了预期的访问次数
        assert len(results) == num_threads
        for result in results:
            assert len(result['accesses']) == accesses_per_thread, \
                f"线程 {result['thread_id']} 应该访问 {accesses_per_thread} 次，" \
                f"但实际访问了 {len(result['accesses'])} 次"
        
        # 验证：每个模型的所有访问返回同一个实例
        model_instances = {}
        for result in results:
            for access in result['accesses']:
                model_name = access['model_name']
                model_id = access['model_id']
                
                if model_name not in model_instances:
                    model_instances[model_name] = model_id
                else:
                    assert model_instances[model_name] == model_id, \
                        f"模型 {model_name} 返回了不同的实例"
    
    @given(st.integers(min_value=10, max_value=100))
    @settings(max_examples=50, deadline=None)
    def test_property_no_deadlock(self, num_threads):
        """
        属性测试：无死锁
        
        For any 大量并发访问，系统不应该出现死锁
        
        测试策略：
        1. 生成大量并发线程（10-100个）
        2. 使用同步屏障确保所有线程同时开始
        3. 设置超时时间检测死锁
        4. 验证所有线程都能在合理时间内完成
        """
        manager = ModelManager.get_instance()
        results = []
        errors = []
        barrier = threading.Barrier(num_threads)
        timeout_seconds = 10  # 10秒超时
        
        def stress_access_thread(thread_id: int):
            """压力测试线程函数"""
            try:
                # 等待所有线程就绪
                barrier.wait()
                
                # 同时访问所有模型
                integrated = manager.get_page_detector_integrated()
                hybrid = manager.get_page_detector_hybrid()
                ocr = manager.get_ocr_thread_pool()
                
                results.append({
                    'thread_id': thread_id,
                    'integrated_id': id(integrated),
                    'hybrid_id': id(hybrid),
                    'ocr_id': id(ocr)
                })
            except Exception as e:
                errors.append({
                    'thread_id': thread_id,
                    'error': str(e)
                })
        
        # 创建大量线程
        threads = [
            threading.Thread(target=stress_access_thread, args=(i,))
            for i in range(num_threads)
        ]
        
        # 启动所有线程
        start_time = time.time()
        for t in threads:
            t.start()
        
        # 等待所有线程完成（带超时）
        for t in threads:
            remaining_time = timeout_seconds - (time.time() - start_time)
            if remaining_time <= 0:
                pytest.fail(f"检测到死锁：{num_threads} 个线程在 {timeout_seconds} 秒内未完成")
            t.join(timeout=remaining_time)
            
            if t.is_alive():
                pytest.fail(f"检测到死锁：线程 {t.name} 在超时后仍在运行")
        
        # 验证：没有错误
        assert len(errors) == 0, \
            f"压力测试中出现错误: {errors}"
        
        # 验证：所有线程都完成了
        assert len(results) == num_threads, \
            f"期望 {num_threads} 个线程完成，但只有 {len(results)} 个完成"
        
        # 验证：所有线程获取的是相同的模型实例
        if results:
            first_result = results[0]
            for result in results[1:]:
                assert result['integrated_id'] == first_result['integrated_id'], \
                    "不同线程获取的 integrated 模型实例不同"
                assert result['hybrid_id'] == first_result['hybrid_id'], \
                    "不同线程获取的 hybrid 模型实例不同"
                assert result['ocr_id'] == first_result['ocr_id'], \
                    "不同线程获取的 ocr 模型实例不同"
    
    def test_thread_safe_uninitialized_access(self):
        """
        单元测试：未初始化时的线程安全访问
        
        验证在模型未初始化时，并发访问会正确抛出异常
        """
        # 重置单例，不初始化模型
        ModelManager._instance = None
        ModelManager._initialized = False
        manager = ModelManager.get_instance()
        
        errors = []
        num_threads = 10
        
        def access_uninitialized_model():
            """尝试访问未初始化的模型"""
            try:
                manager.get_page_detector_integrated()
                errors.append("应该抛出RuntimeError但没有")
            except RuntimeError as e:
                # 预期的异常
                if "未初始化" not in str(e):
                    errors.append(f"错误消息不正确: {e}")
            except Exception as e:
                errors.append(f"意外的异常类型: {type(e).__name__}: {e}")
        
        # 创建多个线程同时访问未初始化的模型
        threads = [
            threading.Thread(target=access_uninitialized_model)
            for _ in range(num_threads)
        ]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        # 验证：所有线程都正确处理了未初始化的情况
        assert len(errors) == 0, f"线程安全访问未初始化模型时出现问题: {errors}"
    
    def test_thread_safe_lock_acquisition(self):
        """
        单元测试：锁获取的线程安全性
        
        验证多个线程同时获取锁时不会出现问题
        """
        manager = ModelManager.get_instance()
        lock_acquisition_order = []
        lock_release_order = []
        num_threads = 20
        barrier = threading.Barrier(num_threads)
        
        def acquire_and_release_lock(thread_id: int):
            """获取并释放锁"""
            # 等待所有线程就绪
            barrier.wait()
            
            # 获取锁
            with manager._lock:
                lock_acquisition_order.append(thread_id)
                # 模拟一些工作
                time.sleep(0.001)
                lock_release_order.append(thread_id)
        
        # 创建多个线程
        threads = [
            threading.Thread(target=acquire_and_release_lock, args=(i,))
            for i in range(num_threads)
        ]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        # 验证：所有线程都获取并释放了锁
        assert len(lock_acquisition_order) == num_threads
        assert len(lock_release_order) == num_threads
        
        # 验证：获取和释放的顺序一致（FIFO）
        assert lock_acquisition_order == lock_release_order, \
            "锁的获取和释放顺序不一致，可能存在竞态条件"
    
    def test_concurrent_access_performance(self):
        """
        性能测试：并发访问性能
        
        验证并发访问不会显著降低性能
        """
        manager = ModelManager.get_instance()
        num_threads = 50
        accesses_per_thread = 100
        results = []
        
        def performance_test_thread():
            """性能测试线程"""
            start_time = time.time()
            
            for _ in range(accesses_per_thread):
                # 访问所有三个模型
                manager.get_page_detector_integrated()
                manager.get_page_detector_hybrid()
                manager.get_ocr_thread_pool()
            
            elapsed = time.time() - start_time
            results.append(elapsed)
        
        # 创建多个线程
        threads = [
            threading.Thread(target=performance_test_thread)
            for _ in range(num_threads)
        ]
        
        overall_start = time.time()
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        overall_elapsed = time.time() - overall_start
        
        # 验证：所有线程都完成了
        assert len(results) == num_threads
        
        # 验证：平均每次访问时间应该很短（< 1ms）
        total_accesses = num_threads * accesses_per_thread * 3  # 3个模型
        avg_access_time = overall_elapsed / total_accesses
        
        # 每次访问应该在1毫秒以内
        assert avg_access_time < 0.001, \
            f"平均访问时间过长: {avg_access_time*1000:.3f}ms，可能存在性能问题"
        
        print(f"\n性能统计:")
        print(f"  总线程数: {num_threads}")
        print(f"  每线程访问次数: {accesses_per_thread}")
        print(f"  总访问次数: {total_accesses}")
        print(f"  总耗时: {overall_elapsed:.3f}秒")
        print(f"  平均访问时间: {avg_access_time*1000:.3f}ms")


if __name__ == '__main__':
    # 运行属性测试
    pytest.main([__file__, '-v', '--tb=short', '-s'])
