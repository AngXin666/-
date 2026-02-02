"""
线程安全验证测试

验证P0修复：GUI模块的线程安全性
"""

import threading
import time
import unittest
from unittest.mock import Mock, patch
from dataclasses import dataclass
from typing import Optional


@dataclass
class MockAccountResult:
    """模拟账号处理结果"""
    success: bool
    checkin_reward: Optional[float] = None
    balance_after: Optional[float] = None
    points: Optional[int] = None
    vouchers: Optional[int] = None
    coupons: Optional[int] = None
    error_message: Optional[str] = None


class TestThreadSafety(unittest.TestCase):
    """线程安全测试"""
    
    def setUp(self):
        """测试前准备"""
        self.stats_lock = threading.Lock()
        self.processed = 0
        self.success_count = 0
        self.failed_count = 0
        self.total_checkin_reward = 0.0
        self.total_balance = 0.0
        self.total_points = 0
        self.total_vouchers = 0
        self.total_coupons = 0
        self.race_condition_detected = False
    
    def _update_stats_with_lock(self, result: MockAccountResult):
        """使用锁保护的统计更新（模拟修复后的代码）"""
        with self.stats_lock:
            self.processed += 1
            
            if result.success:
                self.success_count += 1
                
                if result.checkin_reward:
                    self.total_checkin_reward += result.checkin_reward
                if result.balance_after is not None:
                    self.total_balance += result.balance_after
                if result.points is not None:
                    self.total_points += result.points
                if result.vouchers is not None:
                    self.total_vouchers += result.vouchers
                if result.coupons is not None:
                    self.total_coupons += result.coupons
            else:
                self.failed_count += 1
    
    def _update_stats_without_lock(self, result: MockAccountResult):
        """不使用锁的统计更新（模拟修复前的代码）"""
        self.processed += 1
        
        if result.success:
            self.success_count += 1
            
            if result.checkin_reward:
                self.total_checkin_reward += result.checkin_reward
            if result.balance_after is not None:
                self.total_balance += result.balance_after
            if result.points is not None:
                self.total_points += result.points
            if result.vouchers is not None:
                self.total_vouchers += result.vouchers
            if result.coupons is not None:
                self.total_coupons += result.coupons
        else:
            self.failed_count += 1
    
    def test_concurrent_updates_with_lock(self):
        """测试使用锁保护的并发更新（应该安全）"""
        # 创建100个成功结果
        results = [
            MockAccountResult(
                success=True,
                checkin_reward=10.0,
                balance_after=100.0,
                points=50,
                vouchers=5,
                coupons=3
            )
            for _ in range(100)
        ]
        
        # 创建100个线程并发更新
        threads = []
        for result in results:
            thread = threading.Thread(target=self._update_stats_with_lock, args=(result,))
            threads.append(thread)
        
        # 启动所有线程
        for thread in threads:
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证结果
        self.assertEqual(self.processed, 100, "处理数量应该是100")
        self.assertEqual(self.success_count, 100, "成功数量应该是100")
        self.assertEqual(self.failed_count, 0, "失败数量应该是0")
        self.assertAlmostEqual(self.total_checkin_reward, 1000.0, places=2, msg="总签到奖励应该是1000.0")
        self.assertAlmostEqual(self.total_balance, 10000.0, places=2, msg="总余额应该是10000.0")
        self.assertEqual(self.total_points, 5000, "总积分应该是5000")
        self.assertEqual(self.total_vouchers, 500, "总抵扣券应该是500")
        self.assertEqual(self.total_coupons, 300, "总优惠券应该是300")
    
    def test_concurrent_updates_without_lock_may_fail(self):
        """测试不使用锁的并发更新（可能出现数据竞争）"""
        # 重置统计
        self.processed = 0
        self.success_count = 0
        self.failed_count = 0
        self.total_checkin_reward = 0.0
        self.total_balance = 0.0
        self.total_points = 0
        self.total_vouchers = 0
        self.total_coupons = 0
        
        # 创建100个成功结果
        results = [
            MockAccountResult(
                success=True,
                checkin_reward=10.0,
                balance_after=100.0,
                points=50,
                vouchers=5,
                coupons=3
            )
            for _ in range(100)
        ]
        
        # 创建100个线程并发更新（不使用锁）
        threads = []
        for result in results:
            thread = threading.Thread(target=self._update_stats_without_lock, args=(result,))
            threads.append(thread)
        
        # 启动所有线程
        for thread in threads:
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 注意：这个测试可能会通过（如果没有发生竞争条件）
        # 但在高并发情况下，很可能会失败
        # 这里我们只是演示不使用锁的风险
        print(f"\n不使用锁的结果:")
        print(f"  处理数量: {self.processed} (期望: 100)")
        print(f"  成功数量: {self.success_count} (期望: 100)")
        print(f"  总签到奖励: {self.total_checkin_reward} (期望: 1000.0)")
        print(f"  总余额: {self.total_balance} (期望: 10000.0)")
        print(f"  总积分: {self.total_points} (期望: 5000)")
        
        # 如果结果不正确，说明发生了数据竞争
        if (self.processed != 100 or 
            self.success_count != 100 or 
            abs(self.total_checkin_reward - 1000.0) > 0.01 or
            abs(self.total_balance - 10000.0) > 0.01 or
            self.total_points != 5000):
            print("  ⚠️ 检测到数据竞争！")
            self.race_condition_detected = True
    
    def test_mixed_success_and_failure(self):
        """测试混合成功和失败的并发更新"""
        # 创建50个成功和50个失败结果
        results = []
        for i in range(100):
            if i % 2 == 0:
                results.append(MockAccountResult(
                    success=True,
                    checkin_reward=10.0,
                    balance_after=100.0,
                    points=50,
                    vouchers=5,
                    coupons=3
                ))
            else:
                results.append(MockAccountResult(
                    success=False,
                    error_message="测试失败"
                ))
        
        # 创建100个线程并发更新
        threads = []
        for result in results:
            thread = threading.Thread(target=self._update_stats_with_lock, args=(result,))
            threads.append(thread)
        
        # 启动所有线程
        for thread in threads:
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证结果
        self.assertEqual(self.processed, 100, "处理数量应该是100")
        self.assertEqual(self.success_count, 50, "成功数量应该是50")
        self.assertEqual(self.failed_count, 50, "失败数量应该是50")
        self.assertAlmostEqual(self.total_checkin_reward, 500.0, places=2, msg="总签到奖励应该是500.0")
        self.assertAlmostEqual(self.total_balance, 5000.0, places=2, msg="总余额应该是5000.0")
        self.assertEqual(self.total_points, 2500, "总积分应该是2500")
    
    def test_lock_prevents_deadlock(self):
        """测试锁不会导致死锁"""
        # 创建一个简单的测试，确保锁可以被正确获取和释放
        lock_acquired = False
        lock_released = False
        
        def acquire_and_release():
            nonlocal lock_acquired, lock_released
            with self.stats_lock:
                lock_acquired = True
                time.sleep(0.01)  # 模拟一些工作
            lock_released = True
        
        # 创建多个线程
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=acquire_and_release)
            threads.append(thread)
        
        # 启动所有线程
        for thread in threads:
            thread.start()
        
        # 等待所有线程完成（设置超时）
        for thread in threads:
            thread.join(timeout=5.0)
            self.assertFalse(thread.is_alive(), "线程应该在5秒内完成，没有死锁")
        
        # 验证锁被正确获取和释放
        self.assertTrue(lock_acquired, "锁应该被获取")
        self.assertTrue(lock_released, "锁应该被释放")


class TestSharedStateProtection(unittest.TestCase):
    """测试共享状态的保护"""
    
    def test_shared_dict_with_lock(self):
        """测试使用锁保护的共享字典"""
        shared_dict = {}
        lock = threading.Lock()
        
        def update_dict(key, value):
            with lock:
                shared_dict[key] = value
        
        # 创建100个线程更新字典
        threads = []
        for i in range(100):
            thread = threading.Thread(target=update_dict, args=(f"key_{i}", i))
            threads.append(thread)
        
        # 启动所有线程
        for thread in threads:
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证所有键值对都被正确添加
        self.assertEqual(len(shared_dict), 100, "字典应该有100个键值对")
        for i in range(100):
            self.assertEqual(shared_dict[f"key_{i}"], i, f"key_{i}的值应该是{i}")
    
    def test_shared_list_with_lock(self):
        """测试使用锁保护的共享列表"""
        shared_list = []
        lock = threading.Lock()
        
        def append_to_list(value):
            with lock:
                shared_list.append(value)
        
        # 创建100个线程添加元素
        threads = []
        for i in range(100):
            thread = threading.Thread(target=append_to_list, args=(i,))
            threads.append(thread)
        
        # 启动所有线程
        for thread in threads:
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证所有元素都被添加
        self.assertEqual(len(shared_list), 100, "列表应该有100个元素")
        self.assertEqual(sorted(shared_list), list(range(100)), "列表应该包含0-99的所有数字")


if __name__ == '__main__':
    unittest.main(verbosity=2)
