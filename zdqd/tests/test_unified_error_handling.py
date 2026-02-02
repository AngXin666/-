"""统一错误处理模块测试

测试 error_handling.py 和 resource_manager.py 的功能。
"""

import pytest
import logging
import threading
import time
import sqlite3
from pathlib import Path
import sys

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from error_handling import (
    handle_errors, retry, with_fallback, safe_execute,
    ErrorContext, ErrorSeverity
)
from resource_manager import (
    DatabaseConnectionPool, managed_file, ManagedLock,
    managed_resource, ResourceTracker
)


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestErrorHandling:
    """测试错误处理装饰器"""
    
    def test_handle_errors_success(self):
        """测试正常执行"""
        @handle_errors(logger, "测试失败", reraise=False, default_return=None)
        def normal_function():
            return "success"
        
        result = normal_function()
        assert result == "success"
    
    def test_handle_errors_with_exception(self):
        """测试异常处理"""
        @handle_errors(logger, "测试失败", reraise=False, default_return="default")
        def failing_function():
            raise ValueError("测试异常")
        
        result = failing_function()
        assert result == "default"
    
    def test_handle_errors_reraise(self):
        """测试重新抛出异常"""
        @handle_errors(logger, "测试失败", reraise=True)
        def failing_function():
            raise ValueError("测试异常")
        
        with pytest.raises(ValueError):
            failing_function()
    
    def test_retry_success(self):
        """测试重试成功"""
        attempt_count = [0]
        
        @retry(max_attempts=3, delay=0.01, backoff=1.0, logger=logger)
        def unstable_function():
            attempt_count[0] += 1
            if attempt_count[0] < 2:
                raise ValueError("临时错误")
            return "success"
        
        result = unstable_function()
        assert result == "success"
        assert attempt_count[0] == 2
    
    def test_retry_failure(self):
        """测试重试失败"""
        @retry(max_attempts=3, delay=0.01, backoff=1.0, logger=logger)
        def always_failing():
            raise ValueError("永久错误")
        
        with pytest.raises(ValueError):
            always_failing()
    
    def test_with_fallback_success(self):
        """测试降级处理成功"""
        def fallback_func():
            return "fallback"
        
        @with_fallback(fallback_func, logger)
        def main_func():
            raise ValueError("主函数失败")
        
        result = main_func()
        assert result == "fallback"
    
    def test_safe_execute(self):
        """测试安全执行"""
        def risky_function(x):
            if x < 0:
                raise ValueError("负数")
            return x * 2
        
        # 正常情况
        result = safe_execute(risky_function, 5, logger=logger, default_return=0)
        assert result == 10
        
        # 异常情况
        result = safe_execute(risky_function, -5, logger=logger, default_return=0)
        assert result == 0
    
    def test_error_context(self):
        """测试错误上下文管理器"""
        # 正常情况
        with ErrorContext(logger, "测试操作", reraise=False):
            x = 1 + 1
        
        # 异常情况（不重新抛出）
        with ErrorContext(logger, "测试操作", reraise=False):
            raise ValueError("测试异常")
        
        # 异常情况（重新抛出）
        with pytest.raises(ValueError):
            with ErrorContext(logger, "测试操作", reraise=True):
                raise ValueError("测试异常")


class TestResourceManager:
    """测试资源管理器"""
    
    def test_database_connection_pool(self, tmp_path):
        """测试数据库连接池"""
        db_path = tmp_path / "test.db"
        pool = DatabaseConnectionPool(str(db_path), max_connections=2)
        
        try:
            # 创建表
            with pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("CREATE TABLE test (id INTEGER, value TEXT)")
                conn.commit()
            
            # 插入数据
            with pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO test VALUES (1, 'test')")
                conn.commit()
            
            # 查询数据
            with pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM test")
                result = cursor.fetchone()
                assert result[0] == 1
                assert result[1] == 'test'
        
        finally:
            pool.close_all()
    
    def test_database_connection_pool_concurrent(self, tmp_path):
        """测试数据库连接池并发访问"""
        db_path = tmp_path / "test_concurrent.db"
        pool = DatabaseConnectionPool(str(db_path), max_connections=3)
        
        try:
            # 创建表
            with pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("CREATE TABLE test (id INTEGER, value TEXT)")
                conn.commit()
            
            # 并发插入数据
            def insert_data(thread_id):
                with pool.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("INSERT INTO test VALUES (?, ?)", 
                                 (thread_id, f"thread_{thread_id}"))
                    conn.commit()
            
            threads = []
            for i in range(5):
                t = threading.Thread(target=insert_data, args=(i,))
                threads.append(t)
                t.start()
            
            for t in threads:
                t.join()
            
            # 验证数据
            with pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM test")
                count = cursor.fetchone()[0]
                assert count == 5
        
        finally:
            pool.close_all()
    
    def test_managed_file(self, tmp_path):
        """测试文件管理器"""
        file_path = tmp_path / "test.txt"
        
        # 写入文件
        with managed_file(str(file_path), 'w', logger=logger) as f:
            f.write("test content")
        
        # 读取文件
        with managed_file(str(file_path), 'r', logger=logger) as f:
            content = f.read()
            assert content == "test content"
    
    def test_managed_lock(self):
        """测试锁管理器"""
        lock = threading.Lock()
        shared_data = {'value': 0}
        
        def increment():
            with ManagedLock(lock, timeout=1.0, logger=logger, name="增加计数"):
                current = shared_data['value']
                time.sleep(0.01)  # 模拟处理时间
                shared_data['value'] = current + 1
        
        # 并发增加
        threads = []
        for _ in range(10):
            t = threading.Thread(target=increment)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert shared_data['value'] == 10
    
    def test_managed_lock_timeout(self):
        """测试锁超时"""
        lock = threading.Lock()
        
        def hold_lock():
            with ManagedLock(lock, timeout=5.0, logger=logger, name="持有锁"):
                time.sleep(0.5)
        
        def try_acquire():
            with ManagedLock(lock, timeout=0.1, logger=logger, name="尝试获取锁"):
                pass
        
        # 启动持有锁的线程
        t1 = threading.Thread(target=hold_lock)
        t1.start()
        
        time.sleep(0.1)  # 确保第一个线程获取了锁
        
        # 尝试获取锁（应该超时）
        with pytest.raises(TimeoutError):
            try_acquire()
        
        t1.join()
    
    def test_managed_resource(self):
        """测试通用资源管理器"""
        resource_state = {'acquired': False, 'released': False}
        
        def acquire():
            resource_state['acquired'] = True
            return "resource"
        
        def release(resource):
            resource_state['released'] = True
        
        with managed_resource(acquire, release, logger, "测试资源") as res:
            assert res == "resource"
            assert resource_state['acquired'] is True
            assert resource_state['released'] is False
        
        assert resource_state['released'] is True
    
    def test_resource_tracker(self):
        """测试资源跟踪器"""
        tracker = ResourceTracker()
        
        # 跟踪资源获取
        id1 = tracker.track_acquire("连接", "conn1")
        id2 = tracker.track_acquire("文件", "file1")
        
        # 检查未释放的资源
        leaks = tracker.get_leaks()
        assert len(leaks) == 2
        
        # 释放一个资源
        tracker.track_release(id1)
        
        # 检查剩余未释放的资源
        leaks = tracker.get_leaks()
        assert len(leaks) == 1
        assert leaks[0]['type'] == "文件"
        
        # 释放所有资源
        tracker.track_release(id2)
        leaks = tracker.get_leaks()
        assert len(leaks) == 0


class TestIntegration:
    """集成测试"""
    
    def test_database_with_error_handling(self, tmp_path):
        """测试数据库操作与错误处理的集成"""
        db_path = tmp_path / "test_integration.db"
        pool = DatabaseConnectionPool(str(db_path), max_connections=2)
        
        try:
            @retry(max_attempts=3, delay=0.01, backoff=1.0, 
                   exceptions=(sqlite3.OperationalError,), logger=logger)
            @handle_errors(logger, "数据库操作失败", reraise=False, default_return=False)
            def create_and_insert():
                with pool.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER, value TEXT)")
                    cursor.execute("INSERT INTO test VALUES (1, 'test')")
                    conn.commit()
                    return True
            
            result = create_and_insert()
            assert result is True
            
            # 验证数据
            with pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM test")
                row = cursor.fetchone()
                assert row[0] == 1
                assert row[1] == 'test'
        
        finally:
            pool.close_all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
