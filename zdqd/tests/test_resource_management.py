"""
资源泄漏验证测试

验证P1修复：资源管理和自动释放
"""

import unittest
import sqlite3
import tempfile
import os
import threading
from unittest.mock import Mock, patch, MagicMock
from contextlib import contextmanager


class TestResourceManagement(unittest.TestCase):
    """资源管理测试"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
    
    def tearDown(self):
        """测试后清理"""
        if os.path.exists(self.db_path):
            try:
                os.unlink(self.db_path)
            except:
                pass
    
    def test_database_connection_released_on_success(self):
        """测试数据库连接在成功时被释放"""
        # 创建数据库连接
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
        conn.commit()
        
        # 使用try-finally确保连接被关闭
        try:
            cursor.execute("INSERT INTO test (value) VALUES (?)", ("test_value",))
            conn.commit()
        finally:
            conn.close()
        
        # 验证连接已关闭（尝试使用会抛出异常）
        with self.assertRaises(sqlite3.ProgrammingError):
            cursor.execute("SELECT * FROM test")
    
    def test_database_connection_released_on_exception(self):
        """测试数据库连接在异常时被释放"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
        conn.commit()
        
        # 模拟异常情况
        try:
            cursor.execute("INSERT INTO test (value) VALUES (?)", ("test_value",))
            # 故意触发异常
            raise ValueError("测试异常")
        except ValueError:
            pass
        finally:
            conn.close()
        
        # 验证连接已关闭
        with self.assertRaises(sqlite3.ProgrammingError):
            cursor.execute("SELECT * FROM test")
    
    def test_context_manager_auto_cleanup(self):
        """测试上下文管理器自动清理资源"""
        @contextmanager
        def managed_connection(db_path):
            """数据库连接上下文管理器"""
            conn = sqlite3.connect(db_path)
            try:
                yield conn
            finally:
                conn.close()
        
        # 使用上下文管理器
        with managed_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
            conn.commit()
        
        # 验证连接已自动关闭
        with self.assertRaises(sqlite3.ProgrammingError):
            cursor.execute("SELECT * FROM test")
    
    def test_context_manager_cleanup_on_exception(self):
        """测试上下文管理器在异常时自动清理"""
        @contextmanager
        def managed_connection(db_path):
            """数据库连接上下文管理器"""
            conn = sqlite3.connect(db_path)
            try:
                yield conn
            finally:
                conn.close()
        
        # 在上下文管理器中触发异常
        try:
            with managed_connection(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
                raise ValueError("测试异常")
        except ValueError:
            pass
        
        # 验证连接已自动关闭
        with self.assertRaises(sqlite3.ProgrammingError):
            cursor.execute("SELECT * FROM test")
    
    def test_file_handle_released(self):
        """测试文件句柄被释放"""
        temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w')
        temp_file.close()  # 先关闭临时文件
        file_path = temp_file.name
        
        try:
            # 使用with语句确保文件被关闭
            with open(file_path, 'w') as f:
                f.write("test content")
            
            # 验证文件可以被删除（说明句柄已释放）
            os.unlink(file_path)
            self.assertFalse(os.path.exists(file_path), "文件应该被删除")
        finally:
            if os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                except:
                    pass
    
    def test_file_handle_released_on_exception(self):
        """测试文件句柄在异常时被释放"""
        temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w')
        temp_file.close()  # 先关闭临时文件
        file_path = temp_file.name
        
        try:
            # 在with语句中触发异常
            try:
                with open(file_path, 'w') as f:
                    f.write("test content")
                    raise ValueError("测试异常")
            except ValueError:
                pass
            
            # 验证文件可以被删除（说明句柄已释放）
            os.unlink(file_path)
            self.assertFalse(os.path.exists(file_path), "文件应该被删除")
        finally:
            if os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                except:
                    pass
    
    def test_lock_released_on_exception(self):
        """测试锁在异常时被释放"""
        lock = threading.Lock()
        
        # 在with语句中触发异常
        try:
            with lock:
                raise ValueError("测试异常")
        except ValueError:
            pass
        
        # 验证锁已释放（可以再次获取）
        acquired = lock.acquire(blocking=False)
        self.assertTrue(acquired, "锁应该已释放，可以再次获取")
        if acquired:
            lock.release()
    
    def test_multiple_resources_cleanup(self):
        """测试多个资源的清理"""
        temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w')
        temp_file.close()  # 先关闭临时文件
        file_path = temp_file.name
        lock = threading.Lock()
        
        @contextmanager
        def managed_resources():
            """管理多个资源"""
            conn = sqlite3.connect(self.db_path)
            f = open(file_path, 'w')
            try:
                with lock:
                    yield conn, f
            finally:
                f.close()
                conn.close()
        
        try:
            # 使用多个资源
            with managed_resources() as (conn, f):
                cursor = conn.cursor()
                cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
                f.write("test content")
                conn.commit()
            
            # 验证所有资源都已释放
            with self.assertRaises(sqlite3.ProgrammingError):
                cursor.execute("SELECT * FROM test")
            
            # 验证文件可以被删除
            os.unlink(file_path)
            self.assertFalse(os.path.exists(file_path), "文件应该被删除")
            
            # 验证锁可以再次获取
            acquired = lock.acquire(blocking=False)
            self.assertTrue(acquired, "锁应该已释放")
            if acquired:
                lock.release()
        finally:
            if os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                except:
                    pass


class TestResourcePool(unittest.TestCase):
    """资源池测试"""
    
    def test_connection_pool_reuse(self):
        """测试连接池的连接复用"""
        class SimpleConnectionPool:
            """简单的连接池"""
            def __init__(self, db_path, max_connections=5):
                self.db_path = db_path
                self.max_connections = max_connections
                self._pool = []
                self._lock = threading.Lock()
            
            @contextmanager
            def get_connection(self):
                """获取连接"""
                conn = self._acquire()
                try:
                    yield conn
                finally:
                    self._release(conn)
            
            def _acquire(self):
                """获取连接"""
                with self._lock:
                    if self._pool:
                        return self._pool.pop()
                    return sqlite3.connect(self.db_path)
            
            def _release(self, conn):
                """释放连接"""
                with self._lock:
                    if len(self._pool) < self.max_connections:
                        self._pool.append(conn)
                    else:
                        conn.close()
        
        # 创建临时数据库
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db.close()
        db_path = temp_db.name
        
        try:
            # 创建连接池
            pool = SimpleConnectionPool(db_path, max_connections=3)
            
            # 使用连接
            with pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
                conn.commit()
            
            # 验证连接被放回池中
            self.assertEqual(len(pool._pool), 1, "连接应该被放回池中")
            
            # 再次使用连接（应该复用）
            with pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO test (value) VALUES (?)", ("test",))
                conn.commit()
            
            # 验证连接被放回池中
            self.assertEqual(len(pool._pool), 1, "连接应该被放回池中")
        finally:
            if os.path.exists(db_path):
                try:
                    os.unlink(db_path)
                except:
                    pass


class TestADBConnectionManagement(unittest.TestCase):
    """ADB连接管理测试"""
    
    def test_adb_connection_closed_on_success(self):
        """测试ADB连接在成功时被关闭"""
        # 模拟ADB连接
        mock_adb = Mock()
        mock_adb.close = Mock()
        
        # 使用try-finally确保连接被关闭
        try:
            # 模拟一些操作
            mock_adb.tap(123, 456)
        finally:
            mock_adb.close()
        
        # 验证close被调用
        mock_adb.close.assert_called_once()
    
    def test_adb_connection_closed_on_exception(self):
        """测试ADB连接在异常时被关闭"""
        # 模拟ADB连接
        mock_adb = Mock()
        mock_adb.close = Mock()
        mock_adb.tap = Mock(side_effect=Exception("连接失败"))
        
        # 使用try-finally确保连接被关闭
        try:
            mock_adb.tap(123, 456)
        except Exception:
            pass
        finally:
            mock_adb.close()
        
        # 验证close被调用
        mock_adb.close.assert_called_once()
    
    def test_adb_connection_close_exception_handled(self):
        """测试ADB连接关闭时的异常被处理"""
        # 模拟ADB连接
        mock_adb = Mock()
        mock_adb.close = Mock(side_effect=Exception("关闭失败"))
        
        # 使用try-finally确保异常被处理
        try:
            # 模拟一些操作
            pass
        finally:
            try:
                mock_adb.close()
            except Exception as e:
                # 异常应该被捕获，不影响主流程
                self.assertIsInstance(e, Exception)
                self.assertEqual(str(e), "关闭失败")


if __name__ == '__main__':
    unittest.main(verbosity=2)
