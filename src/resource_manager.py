"""资源管理模块

提供统一的资源管理上下文管理器，包括数据库连接池、文件操作和锁管理。
"""

import logging
import threading
import sqlite3
from typing import Optional, Any, Callable
from contextlib import contextmanager
from pathlib import Path
import queue
from logging_config import setup_logger


# 使用统一的日志配置
logger = setup_logger(__name__)


class DatabaseConnectionPool:
    """数据库连接池
    
    管理数据库连接的创建、复用和释放，避免频繁创建连接的开销。
    
    Example:
        pool = DatabaseConnectionPool('database.db', max_connections=5)
        
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users")
            results = cursor.fetchall()
    """
    
    def __init__(self, db_path: str, max_connections: int = 5, timeout: float = 15.0):
        """初始化连接池
        
        Args:
            db_path: 数据库文件路径
            max_connections: 最大连接数
            timeout: 获取连接的超时时间（秒）
        """
        self.db_path = db_path
        self.max_connections = max_connections
        self.timeout = timeout
        self._pool: queue.Queue = queue.Queue(maxsize=max_connections)
        self._lock = threading.Lock()
        self._connection_count = 0
        self._closed = False
        
        logger.info(f"初始化数据库连接池: {db_path}, 最大连接数: {max_connections}")
    
    def _create_connection(self) -> sqlite3.Connection:
        """创建新的数据库连接"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        logger.debug(f"创建新的数据库连接: {self.db_path}")
        return conn
    
    def _acquire(self) -> sqlite3.Connection:
        """获取连接"""
        if self._closed:
            raise RuntimeError("连接池已关闭")
        
        try:
            # 尝试从池中获取现有连接
            conn = self._pool.get(block=False)
            logger.debug("从连接池获取连接")
            return conn
        except queue.Empty:
            # 池中没有可用连接
            with self._lock:
                if self._connection_count < self.max_connections:
                    # 可以创建新连接
                    self._connection_count += 1
                    return self._create_connection()
            
            # 达到最大连接数，等待可用连接
            logger.debug(f"等待可用连接（超时: {self.timeout}秒）")
            try:
                conn = self._pool.get(timeout=self.timeout)
                return conn
            except queue.Empty:
                raise TimeoutError(f"获取数据库连接超时（{self.timeout}秒）")
    
    def _release(self, conn: sqlite3.Connection):
        """释放连接回池中"""
        if self._closed:
            conn.close()
            return
        
        try:
            # 回滚任何未提交的事务
            conn.rollback()
            self._pool.put(conn, block=False)
            logger.debug("连接已释放回连接池")
        except queue.Full:
            # 池已满，关闭连接
            conn.close()
            with self._lock:
                self._connection_count -= 1
            logger.debug("连接池已满，关闭连接")
    
    @contextmanager
    def get_connection(self):
        """获取数据库连接（上下文管理器）
        
        Yields:
            sqlite3.Connection: 数据库连接
            
        Example:
            with pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM users")
        """
        conn = self._acquire()
        try:
            yield conn
        except Exception as e:
            # 发生异常时回滚事务
            try:
                conn.rollback()
            except Exception:
                pass
            raise
        finally:
            self._release(conn)
    
    def close_all(self):
        """关闭所有连接"""
        self._closed = True
        
        # 关闭池中的所有连接
        while not self._pool.empty():
            try:
                conn = self._pool.get(block=False)
                conn.close()
            except queue.Empty:
                break
        
        logger.info("数据库连接池已关闭")
    
    def __del__(self):
        """析构函数，确保连接被关闭"""
        if not self._closed:
            self.close_all()


@contextmanager
def managed_file(
    file_path: str,
    mode: str = 'r',
    encoding: str = 'utf-8',
    logger: Optional[logging.Logger] = None
):
    """文件操作上下文管理器
    
    自动处理文件的打开和关闭，确保即使发生异常也能正确关闭文件。
    
    Args:
        file_path: 文件路径
        mode: 打开模式（'r', 'w', 'a', 'rb', 'wb' 等）
        encoding: 文件编码（文本模式时使用）
        logger: 日志记录器（可选）
        
    Yields:
        文件对象
        
    Example:
        with managed_file('data.txt', 'r') as f:
            content = f.read()
    """
    file_obj = None
    try:
        # 确保目录存在
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 打开文件
        if 'b' in mode:
            file_obj = open(file_path, mode)
        else:
            file_obj = open(file_path, mode, encoding=encoding)
        
        if logger:
            logger.debug(f"打开文件: {file_path}, 模式: {mode}")
        
        yield file_obj
        
    except Exception as e:
        if logger:
            logger.error(f"文件操作失败: {file_path}, 错误: {e}", exc_info=True)
        raise
    finally:
        if file_obj is not None:
            try:
                file_obj.close()
                if logger:
                    logger.debug(f"关闭文件: {file_path}")
            except Exception as e:
                if logger:
                    logger.warning(f"关闭文件失败: {file_path}, 错误: {e}")


class ManagedLock:
    """锁管理上下文管理器
    
    提供带超时和日志记录的锁管理。
    
    Example:
        lock = threading.Lock()
        
        with ManagedLock(lock, timeout=5.0, logger=logger, name="数据更新"):
            # 临界区代码
            update_data()
    """
    
    def __init__(
        self,
        lock: threading.Lock,
        timeout: Optional[float] = None,
        logger: Optional[logging.Logger] = None,
        name: str = "未命名锁"
    ):
        """初始化锁管理器
        
        Args:
            lock: 线程锁对象
            timeout: 获取锁的超时时间（秒），None表示无限等待
            logger: 日志记录器（可选）
            name: 锁的名称（用于日志记录）
        """
        self.lock = lock
        self.timeout = timeout
        self.logger = logger
        self.name = name
        self._acquired = False
    
    def __enter__(self):
        """获取锁"""
        if self.logger:
            self.logger.debug(f"尝试获取锁: {self.name}")
        
        if self.timeout is None:
            # 无限等待
            self.lock.acquire()
            self._acquired = True
        else:
            # 带超时的获取
            self._acquired = self.lock.acquire(timeout=self.timeout)
            
            if not self._acquired:
                error_msg = f"获取锁超时: {self.name} (超时: {self.timeout}秒)"
                if self.logger:
                    self.logger.error(error_msg)
                raise TimeoutError(error_msg)
        
        if self.logger:
            self.logger.debug(f"已获取锁: {self.name}")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """释放锁"""
        if self._acquired:
            self.lock.release()
            if self.logger:
                self.logger.debug(f"已释放锁: {self.name}")
        
        return False


@contextmanager
def managed_resource(
    acquire_func: Callable[[], Any],
    release_func: Callable[[Any], None],
    logger: Optional[logging.Logger] = None,
    resource_name: str = "资源"
):
    """通用资源管理上下文管理器
    
    可以管理任何需要获取和释放的资源。
    
    Args:
        acquire_func: 获取资源的函数
        release_func: 释放资源的函数
        logger: 日志记录器（可选）
        resource_name: 资源名称（用于日志记录）
        
    Yields:
        获取的资源
        
    Example:
        def acquire_device():
            return open_device()
        
        def release_device(device):
            device.close()
        
        with managed_resource(acquire_device, release_device, logger, "设备") as device:
            device.do_something()
    """
    resource = None
    try:
        if logger:
            logger.debug(f"获取资源: {resource_name}")
        
        resource = acquire_func()
        
        if logger:
            logger.debug(f"已获取资源: {resource_name}")
        
        yield resource
        
    except Exception as e:
        if logger:
            logger.error(f"资源操作失败: {resource_name}, 错误: {e}", exc_info=True)
        raise
    finally:
        if resource is not None:
            try:
                if logger:
                    logger.debug(f"释放资源: {resource_name}")
                
                release_func(resource)
                
                if logger:
                    logger.debug(f"已释放资源: {resource_name}")
            except Exception as e:
                if logger:
                    logger.warning(f"释放资源失败: {resource_name}, 错误: {e}")


class ResourceTracker:
    """资源跟踪器
    
    跟踪资源的获取和释放，帮助检测资源泄漏。
    
    Example:
        tracker = ResourceTracker()
        
        resource_id = tracker.track_acquire("数据库连接", conn)
        try:
            # 使用资源
            pass
        finally:
            tracker.track_release(resource_id)
        
        # 检查是否有未释放的资源
        leaks = tracker.get_leaks()
    """
    
    def __init__(self):
        """初始化资源跟踪器"""
        self._resources = {}
        self._lock = threading.Lock()
        self._next_id = 0
    
    def track_acquire(self, resource_type: str, resource: Any) -> int:
        """跟踪资源获取
        
        Args:
            resource_type: 资源类型
            resource: 资源对象
            
        Returns:
            资源ID
        """
        with self._lock:
            resource_id = self._next_id
            self._next_id += 1
            
            self._resources[resource_id] = {
                'type': resource_type,
                'resource': resource,
                'acquired_at': threading.current_thread().name
            }
            
            logger.debug(f"跟踪资源获取: ID={resource_id}, 类型={resource_type}")
            return resource_id
    
    def track_release(self, resource_id: int):
        """跟踪资源释放
        
        Args:
            resource_id: 资源ID
        """
        with self._lock:
            if resource_id in self._resources:
                resource_info = self._resources.pop(resource_id)
                logger.debug(f"跟踪资源释放: ID={resource_id}, 类型={resource_info['type']}")
            else:
                logger.warning(f"尝试释放未跟踪的资源: ID={resource_id}")
    
    def get_leaks(self) -> list:
        """获取未释放的资源列表
        
        Returns:
            未释放的资源信息列表
        """
        with self._lock:
            return [
                {
                    'id': rid,
                    'type': info['type'],
                    'acquired_at': info['acquired_at']
                }
                for rid, info in self._resources.items()
            ]
    
    def clear(self):
        """清除所有跟踪记录"""
        with self._lock:
            self._resources.clear()
            logger.debug("清除所有资源跟踪记录")
