"""
简单的单实例限制模块
Simple Single Instance Control Module
"""

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional


class SingleInstance:
    """简单的单实例控制器"""
    
    def __init__(self, app_name: str = "AutoSignInHelper"):
        """初始化单实例控制器
        
        Args:
            app_name: 应用程序名称
        """
        self.app_name = app_name
        self.lock_file: Optional[Path] = None
        self.lock_handle = None
        
        # 在临时目录创建锁文件
        temp_dir = Path(tempfile.gettempdir())
        self.lock_file = temp_dir / f"{app_name}.lock"
    
    def is_running(self) -> bool:
        """检查程序是否已在运行
        
        Returns:
            bool: True 如果已在运行，False 如果未运行
        """
        if not self.lock_file.exists():
            return False
        
        try:
            # 读取锁文件中的进程ID
            with open(self.lock_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    # 空文件，删除并返回未运行
                    self.lock_file.unlink()
                    return False
                
                pid = int(content)
                
                # 检查进程是否存在
                if self._is_process_alive(pid):
                    return True
                else:
                    # 进程不存在，删除残留锁文件
                    self.lock_file.unlink()
                    return False
                    
        except (ValueError, FileNotFoundError, PermissionError):
            # 文件格式错误或无法访问，删除并返回未运行
            try:
                self.lock_file.unlink()
            except:
                pass
            return False
    
    def acquire_lock(self) -> bool:
        """获取单实例锁
        
        Returns:
            bool: True 如果成功获取锁，False 如果程序已在运行
        """
        if self.is_running():
            return False
        
        try:
            # 创建锁文件并写入当前进程ID
            with open(self.lock_file, 'w', encoding='utf-8') as f:
                f.write(str(os.getpid()))
            
            return True
            
        except Exception as e:
            print(f"获取单实例锁失败: {e}")
            return False
    
    def release_lock(self):
        """释放单实例锁"""
        try:
            if self.lock_file and self.lock_file.exists():
                self.lock_file.unlink()
        except Exception as e:
            print(f"释放单实例锁失败: {e}")
    
    def _is_process_alive(self, pid: int) -> bool:
        """检查指定PID的进程是否存在
        
        Args:
            pid: 进程ID
            
        Returns:
            bool: True 如果进程存在，False 如果不存在
        """
        try:
            # 检查PID是否为当前进程
            if pid == os.getpid():
                return False  # 如果是当前进程，说明锁文件是残留的
            
            if sys.platform == 'win32':
                # Windows 平台
                import ctypes
                from ctypes import wintypes
                
                # 打开进程句柄
                PROCESS_QUERY_INFORMATION = 0x0400
                handle = ctypes.windll.kernel32.OpenProcess(
                    PROCESS_QUERY_INFORMATION, False, pid
                )
                
                if handle:
                    # 获取进程退出代码
                    exit_code = wintypes.DWORD()
                    if ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                        ctypes.windll.kernel32.CloseHandle(handle)
                        # STILL_ACTIVE = 259
                        return exit_code.value == 259
                    ctypes.windll.kernel32.CloseHandle(handle)
                
                return False
            else:
                # Unix/Linux 平台
                os.kill(pid, 0)  # 发送信号0，不会杀死进程，只是检查是否存在
                return True
                
        except (OSError, ProcessLookupError):
            return False
        except Exception:
            return False
    
    def __enter__(self):
        """上下文管理器入口"""
        if not self.acquire_lock():
            raise RuntimeError("程序已在运行，无法启动多个实例")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.release_lock()
    
    def __del__(self):
        """析构函数，确保释放锁"""
        self.release_lock()


def check_single_instance(app_name: str = "AutoSignInHelper") -> bool:
    """检查并确保单实例运行
    
    Args:
        app_name: 应用程序名称
        
    Returns:
        bool: True 如果成功获取锁，False 如果程序已在运行
    """
    instance = SingleInstance(app_name)
    return instance.acquire_lock()