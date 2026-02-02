"""
转账锁管理模块
Transfer Lock Manager Module

防止同一账号的并发转账操作
"""

import json
import time
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime, timedelta


class TransferLock:
    """转账锁管理器
    
    使用文件锁机制防止同一账号的并发转账。
    
    特性：
    - 基于文件的分布式锁
    - 自动超时释放（防止死锁）
    - 锁状态持久化
    
    使用示例：
        lock_manager = TransferLock()
        
        # 尝试获取锁
        if lock_manager.acquire_lock("13800138000"):
            try:
                # 执行转账操作
                result = await transfer_balance(...)
            finally:
                # 释放锁
                lock_manager.release_lock("13800138000")
        else:
            print("转账进行中，跳过")
    """
    
    def __init__(self, lock_timeout: int = 300):
        """初始化锁管理器
        
        Args:
            lock_timeout: 锁超时时间（秒），默认5分钟
        """
        self.lock_file = Path("runtime_data/transfer_locks.json")
        self.lock_timeout = lock_timeout
        self.locks: Dict[str, dict] = self._load_locks()
    
    def acquire_lock(self, phone: str) -> bool:
        """尝试获取转账锁
        
        Args:
            phone: 手机号（用作锁的键）
            
        Returns:
            bool: 是否成功获取锁
        """
        # 清理过期的锁
        self._cleanup_expired_locks()
        
        # 检查是否已有锁
        if phone in self.locks:
            lock_info = self.locks[phone]
            lock_time = datetime.fromisoformat(lock_info['timestamp'])
            
            # 检查锁是否过期
            if datetime.now() - lock_time < timedelta(seconds=self.lock_timeout):
                # 锁未过期，无法获取
                return False
            else:
                # 锁已过期，可以重新获取
                print(f"  [转账锁] ⚠️ 检测到过期锁，自动释放: {phone}")
        
        # 获取锁
        self.locks[phone] = {
            'timestamp': datetime.now().isoformat(),
            'timeout': self.lock_timeout
        }
        
        # 保存锁状态
        self._save_locks()
        
        return True
    
    def release_lock(self, phone: str):
        """释放转账锁
        
        Args:
            phone: 手机号
        """
        if phone in self.locks:
            del self.locks[phone]
            self._save_locks()
    
    def is_locked(self, phone: str) -> bool:
        """检查是否已锁定
        
        Args:
            phone: 手机号
            
        Returns:
            bool: 是否已锁定
        """
        # 清理过期的锁
        self._cleanup_expired_locks()
        
        if phone not in self.locks:
            return False
        
        lock_info = self.locks[phone]
        lock_time = datetime.fromisoformat(lock_info['timestamp'])
        
        # 检查锁是否过期
        return datetime.now() - lock_time < timedelta(seconds=self.lock_timeout)
    
    def get_lock_info(self, phone: str) -> Optional[dict]:
        """获取锁信息
        
        Args:
            phone: 手机号
            
        Returns:
            dict: 锁信息，如果未锁定返回None
        """
        if not self.is_locked(phone):
            return None
        
        lock_info = self.locks[phone]
        lock_time = datetime.fromisoformat(lock_info['timestamp'])
        elapsed = (datetime.now() - lock_time).total_seconds()
        
        return {
            'phone': phone,
            'timestamp': lock_info['timestamp'],
            'elapsed': elapsed,
            'timeout': lock_info['timeout'],
            'remaining': lock_info['timeout'] - elapsed
        }
    
    def _cleanup_expired_locks(self):
        """清理过期的锁"""
        expired_phones = []
        
        for phone, lock_info in self.locks.items():
            lock_time = datetime.fromisoformat(lock_info['timestamp'])
            if datetime.now() - lock_time >= timedelta(seconds=self.lock_timeout):
                expired_phones.append(phone)
        
        for phone in expired_phones:
            del self.locks[phone]
        
        if expired_phones:
            self._save_locks()
    
    def _load_locks(self) -> Dict[str, dict]:
        """加载锁状态
        
        Returns:
            锁状态字典
        """
        try:
            if self.lock_file.exists():
                with open(self.lock_file, 'r', encoding='utf-8') as f:
                    locks = json.load(f)
                    # 验证数据格式
                    if isinstance(locks, dict):
                        return locks
            return {}
        except Exception as e:
            print(f"加载转账锁状态失败: {e}")
            return {}
    
    def _save_locks(self):
        """保存锁状态"""
        try:
            # 确保目录存在
            self.lock_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存状态
            with open(self.lock_file, 'w', encoding='utf-8') as f:
                json.dump(self.locks, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存转账锁状态失败: {e}")
    
    def clear_all_locks(self):
        """清除所有锁（用于调试或紧急情况）"""
        self.locks = {}
        self._save_locks()
        print("  [转账锁] ✓ 已清除所有转账锁")


# 全局实例
_transfer_lock = None


def get_transfer_lock(lock_timeout: int = 300) -> TransferLock:
    """获取转账锁管理器实例
    
    Args:
        lock_timeout: 锁超时时间（秒），默认5分钟
        
    Returns:
        TransferLock实例
    """
    global _transfer_lock
    if _transfer_lock is None:
        _transfer_lock = TransferLock(lock_timeout)
    return _transfer_lock
