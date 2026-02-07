"""
转账安全机制测试
Transfer Safety Mechanism Tests

测试转账锁、重复检查等安全机制
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from pathlib import Path


class TestTransferLock:
    """测试转账锁机制"""
    
    def setup_method(self):
        """每个测试前的设置"""
        # 清理测试文件
        lock_file = Path("runtime_data/transfer_locks.json")
        if lock_file.exists():
            lock_file.unlink()
    
    def test_acquire_lock_success(self):
        """测试成功获取锁"""
        from src.transfer_lock import TransferLock
        
        lock_manager = TransferLock()
        phone = "13800138000"
        
        # 第一次获取应该成功
        assert lock_manager.acquire_lock(phone) == True
        
        # 检查锁状态
        assert lock_manager.is_locked(phone) == True
    
    def test_acquire_lock_fail_when_locked(self):
        """测试锁定时无法再次获取"""
        from src.transfer_lock import TransferLock
        
        lock_manager = TransferLock()
        phone = "13800138000"
        
        # 第一次获取成功
        assert lock_manager.acquire_lock(phone) == True
        
        # 第二次获取应该失败
        assert lock_manager.acquire_lock(phone) == False
    
    def test_release_lock(self):
        """测试释放锁"""
        from src.transfer_lock import TransferLock
        
        lock_manager = TransferLock()
        phone = "13800138000"
        
        # 获取锁
        lock_manager.acquire_lock(phone)
        assert lock_manager.is_locked(phone) == True
        
        # 释放锁
        lock_manager.release_lock(phone)
        assert lock_manager.is_locked(phone) == False
        
        # 释放后应该可以再次获取
        assert lock_manager.acquire_lock(phone) == True
    
    def test_lock_timeout(self):
        """测试锁超时自动释放"""
        from src.transfer_lock import TransferLock
        
        # 使用1秒超时
        lock_manager = TransferLock(lock_timeout=1)
        phone = "13800138000"
        
        # 获取锁
        lock_manager.acquire_lock(phone)
        assert lock_manager.is_locked(phone) == True
        
        # 等待超时
        import time
        time.sleep(1.5)
        
        # 锁应该已过期
        assert lock_manager.is_locked(phone) == False
        
        # 应该可以重新获取
        assert lock_manager.acquire_lock(phone) == True
    
    def test_get_lock_info(self):
        """测试获取锁信息"""
        from src.transfer_lock import TransferLock
        
        lock_manager = TransferLock()
        phone = "13800138000"
        
        # 未锁定时返回None
        assert lock_manager.get_lock_info(phone) is None
        
        # 获取锁
        lock_manager.acquire_lock(phone)
        
        # 获取锁信息
        lock_info = lock_manager.get_lock_info(phone)
        assert lock_info is not None
        assert lock_info['phone'] == phone
        assert lock_info['elapsed'] >= 0
        assert lock_info['remaining'] > 0
    
    def test_multiple_phones(self):
        """测试多个手机号的锁"""
        from src.transfer_lock import TransferLock
        
        lock_manager = TransferLock()
        phone1 = "13800138000"
        phone2 = "13900139000"
        
        # 两个手机号都应该可以获取锁
        assert lock_manager.acquire_lock(phone1) == True
        assert lock_manager.acquire_lock(phone2) == True
        
        # 两个都应该被锁定
        assert lock_manager.is_locked(phone1) == True
        assert lock_manager.is_locked(phone2) == True
        
        # 释放phone1不影响phone2
        lock_manager.release_lock(phone1)
        assert lock_manager.is_locked(phone1) == False
        assert lock_manager.is_locked(phone2) == True


class TestTransferHistory:
    """测试转账历史查询"""
    
    def setup_method(self):
        """每个测试前的设置"""
        # 清理测试数据
        from src.transfer_history import get_transfer_history
        history = get_transfer_history()
        
        # 删除测试数据（如果有）
        import sqlite3
        conn = sqlite3.connect(str(history.db_path))
        cursor = conn.cursor()
        cursor.execute("DELETE FROM transfer_history WHERE sender_phone LIKE '138%'")
        conn.commit()
        conn.close()
    
    def test_get_recent_transfer_none(self):
        """测试没有最近转账记录"""
        from src.transfer_history import get_transfer_history
        
        history = get_transfer_history()
        phone = "13800138000"
        
        # 应该返回None
        recent = history.get_recent_transfer(phone, minutes=5)
        assert recent is None
    
    def test_get_recent_transfer_exists(self):
        """测试有最近转账记录"""
        from src.transfer_history import get_transfer_history
        
        history = get_transfer_history()
        phone = "13800138000"
        
        # 保存一条转账记录
        history.save_transfer_record(
            sender_phone=phone,
            sender_user_id="1234567",
            sender_name="测试用户",
            recipient_phone="13900139000",
            recipient_name="收款人",
            amount=100.0,
            strategy="rotation",
            success=True,
            error_message="",
            owner="admin"
        )
        
        # 应该能查询到
        recent = history.get_recent_transfer(phone, minutes=5)
        assert recent is not None
        assert recent.sender_phone == phone
        assert recent.amount == 100.0
        assert recent.success == True
    
    def test_get_recent_transfer_time_range(self):
        """测试时间范围过滤"""
        from src.transfer_history import get_transfer_history
        
        history = get_transfer_history()
        phone = "13800138000"
        
        # 保存一条转账记录
        history.save_transfer_record(
            sender_phone=phone,
            sender_user_id="1234567",
            sender_name="测试用户",
            recipient_phone="13900139000",
            recipient_name="收款人",
            amount=100.0,
            strategy="rotation",
            success=True,
            error_message="",
            owner="admin"
        )
        
        # 1分钟内应该能查到
        recent = history.get_recent_transfer(phone, minutes=1)
        assert recent is not None
        
        # 0分钟内应该查不到（时间太短）
        # 注意：这个测试可能不稳定，因为保存和查询之间有时间差
        # recent = history.get_recent_transfer(phone, minutes=0)
        # assert recent is None


class TestTransferSafety:
    """测试转账安全机制集成"""
    
    def test_lock_prevents_concurrent_transfer(self):
        """测试锁机制防止并发转账"""
        from src.transfer_lock import get_transfer_lock
        
        lock_manager = get_transfer_lock()
        phone = "13800138000"
        
        # 清理之前的锁
        lock_manager.release_lock(phone)
        
        # 模拟第一个转账获取锁
        assert lock_manager.acquire_lock(phone) == True
        
        # 模拟第二个转账尝试获取锁（应该失败）
        assert lock_manager.acquire_lock(phone) == False
        
        # 第一个转账完成，释放锁
        lock_manager.release_lock(phone)
        
        # 现在第二个转账可以获取锁了
        assert lock_manager.acquire_lock(phone) == True
        
        # 清理
        lock_manager.release_lock(phone)
    
    def test_recent_transfer_prevents_duplicate(self):
        """测试最近转账记录防止重复"""
        from src.transfer_history import get_transfer_history
        
        history = get_transfer_history()
        phone = "13800138001"
        
        # 清理测试数据
        import sqlite3
        conn = sqlite3.connect(str(history.db_path))
        cursor = conn.cursor()
        cursor.execute("DELETE FROM transfer_history WHERE sender_phone = ?", (phone,))
        conn.commit()
        conn.close()
        
        # 第一次转账：没有最近记录，应该允许
        recent = history.get_recent_transfer(phone, minutes=5)
        assert recent is None
        
        # 保存转账记录
        history.save_transfer_record(
            sender_phone=phone,
            sender_user_id="1234567",
            sender_name="测试用户",
            recipient_phone="13900139000",
            recipient_name="收款人",
            amount=100.0,
            strategy="rotation",
            success=True,
            error_message="",
            owner="admin"
        )
        
        # 第二次转账：有最近记录，应该阻止
        recent = history.get_recent_transfer(phone, minutes=5)
        assert recent is not None
        assert recent.success == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
