"""
转账历史记录和统计模块
Transfer History and Statistics Module
"""

import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TransferRecord:
    """转账记录"""
    id: int
    sender_phone: str
    sender_user_id: str
    sender_name: str
    recipient_phone: str
    recipient_name: str
    amount: float
    strategy: str  # rotation 或 random
    success: bool
    error_message: str
    timestamp: str
    owner: str  # 管理员


class TransferHistory:
    """转账历史记录管理器"""
    
    def __init__(self):
        """初始化转账历史管理器"""
        self.db_path = Path("runtime_data") / "license.db"
        self._init_table()
    
    def _init_table(self):
        """初始化转账历史表"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # 创建转账历史表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transfer_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sender_phone TEXT NOT NULL,
                sender_user_id TEXT NOT NULL,
                sender_name TEXT,
                recipient_phone TEXT NOT NULL,
                recipient_name TEXT,
                amount REAL NOT NULL,
                strategy TEXT NOT NULL,
                success INTEGER NOT NULL,
                error_message TEXT,
                timestamp TEXT NOT NULL,
                owner TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 创建索引
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_transfer_sender 
            ON transfer_history(sender_phone)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_transfer_recipient 
            ON transfer_history(recipient_phone)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_transfer_timestamp 
            ON transfer_history(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_transfer_owner 
            ON transfer_history(owner)
        """)
        
        conn.commit()
        conn.close()
    
    def save_transfer_record(
        self,
        sender_phone: str,
        sender_user_id: str,
        sender_name: str,
        recipient_phone: str,
        recipient_name: str,
        amount: float,
        strategy: str,
        success: bool,
        error_message: str = "",
        owner: str = ""
    ) -> bool:
        """保存转账记录
        
        Args:
            sender_phone: 发送人手机号
            sender_user_id: 发送人用户ID
            sender_name: 发送人姓名
            recipient_phone: 收款人手机号
            recipient_name: 收款人姓名
            amount: 转账金额
            strategy: 选择策略（rotation/random）
            success: 是否成功
            error_message: 错误信息（如果失败）
            owner: 管理员
            
        Returns:
            是否保存成功
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO transfer_history (
                    sender_phone, sender_user_id, sender_name,
                    recipient_phone, recipient_name, amount,
                    strategy, success, error_message, timestamp, owner
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                sender_phone, sender_user_id, sender_name,
                recipient_phone, recipient_name, amount,
                strategy, 1 if success else 0, error_message,
                datetime.now().isoformat(), owner
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"[转账历史] 保存记录失败: {e}")
            return False
    
    def get_transfer_records(
        self,
        sender_phone: str = None,
        recipient_phone: str = None,
        owner: str = None,
        start_date: str = None,
        end_date: str = None,
        limit: int = 100
    ) -> List[TransferRecord]:
        """获取转账记录
        
        Args:
            sender_phone: 发送人手机号（可选）
            recipient_phone: 收款人手机号（可选）
            owner: 管理员（可选）
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）
            limit: 最大记录数
            
        Returns:
            转账记录列表
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # 构建查询条件
            conditions = []
            params = []
            
            if sender_phone:
                conditions.append("sender_phone = ?")
                params.append(sender_phone)
            
            if recipient_phone:
                conditions.append("recipient_phone = ?")
                params.append(recipient_phone)
            
            if owner:
                conditions.append("owner = ?")
                params.append(owner)
            
            if start_date:
                conditions.append("timestamp >= ?")
                params.append(start_date)
            
            if end_date:
                conditions.append("timestamp <= ?")
                params.append(end_date)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            query = f"""
                SELECT id, sender_phone, sender_user_id, sender_name,
                       recipient_phone, recipient_name, amount, strategy,
                       success, error_message, timestamp, owner
                FROM transfer_history
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT ?
            """
            
            params.append(limit)
            cursor.execute(query, params)
            
            records = []
            for row in cursor.fetchall():
                records.append(TransferRecord(
                    id=row[0],
                    sender_phone=row[1],
                    sender_user_id=row[2],
                    sender_name=row[3],
                    recipient_phone=row[4],
                    recipient_name=row[5],
                    amount=row[6],
                    strategy=row[7],
                    success=bool(row[8]),
                    error_message=row[9] or "",
                    timestamp=row[10],
                    owner=row[11] or ""
                ))
            
            conn.close()
            return records
            
        except Exception as e:
            print(f"[转账历史] 获取记录失败: {e}")
            return []
    
    def get_transfer_statistics(
        self,
        sender_phone: str = None,
        recipient_phone: str = None,
        owner: str = None,
        days: int = 30
    ) -> Dict:
        """获取转账统计信息
        
        Args:
            sender_phone: 发送人手机号（可选）
            recipient_phone: 收款人手机号（可选）
            owner: 管理员（可选）
            days: 统计天数（默认30天）
            
        Returns:
            统计信息字典
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # 计算开始日期
            start_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            # 构建查询条件
            conditions = ["timestamp >= ?"]
            params = [start_date]
            
            if sender_phone:
                conditions.append("sender_phone = ?")
                params.append(sender_phone)
            
            if recipient_phone:
                conditions.append("recipient_phone = ?")
                params.append(recipient_phone)
            
            if owner:
                conditions.append("owner = ?")
                params.append(owner)
            
            where_clause = " AND ".join(conditions)
            
            # 总转账次数和金额
            cursor.execute(f"""
                SELECT COUNT(*), SUM(amount), SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END)
                FROM transfer_history
                WHERE {where_clause}
            """, params)
            
            row = cursor.fetchone()
            total_count = row[0] or 0
            total_amount = row[1] or 0.0
            success_count = row[2] or 0
            
            # 按收款人统计
            cursor.execute(f"""
                SELECT recipient_phone, recipient_name, COUNT(*), SUM(amount)
                FROM transfer_history
                WHERE {where_clause} AND success = 1
                GROUP BY recipient_phone
                ORDER BY COUNT(*) DESC
            """, params)
            
            recipient_stats = []
            for row in cursor.fetchall():
                recipient_stats.append({
                    'phone': row[0],
                    'name': row[1] or row[0],
                    'count': row[2],
                    'amount': row[3]
                })
            
            # 按日期统计
            cursor.execute(f"""
                SELECT DATE(timestamp) as date, COUNT(*), SUM(amount)
                FROM transfer_history
                WHERE {where_clause} AND success = 1
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
                LIMIT 30
            """, params)
            
            daily_stats = []
            for row in cursor.fetchall():
                daily_stats.append({
                    'date': row[0],
                    'count': row[1],
                    'amount': row[2]
                })
            
            conn.close()
            
            return {
                'total_count': total_count,
                'success_count': success_count,
                'failed_count': total_count - success_count,
                'success_rate': success_count / total_count * 100 if total_count > 0 else 0,
                'total_amount': total_amount,
                'recipient_stats': recipient_stats,
                'daily_stats': daily_stats,
                'days': days
            }
            
        except Exception as e:
            print(f"[转账历史] 获取统计失败: {e}")
            return {
                'total_count': 0,
                'success_count': 0,
                'failed_count': 0,
                'success_rate': 0,
                'total_amount': 0,
                'recipient_stats': [],
                'daily_stats': [],
                'days': days
            }
    
    def get_recipient_statistics(self, owner: str = None) -> List[Dict]:
        """获取每个收款人的统计信息
        
        Args:
            owner: 管理员（可选）
            
        Returns:
            收款人统计列表
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # 构建查询条件
            where_clause = "success = 1"
            params = []
            
            if owner:
                where_clause += " AND owner = ?"
                params.append(owner)
            
            cursor.execute(f"""
                SELECT 
                    recipient_phone,
                    recipient_name,
                    COUNT(*) as transfer_count,
                    SUM(amount) as total_amount,
                    AVG(amount) as avg_amount,
                    MIN(timestamp) as first_transfer,
                    MAX(timestamp) as last_transfer
                FROM transfer_history
                WHERE {where_clause}
                GROUP BY recipient_phone
                ORDER BY transfer_count DESC
            """, params)
            
            stats = []
            for row in cursor.fetchall():
                stats.append({
                    'phone': row[0],
                    'name': row[1] or row[0],
                    'count': row[2],
                    'total_amount': row[3],
                    'avg_amount': row[4],
                    'first_transfer': row[5],
                    'last_transfer': row[6]
                })
            
            conn.close()
            return stats
            
        except Exception as e:
            print(f"[转账历史] 获取收款人统计失败: {e}")
            return []
    
    def clear_old_records(self, days: int = 90) -> int:
        """清理旧的转账记录
        
        Args:
            days: 保留天数（默认90天）
            
        Returns:
            删除的记录数
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            cursor.execute("""
                DELETE FROM transfer_history
                WHERE timestamp < ?
            """, (cutoff_date,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            print(f"[转账历史] 清理了 {deleted_count} 条旧记录")
            return deleted_count
            
        except Exception as e:
            print(f"[转账历史] 清理旧记录失败: {e}")
            return 0


# 全局实例
_transfer_history = None


def get_transfer_history() -> TransferHistory:
    """获取转账历史管理器实例"""
    global _transfer_history
    if _transfer_history is None:
        _transfer_history = TransferHistory()
    return _transfer_history
