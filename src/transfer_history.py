"""
转账历史记录和统计模块
Transfer History and Statistics Module
"""

import sqlite3
import logging
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
        self._init_transfer_logger()
    
    def _init_transfer_logger(self):
        """初始化转账专用日志记录器"""
        # 创建logs目录
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建转账日志记录器
        self.transfer_logger = logging.getLogger("transfer_history")
        self.transfer_logger.setLevel(logging.INFO)
        self.transfer_logger.propagate = False  # 不传播到父logger
        
        # 清除已有的处理器
        self.transfer_logger.handlers.clear()
        
        # 转账日志文件（成功和失败都记录）
        transfer_log_file = log_dir / f"transfer_{datetime.now().strftime('%Y%m%d')}.log"
        transfer_handler = logging.FileHandler(transfer_log_file, encoding='utf-8')
        transfer_handler.setLevel(logging.INFO)
        transfer_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        transfer_handler.setFormatter(transfer_formatter)
        self.transfer_logger.addHandler(transfer_handler)
        
        # 失败日志文件（只记录失败）
        self.failure_logger = logging.getLogger("transfer_failure")
        self.failure_logger.setLevel(logging.ERROR)
        self.failure_logger.propagate = False
        self.failure_logger.handlers.clear()
        
        failure_log_file = log_dir / f"transfer_failure_{datetime.now().strftime('%Y%m%d')}.log"
        failure_handler = logging.FileHandler(failure_log_file, encoding='utf-8')
        failure_handler.setLevel(logging.ERROR)
        failure_handler.setFormatter(transfer_formatter)
        self.failure_logger.addHandler(failure_handler)
    
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
        """保存转账记录（UPSERT逻辑：成功记录覆盖当天的失败记录）
        
        逻辑：
        - 如果当天该账号已有失败记录，成功记录会覆盖它
        - 如果当天该账号已有成功记录，不覆盖（保留成功记录）
        - 失败记录不覆盖任何记录（只在没有记录时插入）
        
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
            
            # 获取今天的日期（用于判断是否当天记录）
            today = datetime.now().date().isoformat()
            
            # 查询当天是否已有该账号的转账记录
            cursor.execute("""
                SELECT id, success FROM transfer_history
                WHERE sender_phone = ? 
                AND DATE(timestamp) = ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (sender_phone, today))
            
            existing = cursor.fetchone()
            
            if existing:
                existing_id, existing_success = existing
                
                if success and existing_success == 0:
                    # 当前是成功记录，且已有失败记录 → 更新为成功
                    cursor.execute("""
                        UPDATE transfer_history SET
                            sender_user_id = ?,
                            sender_name = ?,
                            recipient_phone = ?,
                            recipient_name = ?,
                            amount = ?,
                            strategy = ?,
                            success = 1,
                            error_message = ?,
                            timestamp = ?,
                            owner = ?
                        WHERE id = ?
                    """, (
                        sender_user_id, sender_name,
                        recipient_phone, recipient_name, amount,
                        strategy, "", datetime.now().isoformat(), owner,
                        existing_id
                    ))
                    print(f"[转账历史] 成功记录覆盖当天失败记录: {sender_phone}")
                elif success and existing_success == 1:
                    # 当前是成功记录，已有成功记录 → 不覆盖，保留原记录
                    print(f"[转账历史] 当天已有成功记录，跳过: {sender_phone}")
                    conn.close()
                    return True
                elif not success:
                    # 当前是失败记录 → 不覆盖任何记录，直接插入
                    cursor.execute("""
                        INSERT INTO transfer_history (
                            sender_phone, sender_user_id, sender_name,
                            recipient_phone, recipient_name, amount,
                            strategy, success, error_message, timestamp, owner
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        sender_phone, sender_user_id, sender_name,
                        recipient_phone, recipient_name, amount,
                        strategy, 0, error_message,
                        datetime.now().isoformat(), owner
                    ))
                    print(f"[转账历史] 插入失败记录: {sender_phone}")
            else:
                # 当天没有记录 → 直接插入
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
                print(f"[转账历史] 插入新记录: {sender_phone} ({'成功' if success else '失败'})")
            
            conn.commit()
            conn.close()
            
            # 记录到转账日志文件
            if success:
                self.transfer_logger.info(
                    f"✓ 转账成功 | 发送方: {sender_name}({sender_phone}) | "
                    f"接收方: {recipient_name}({recipient_phone}) | "
                    f"金额: {amount:.2f}元 | 策略: {strategy}"
                )
            else:
                # 同时记录到转账日志和失败日志
                error_log = (
                    f"❌ 转账失败 | 发送方: {sender_name}({sender_phone}) | "
                    f"接收方: {recipient_name}({recipient_phone}) | "
                    f"金额: {amount:.2f}元 | 策略: {strategy} | 错误: {error_message}"
                )
                self.transfer_logger.error(error_log)
                self.failure_logger.error(error_log)
            
            return True
            
        except Exception as e:
            error_msg = f"[转账历史] 保存记录失败: {e}"
            print(error_msg)
            self.failure_logger.error(error_msg)
            return False
    
    def get_recent_transfer(
        self,
        sender_phone: str,
        minutes: int = 5
    ) -> Optional[TransferRecord]:
        """获取最近的转账记录（用于防止重复转账）
        
        Args:
            sender_phone: 发送人手机号
            minutes: 时间范围（分钟），默认5分钟
            
        Returns:
            最近的转账记录，如果没有返回None
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # 计算时间阈值
            threshold = (datetime.now() - timedelta(minutes=minutes)).isoformat()
            
            cursor.execute("""
                SELECT id, sender_phone, sender_user_id, sender_name,
                       recipient_phone, recipient_name, amount, strategy,
                       success, error_message, timestamp, owner
                FROM transfer_history
                WHERE sender_phone = ? AND timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (sender_phone, threshold))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return TransferRecord(
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
                )
            
            return None
            
        except Exception as e:
            print(f"[转账历史] 查询最近转账失败: {e}")
            return None
    
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
