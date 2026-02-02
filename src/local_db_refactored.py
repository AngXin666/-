"""本地数据库管理（重构版本 - 使用统一错误处理和资源管理）

这是一个示例重构文件，展示如何使用新的错误处理和资源管理模块。
主要改进：
1. 使用 DatabaseConnectionPool 管理连接
2. 使用 @handle_errors 装饰器统一错误处理
3. 使用 @retry 装饰器处理临时性错误
4. 使用 ManagedLock 管理线程锁
"""

import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

# 导入新的错误处理和资源管理模块
from error_handling import handle_errors, retry, ErrorSeverity, ErrorContext
from resource_manager import DatabaseConnectionPool, ManagedLock
from logging_config import setup_logger

# 使用统一的日志配置
logger = setup_logger(__name__)


class LocalDatabaseRefactored:
    """本地数据库管理器（重构版本）
    
    使用统一的错误处理和资源管理机制。
    """
    
    def __init__(self):
        """初始化数据库"""
        self.db_path = Path("runtime_data") / "license.db"
        self.backup_dir = Path("runtime_data") / "backups"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # 使用连接池管理数据库连接
        self._connection_pool = DatabaseConnectionPool(
            str(self.db_path),
            max_connections=5,
            timeout=30.0
        )
        
        # 使用 ManagedLock 管理线程锁
        import threading
        self._lock = threading.Lock()
        
        logger.info("初始化数据库管理器")
        
        # 初始化数据库表
        self._init_database()
    
    @handle_errors(
        logger=logger,
        error_message="初始化数据库失败",
        reraise=True,
        severity=ErrorSeverity.CRITICAL
    )
    def _init_database(self):
        """初始化数据库表（使用统一错误处理）"""
        with ManagedLock(self._lock, timeout=10.0, logger=logger, name="初始化数据库"):
            with self._connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                
                # 创建许可证表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS license (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        license_key_encrypted TEXT NOT NULL,
                        machine_id_encrypted TEXT NOT NULL,
                        status TEXT NOT NULL,
                        expires_at TEXT,
                        max_devices INTEGER DEFAULT 1,
                        activated_at TEXT,
                        last_online_check TEXT,
                        last_local_check TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # 创建历史记录表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS history_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        phone TEXT NOT NULL,
                        nickname TEXT,
                        user_id TEXT,
                        balance_before REAL,
                        points INTEGER,
                        vouchers REAL,
                        coupons INTEGER,
                        checkin_reward REAL,
                        checkin_total_times INTEGER,
                        checkin_balance_after REAL,
                        balance_after REAL,
                        transfer_amount REAL DEFAULT 0.0,
                        transfer_recipient TEXT,
                        owner TEXT,
                        duration REAL,
                        status TEXT,
                        login_method TEXT,
                        run_date TEXT NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(phone, run_date)
                    )
                """)
                
                # 创建索引
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_phone ON history_records(phone)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_run_date ON history_records(run_date)
                """)
                
                conn.commit()
                logger.info("数据库表初始化完成")
    
    @retry(
        max_attempts=3,
        delay=0.1,
        backoff=2.0,
        exceptions=(sqlite3.OperationalError,),
        logger=logger
    )
    @handle_errors(
        logger=logger,
        error_message="保存历史记录失败",
        reraise=False,
        default_return=False,
        severity=ErrorSeverity.HIGH
    )
    def save_history_record(self, record: Dict[str, Any]) -> bool:
        """保存历史记录（使用重试机制和统一错误处理）
        
        Args:
            record: 历史记录数据字典
            
        Returns:
            是否成功
        """
        # 验证必填字段
        required_fields = ['phone', 'run_date']
        for field in required_fields:
            if field not in record or record[field] is None:
                logger.error(f"缺少必填字段: {field}")
                return False
        
        with ManagedLock(self._lock, timeout=10.0, logger=logger, name="保存历史记录"):
            with self._connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                
                # 检查是否已存在
                cursor.execute("""
                    SELECT id FROM history_records 
                    WHERE phone = ? AND run_date = ?
                """, (record.get('phone'), record.get('run_date')))
                
                existing = cursor.fetchone()
                
                if existing:
                    # 更新现有记录
                    cursor.execute("""
                        UPDATE history_records SET
                            nickname = ?, user_id = ?, balance_before = ?,
                            points = ?, vouchers = ?, coupons = ?,
                            checkin_reward = ?, checkin_total_times = ?,
                            checkin_balance_after = ?, balance_after = ?,
                            transfer_amount = ?, transfer_recipient = ?,
                            owner = ?, duration = ?, status = ?, login_method = ?
                        WHERE phone = ? AND run_date = ?
                    """, (
                        record.get('nickname'),
                        record.get('user_id'),
                        record.get('balance_before'),
                        record.get('points'),
                        record.get('vouchers'),
                        record.get('coupons'),
                        record.get('checkin_reward'),
                        record.get('checkin_total_times'),
                        record.get('checkin_balance_after'),
                        record.get('balance_after'),
                        record.get('transfer_amount', 0.0),
                        record.get('transfer_recipient'),
                        record.get('owner'),
                        record.get('duration'),
                        record.get('status'),
                        record.get('login_method'),
                        record.get('phone'),
                        record.get('run_date')
                    ))
                    logger.info(f"更新历史记录: {record.get('phone')} - {record.get('run_date')}")
                else:
                    # 插入新记录
                    cursor.execute("""
                        INSERT INTO history_records (
                            phone, nickname, user_id, balance_before, points, vouchers, coupons,
                            checkin_reward, checkin_total_times, checkin_balance_after,
                            balance_after, transfer_amount, transfer_recipient, owner,
                            duration, status, login_method, run_date
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        record.get('phone'),
                        record.get('nickname'),
                        record.get('user_id'),
                        record.get('balance_before'),
                        record.get('points'),
                        record.get('vouchers'),
                        record.get('coupons'),
                        record.get('checkin_reward'),
                        record.get('checkin_total_times'),
                        record.get('checkin_balance_after'),
                        record.get('balance_after'),
                        record.get('transfer_amount', 0.0),
                        record.get('transfer_recipient'),
                        record.get('owner'),
                        record.get('duration'),
                        record.get('status'),
                        record.get('login_method'),
                        record.get('run_date')
                    ))
                    logger.info(f"插入历史记录: {record.get('phone')} - {record.get('run_date')}")
                
                conn.commit()
                return True
    
    @handle_errors(
        logger=logger,
        error_message="查询历史记录失败",
        reraise=False,
        default_return=[],
        severity=ErrorSeverity.MEDIUM
    )
    def get_history_records(
        self,
        phone: str = None,
        start_date: str = None,
        end_date: str = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """查询历史记录（使用统一错误处理）
        
        Args:
            phone: 手机号（可选）
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）
            limit: 返回记录数量限制
            
        Returns:
            历史记录列表
        """
        with ManagedLock(self._lock, timeout=10.0, logger=logger, name="查询历史记录"):
            with self._connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                
                # 构建查询条件
                conditions = ["(status LIKE '%成功%' OR status LIKE '%失败%')"]
                params = []
                
                if phone:
                    conditions.append("phone = ?")
                    params.append(phone)
                
                if start_date:
                    conditions.append("run_date >= ?")
                    params.append(start_date)
                
                if end_date:
                    conditions.append("run_date <= ?")
                    params.append(end_date)
                
                where_clause = " AND ".join(conditions)
                params.append(limit)
                
                query = f"""
                    SELECT phone, nickname, user_id, balance_before, points, vouchers, coupons,
                           checkin_reward, checkin_total_times, checkin_balance_after,
                           balance_after, transfer_amount, transfer_recipient, duration,
                           status, login_method, run_date, created_at, owner
                    FROM history_records
                    WHERE {where_clause}
                    ORDER BY created_at DESC
                    LIMIT ?
                """
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                # 转换为字典列表
                records = []
                for row in rows:
                    records.append({
                        'phone': row[0],
                        'nickname': row[1],
                        'user_id': row[2],
                        'balance_before': row[3],
                        'points': row[4],
                        'vouchers': row[5],
                        'coupons': row[6],
                        'checkin_reward': row[7],
                        'checkin_total_times': row[8],
                        'checkin_balance_after': row[9],
                        'balance_after': row[10],
                        'transfer_amount': row[11],
                        'transfer_recipient': row[12],
                        'duration': row[13],
                        'status': row[14],
                        'login_method': row[15],
                        'run_date': row[16],
                        'created_at': row[17],
                        'owner': row[18]
                    })
                
                logger.info(f"查询到 {len(records)} 条历史记录")
                return records
    
    @handle_errors(
        logger=logger,
        error_message="获取账号汇总失败",
        reraise=False,
        default_return=None,
        severity=ErrorSeverity.MEDIUM
    )
    def get_account_summary(self, phone: str) -> Optional[Dict[str, Any]]:
        """获取指定账号的汇总统计（使用统一错误处理）
        
        Args:
            phone: 手机号
            
        Returns:
            账号汇总信息
        """
        with ManagedLock(self._lock, timeout=10.0, logger=logger, name="获取账号汇总"):
            with self._connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                
                # 获取基本统计
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN status = '成功' THEN 1 ELSE 0 END) as success,
                        SUM(CASE WHEN status = '失败' THEN 1 ELSE 0 END) as fail,
                        SUM(checkin_reward) as total_reward,
                        AVG(checkin_reward) as avg_reward,
                        MIN(run_date) as first_date,
                        MAX(run_date) as latest_date
                    FROM history_records
                    WHERE phone = ?
                """, (phone,))
                
                row = cursor.fetchone()
                if not row or row[0] == 0:
                    return None
                
                # 获取最新记录
                cursor.execute("""
                    SELECT nickname, user_id, balance_after
                    FROM history_records
                    WHERE phone = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                """, (phone,))
                
                latest = cursor.fetchone()
                
                total = row[0]
                success = row[1] or 0
                fail = row[2] or 0
                success_rate = (success / total * 100) if total > 0 else 0
                
                summary = {
                    'phone': phone,
                    'nickname': latest[0] if latest else None,
                    'user_id': latest[1] if latest else None,
                    'total_records': total,
                    'success_count': success,
                    'fail_count': fail,
                    'success_rate': round(success_rate, 2),
                    'total_checkin_reward': round(row[3] or 0, 2),
                    'avg_checkin_reward': round(row[4] or 0, 2),
                    'latest_balance': round(latest[2], 2) if latest and latest[2] else None,
                    'first_date': row[5],
                    'latest_date': row[6]
                }
                
                logger.info(f"获取账号汇总: {phone}, 总记录数: {total}")
                return summary
    
    def close(self):
        """关闭数据库连接池"""
        with ErrorContext(logger, "关闭数据库连接池", reraise=False):
            self._connection_pool.close_all()
            logger.info("数据库连接池已关闭")
    
    def __del__(self):
        """析构函数，确保资源被释放"""
        try:
            self.close()
        except Exception:
            pass


# 使用示例
if __name__ == '__main__':
    # 使用统一的日志配置
    from logging_config import init_logging
    init_logging(log_dir="logs", level="INFO")
    
    # 创建数据库实例
    db = LocalDatabaseRefactored()
    
    # 保存记录（自动重试和错误处理）
    record = {
        'phone': '13800138000',
        'nickname': '测试用户',
        'user_id': 'test123',
        'balance_before': 100.0,
        'checkin_reward': 5.0,
        'balance_after': 105.0,
        'status': '成功',
        'run_date': datetime.now().strftime('%Y-%m-%d')
    }
    
    success = db.save_history_record(record)
    print(f"保存记录: {'成功' if success else '失败'}")
    
    # 查询记录
    records = db.get_history_records(phone='13800138000', limit=10)
    print(f"查询到 {len(records)} 条记录")
    
    # 获取汇总
    summary = db.get_account_summary('13800138000')
    if summary:
        print(f"账号汇总: {summary}")
    
    # 关闭数据库
    db.close()
