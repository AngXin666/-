"""
本地数据库管理
使用 SQLite 存储许可证信息，支持离线使用
敏感字段使用 AES-256 加密
"""

import sqlite3
import json
import shutil
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import threading


class LocalDatabase:
    """本地数据库管理器（加密存储）"""
    
    def __init__(self):
        """初始化数据库"""
        self.db_path = Path("runtime_data") / "license.db"
        self.backup_dir = Path("runtime_data") / "backups"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        
        # 启动时检查数据库完整性
        self._check_database_integrity()
        
        self._init_database()
        
        # 优化 SQLite 配置以支持多线程并发
        self._optimize_sqlite_settings()
    
    def _get_connection(self, timeout=15.0):
        """获取数据库连接（确保UTF-8编码）
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            sqlite3.Connection对象
        """
        conn = sqlite3.connect(str(self.db_path), timeout=timeout)
        # 设置text_factory为str，确保返回的文本是UTF-8字符串
        conn.text_factory = str
        return conn
    
    def _init_database(self):
        """初始化数据库表"""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # 创建许可证表（敏感字段加密存储）
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
            
            # 检查 history_records 表是否存在
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='history_records'
            """)
            table_exists = cursor.fetchone() is not None
            
            if not table_exists:
                # 创建新表（带唯一约束和转账字段）
                cursor.execute("""
                    CREATE TABLE history_records (
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
                print("[数据库] ✅ 创建 history_records 表（带唯一约束和转账字段）")
            else:
                # 表已存在，执行迁移
                self._migrate_add_unique_constraint(cursor)
                # 添加转账字段（如果不存在）
                self._migrate_add_transfer_fields(cursor)
                # 添加管理员字段（如果不存在）
                self._migrate_add_owner_field(cursor)
            
            # 创建索引以提高查询效率
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_phone ON history_records(phone)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_run_date ON history_records(run_date)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON history_records(created_at)
            """)
            
            conn.commit()
            conn.close()
    
    def _optimize_sqlite_settings(self):
        """优化 SQLite 配置以支持多线程并发访问
        
        配置说明：
        1. WAL 模式（Write-Ahead Logging）：允许读写并发，大幅提升并发性能
        2. NORMAL 同步模式：平衡性能和安全性
        3. 增加缓存大小：提高查询性能
        4. 内存临时存储：加快临时表操作
        """
        try:
            with self._lock:
                conn = sqlite3.connect(str(self.db_path), timeout=15.0)
                cursor = conn.cursor()
                
                # 启用 WAL 模式（Write-Ahead Logging）
                # WAL 模式允许读写并发，多个读者可以同时访问数据库
                cursor.execute("PRAGMA journal_mode=WAL")
                
                # 设置同步模式为 NORMAL（平衡性能和安全性）
                # NORMAL 模式在大多数情况下是安全的，且性能更好
                cursor.execute("PRAGMA synchronous=NORMAL")
                
                # 增加缓存大小（单位：页，默认页大小 4KB）
                # -10000 表示 10MB 缓存（负数表示以 KB 为单位）
                cursor.execute("PRAGMA cache_size=-10000")
                
                # 使用内存存储临时表和索引
                cursor.execute("PRAGMA temp_store=MEMORY")
                
                # 设置忙碌超时（毫秒）
                # 当数据库被锁定时，等待最多 15 秒
                cursor.execute("PRAGMA busy_timeout=15000")
                
                conn.commit()
                conn.close()
                
                print("[数据库] ✅ SQLite 优化配置已应用（WAL 模式）")
                
        except Exception as e:
            print(f"[数据库] ⚠️ SQLite 优化配置失败: {e}")
            # 配置失败不影响数据库使用，只是性能可能不是最优
    
    def _migrate_add_unique_constraint(self, cursor) -> None:
        """迁移：为 history_records 表添加唯一约束
        
        策略：
        1. 检查是否已有唯一约束
        2. 如果没有，创建临时表（带约束）
        3. 迁移数据（保留最新记录）
        4. 删除旧表，重命名新表
        
        Args:
            cursor: 数据库游标
        """
        try:
            # 检查是否已有唯一约束
            cursor.execute("""
                SELECT sql FROM sqlite_master 
                WHERE type='table' AND name='history_records'
            """)
            result = cursor.fetchone()
            if not result:
                return
                
            table_sql = result[0]
            
            # 检查是否已有唯一约束
            if 'UNIQUE' in table_sql and 'phone' in table_sql and 'run_date' in table_sql:
                print("[数据库] ✅ 唯一约束已存在，跳过迁移")
                return
            
            print("[数据库] 开始迁移：添加唯一约束...")
            
            # 创建临时表（带唯一约束）
            cursor.execute("""
                CREATE TABLE history_records_new (
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
                    duration REAL,
                    status TEXT,
                    login_method TEXT,
                    run_date TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(phone, run_date)
                )
            """)
            
            # 迁移数据（保留每个 phone+run_date 组合的最新记录）
            cursor.execute("""
                INSERT INTO history_records_new 
                SELECT * FROM history_records
                WHERE id IN (
                    SELECT MAX(id) FROM history_records
                    GROUP BY phone, run_date
                )
                ORDER BY id
            """)
            
            migrated_count = cursor.rowcount
            
            # 删除旧表
            cursor.execute("DROP TABLE history_records")
            
            # 重命名新表
            cursor.execute("ALTER TABLE history_records_new RENAME TO history_records")
            
            print(f"[数据库] ✅ 迁移完成，保留了 {migrated_count} 条记录")
            
        except Exception as e:
            print(f"[数据库] ❌ 迁移失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _migrate_add_transfer_fields(self, cursor) -> None:
        """迁移：为 history_records 表添加转账字段
        
        Args:
            cursor: 数据库游标
        """
        try:
            # 检查是否已有转账字段
            cursor.execute("PRAGMA table_info(history_records)")
            columns = [row[1] for row in cursor.fetchall()]
            
            if 'transfer_amount' not in columns:
                print("[数据库] 添加转账金额字段...")
                cursor.execute("""
                    ALTER TABLE history_records 
                    ADD COLUMN transfer_amount REAL DEFAULT 0.0
                """)
            
            if 'transfer_recipient' not in columns:
                print("[数据库] 添加转账收款人字段...")
                cursor.execute("""
                    ALTER TABLE history_records 
                    ADD COLUMN transfer_recipient TEXT
                """)
            
            print("[数据库] ✅ 转账字段添加完成")
            
        except Exception as e:
            print(f"[数据库] ⚠️ 添加转账字段失败: {e}")
            # 字段添加失败不影响主要功能
    
    def _migrate_add_owner_field(self, cursor) -> None:
        """迁移：为 history_records 表添加管理员字段
        
        Args:
            cursor: 数据库游标
        """
        try:
            # 检查是否已有管理员字段
            cursor.execute("PRAGMA table_info(history_records)")
            columns = [row[1] for row in cursor.fetchall()]
            
            if 'owner' not in columns:
                print("[数据库] 添加管理员字段...")
                cursor.execute("""
                    ALTER TABLE history_records 
                    ADD COLUMN owner TEXT
                """)
                print("[数据库] ✅ 管理员字段添加完成")
            else:
                print("[数据库] ✅ 管理员字段已存在")
            
        except Exception as e:
            print(f"[数据库] ⚠️ 添加管理员字段失败: {e}")
            # 字段添加失败不影响主要功能
    
    def _check_database_integrity(self):
        """检查数据库完整性，如果损坏则尝试从备份恢复"""
        if not self.db_path.exists():
            print("[数据库] 数据库文件不存在，将创建新数据库")
            return
        
        conn = None
        try:
            # 尝试打开数据库并执行完整性检查
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchone()
            
            if result and result[0] == 'ok':
                print("[数据库] ✅ 数据库完整性检查通过")
                return
            else:
                print(f"[数据库] ❌ 数据库完整性检查失败: {result}")
                
        except Exception as e:
            print(f"[数据库] ❌ 数据库损坏: {e}")
            
        finally:
            # 确保关闭连接
            if conn:
                try:
                    conn.close()
                except:
                    pass
        
        # 如果到这里，说明数据库有问题，尝试恢复
        self._restore_from_backup()
    
    def _backup_database(self):
        """备份数据库文件"""
        try:
            if not self.db_path.exists():
                return
            
            # 生成备份文件名（带时间戳）
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = self.backup_dir / f"license_backup_{timestamp}.db"
            
            # 复制数据库文件
            shutil.copy2(self.db_path, backup_file)
            print(f"[数据库] ✅ 备份成功: {backup_file.name}")
            
            # 清理旧备份（保留最近 10 个）
            self._cleanup_old_backups(keep=10)
            
        except Exception as e:
            print(f"[数据库] ❌ 备份失败: {e}")
    
    def _restore_from_backup(self):
        """从最新的备份恢复数据库"""
        try:
            # 查找最新的备份文件
            backups = sorted(self.backup_dir.glob("license_backup_*.db"), 
                           key=lambda p: p.stat().st_mtime, 
                           reverse=True)
            
            if not backups:
                print("[数据库] ⚠️ 没有找到备份文件，将创建新数据库")
                # 删除损坏的数据库
                if self.db_path.exists():
                    try:
                        self.db_path.unlink()
                    except:
                        pass
                return
            
            latest_backup = backups[0]
            print(f"[数据库] 正在从备份恢复: {latest_backup.name}")
            
            # 保存损坏的数据库（如果可以移动的话）
            if self.db_path.exists():
                try:
                    corrupted_file = self.db_path.parent / f"license_corrupted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
                    shutil.move(str(self.db_path), str(corrupted_file))
                    print(f"[数据库] 已保存损坏的数据库: {corrupted_file.name}")
                except Exception as e:
                    print(f"[数据库] 无法移动损坏的数据库: {e}")
                    # 如果无法移动，尝试直接删除
                    try:
                        self.db_path.unlink()
                        print(f"[数据库] 已删除损坏的数据库")
                    except Exception as e2:
                        print(f"[数据库] 无法删除损坏的数据库: {e2}")
                        return
            
            # 恢复备份
            shutil.copy2(latest_backup, self.db_path)
            print(f"[数据库] ✅ 恢复成功")
            
            # 验证恢复的数据库
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchone()
            conn.close()
            
            if result and result[0] == 'ok':
                print("[数据库] ✅ 恢复的数据库完整性验证通过")
            else:
                print(f"[数据库] ❌ 恢复的数据库仍然损坏: {result}")
                
        except Exception as e:
            print(f"[数据库] ❌ 恢复失败: {e}")
    
    def _cleanup_old_backups(self, keep: int = 10):
        """清理旧的备份文件，只保留最近的 N 个
        
        Args:
            keep: 保留的备份数量
        """
        try:
            backups = sorted(self.backup_dir.glob("license_backup_*.db"), 
                           key=lambda p: p.stat().st_mtime, 
                           reverse=True)
            
            # 删除多余的备份
            for old_backup in backups[keep:]:
                old_backup.unlink()
                print(f"[数据库] 已清理旧备份: {old_backup.name}")
                
        except Exception as e:
            print(f"[数据库] 清理旧备份失败: {e}")
    
    def _encrypt_field(self, value: str) -> str:
        """加密字段值
        
        Args:
            value: 原始值
            
        Returns:
            加密后的值
        """
        try:
            from crypto_utils import crypto
        except ImportError:
            from .crypto_utils import crypto
        return crypto.encrypt_database_value(value)
    
    def _decrypt_field(self, encrypted_value: str) -> str:
        """解密字段值
        
        Args:
            encrypted_value: 加密的值
            
        Returns:
            解密后的值
        """
        try:
            from crypto_utils import crypto
        except ImportError:
            from .crypto_utils import crypto
        return crypto.decrypt_database_value(encrypted_value)
    
    def save_license(self, license_data: Dict[str, Any]) -> bool:
        """保存或更新许可证信息（加密存储）
        
        Args:
            license_data: 许可证数据字典
            
        Returns:
            是否成功
        """
        try:
            print(f"[数据库] 开始保存许可证数据...")
            print(f"[数据库] 数据库路径: {self.db_path}")
            print(f"[数据库] 卡密: {license_data.get('license_key', 'N/A')}")
            print(f"[数据库] 状态: {license_data.get('status', 'N/A')}")
            
            with self._lock:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                # 加密敏感字段
                license_key_encrypted = self._encrypt_field(license_data.get('license_key', ''))
                machine_id_encrypted = self._encrypt_field(license_data.get('machine_id', ''))
                
                print(f"[数据库] 敏感字段已加密")
                
                # 检查是否已存在
                cursor.execute("SELECT id FROM license LIMIT 1")
                existing = cursor.fetchone()
                
                if existing:
                    print(f"[数据库] 更新现有记录 (ID: {existing[0]})")
                    # 更新现有记录
                    cursor.execute("""
                        UPDATE license SET
                            license_key_encrypted = ?,
                            machine_id_encrypted = ?,
                            status = ?,
                            expires_at = ?,
                            max_devices = ?,
                            activated_at = ?,
                            last_online_check = ?,
                            updated_at = ?
                        WHERE id = ?
                    """, (
                        license_key_encrypted,
                        machine_id_encrypted,
                        license_data.get('status', 'active'),
                        license_data.get('expires_at', ''),
                        license_data.get('max_devices', 1),
                        license_data.get('activated_at', ''),
                        license_data.get('last_online_check', ''),
                        datetime.now().isoformat(),
                        existing[0]
                    ))
                else:
                    print(f"[数据库] 插入新记录")
                    # 插入新记录
                    cursor.execute("""
                        INSERT INTO license (
                            license_key_encrypted, machine_id_encrypted, status, expires_at,
                            max_devices, activated_at, last_online_check
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        license_key_encrypted,
                        machine_id_encrypted,
                        license_data.get('status', 'active'),
                        license_data.get('expires_at', ''),
                        license_data.get('max_devices', 1),
                        license_data.get('activated_at', ''),
                        license_data.get('last_online_check', '')
                    ))
                
                conn.commit()
                print(f"[数据库] 数据已提交")
                conn.close()
                print(f"[数据库] ✅ 保存成功")
                
                # 保存成功后立即备份
                self._backup_database()
                
                return True
                
        except Exception as e:
            print(f"[数据库] ❌ 保存许可证失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_license(self) -> Optional[Dict[str, Any]]:
        """获取许可证信息（自动解密）
        
        Returns:
            许可证数据字典，如果不存在返回 None
        """
        try:
            with self._lock:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT license_key_encrypted, machine_id_encrypted, status, expires_at,
                           max_devices, activated_at, last_online_check,
                           last_local_check, created_at, updated_at
                    FROM license
                    LIMIT 1
                """)
                
                row = cursor.fetchone()
                conn.close()
                
                if not row:
                    return None
                
                # 解密敏感字段
                license_key = self._decrypt_field(row[0])
                machine_id = self._decrypt_field(row[1])
                
                return {
                    'license_key': license_key,
                    'machine_id': machine_id,
                    'status': row[2],
                    'expires_at': row[3],
                    'max_devices': row[4],
                    'activated_at': row[5],
                    'last_online_check': row[6],
                    'last_local_check': row[7],
                    'created_at': row[8],
                    'updated_at': row[9]
                }
                
        except Exception as e:
            print(f"读取许可证失败: {e}")
            return None
    
    def update_last_check(self, check_type: str = 'local'):
        """更新最后检查时间
        
        Args:
            check_type: 'local' 或 'online'
        """
        try:
            with self._lock:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                if check_type == 'online':
                    cursor.execute("""
                        UPDATE license SET
                            last_online_check = ?,
                            updated_at = ?
                    """, (datetime.now().isoformat(), datetime.now().isoformat()))
                else:
                    cursor.execute("""
                        UPDATE license SET
                            last_local_check = ?,
                            updated_at = ?
                    """, (datetime.now().isoformat(), datetime.now().isoformat()))
                
                conn.commit()
                conn.close()
                
        except Exception as e:
            print(f"更新检查时间失败: {e}")
    
    def delete_license(self) -> bool:
        """删除许可证信息
        
        Returns:
            是否成功
        """
        try:
            with self._lock:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                cursor.execute("DELETE FROM license")
                
                conn.commit()
                conn.close()
                return True
                
        except Exception as e:
            print(f"删除许可证失败: {e}")
            return False
    
    def is_activated(self) -> bool:
        """检查是否已激活
        
        Returns:
            是否已激活
        """
        license_data = self.get_license()
        return license_data is not None
    
    def save_history_record(self, record: Dict[str, Any]) -> bool:
        """保存历史记录
        
        Args:
            record: 历史记录数据字典，包含：
                - phone: 手机号
                - nickname: 昵称
                - user_id: 用户ID
                - balance_before: 余额前
                - points: 积分
                - vouchers: 抵扣券
                - coupons: 优惠券
                - checkin_reward: 签到奖励
                - checkin_total_times: 签到总次数
                - checkin_balance_after: 签到后余额（内部字段）
                - balance_after: 余额（最终余额）
                - duration: 耗时
                - status: 状态
                - login_method: 登录方式
                - run_date: 运行日期（YYYY-MM-DD）
                
        Returns:
            是否成功
        """
        try:
            with self._lock:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO history_records (
                        phone, nickname, user_id, balance_before, points, vouchers, coupons,
                        checkin_reward, checkin_total_times, checkin_balance_after,
                        balance_after, duration, status, login_method, run_date
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.get('phone', ''),
                    record.get('nickname', ''),
                    record.get('user_id', ''),
                    record.get('balance_before', 0),
                    record.get('points', 0),
                    record.get('vouchers', 0),
                    record.get('coupons', 0),
                    record.get('checkin_reward', 0),
                    record.get('checkin_total_times', 0),
                    record.get('checkin_balance_after', 0),
                    record.get('balance_after', 0),
                    record.get('duration', 0),
                    record.get('status', ''),
                    record.get('login_method', ''),
                    record.get('run_date', datetime.now().strftime('%Y-%m-%d'))
                ))
                
                conn.commit()
                conn.close()
                return True
                
        except Exception as e:
            print(f"保存历史记录失败: {e}")
            return False
    
    # 定义所有历史记录字段（单一真相来源）
    # 注意：字段顺序必须与数据库表结构完全一致（排除 id 和 created_at，因为它们有默认值）
    # 数据库表实际顺序（通过 PRAGMA table_info 查询）：
    #   phone, nickname, user_id, balance_before, points, vouchers, coupons,
    #   checkin_reward, checkin_total_times, checkin_balance_after, balance_after,
    #   duration, status, login_method, run_date, created_at,
    #   transfer_amount, transfer_recipient, owner
    # 
    # 重要：owner 字段是通过 ALTER TABLE 添加的，所以在最后！
    HISTORY_RECORD_FIELDS = [
        'phone', 'nickname', 'user_id', 'balance_before', 'points',
        'vouchers', 'coupons', 'checkin_reward', 'checkin_total_times',
        'checkin_balance_after', 'balance_after', 'duration', 'status',
        'login_method', 'run_date', 'transfer_amount', 'transfer_recipient', 'owner'
    ]
    
    def upsert_history_record(self, record: Dict[str, Any]) -> bool:
        """保存或更新历史记录（UPSERT 操作）- 使用字典映射避免字段错位
        
        每个账号每天只有一条记录。如果当天已有记录则更新，否则插入新记录。
        
        Args:
            record: 历史记录数据字典
                
        Returns:
            是否成功
        """
        # 验证必填字段
        required_fields = ['phone', 'run_date']
        for field in required_fields:
            if field not in record or record[field] is None:
                print(f"[数据库] ❌ 缺少必填字段: {field}")
                return False
        
        max_retries = 3
        retry_delay = 0.1  # 100ms
        
        for attempt in range(max_retries):
            try:
                with self._lock:
                    # 增加超时时间到15秒
                    conn = sqlite3.connect(str(self.db_path), timeout=15.0)
                    cursor = conn.cursor()
                    
                    # 先查询是否存在
                    cursor.execute("""
                        SELECT id FROM history_records 
                        WHERE phone = ? AND run_date = ?
                    """, (record.get('phone'), record.get('run_date')))
                    
                    existing = cursor.fetchone()
                    
                    if existing:
                        # 更新现有记录 - 智能合并：只用有效值更新，保留原有值
                        # 先查询现有记录的值（用于累计计算）
                        cursor.execute("""
                            SELECT checkin_reward, transfer_amount
                            FROM history_records 
                            WHERE phone = ? AND run_date = ?
                        """, (record.get('phone'), record.get('run_date')))
                        existing_data = cursor.fetchone()
                        existing_checkin_reward = existing_data[0] if existing_data and existing_data[0] else 0
                        existing_transfer_amount = existing_data[1] if existing_data and existing_data[1] else 0
                        
                        update_fields = []
                        values = []
                        
                        for field in self.HISTORY_RECORD_FIELDS:
                            if field in ['phone', 'run_date']:
                                continue  # 跳过主键字段
                            
                            field_value = record.get(field)
                            
                            # 智能判断：只更新有意义的值
                            should_update = False
                            actual_value = field_value  # 实际要更新的值（可能是累计值）
                            
                            if field_value is None:
                                # None值不更新
                                should_update = False
                            elif field in ['nickname', 'user_id', 'owner']:
                                # 字符串字段：只有非空时才更新
                                should_update = bool(field_value and str(field_value).strip())
                            elif field == 'checkin_reward':
                                # 签到奖励：累计（如果新值>0）
                                if isinstance(field_value, (int, float)) and field_value > 0:
                                    actual_value = existing_checkin_reward + field_value
                                    should_update = True
                            elif field == 'transfer_amount':
                                # 转账金额：累计（如果新值>0）
                                if isinstance(field_value, (int, float)) and field_value > 0:
                                    actual_value = existing_transfer_amount + field_value
                                    should_update = True
                            elif field in ['balance_before', 'balance_after', 'points', 'vouchers', 'coupons',
                                         'checkin_total_times', 'checkin_balance_after']:
                                # 其他数值字段：只有大于0时才更新（0可能是无效数据）
                                should_update = (isinstance(field_value, (int, float)) and field_value > 0)
                            elif field == 'transfer_recipient':
                                # 转账收款人：只有非空且不是"失败"时才更新
                                should_update = (field_value and str(field_value).strip() and 
                                               str(field_value).strip() not in ['失败', ''])
                            elif field == 'status':
                                # 状态字段：只有成功状态才更新（不用失败覆盖成功）
                                should_update = (field_value in ['成功', 'success', '完成'])
                            elif field in ['duration', 'timestamp']:
                                # 时间字段：总是更新（记录最新的执行时间）
                                should_update = True
                            elif field == 'error_type':
                                # 错误类型：只有非空时才更新
                                should_update = bool(field_value)
                            else:
                                # 其他字段：非None就更新
                                should_update = True
                            
                            if should_update:
                                update_fields.append(field)
                                # 对浮点数字段进行精度控制（保留2位小数）
                                float_fields = ['balance_before', 'balance_after', 'vouchers', 
                                               'checkin_reward', 'checkin_balance_after', 
                                               'transfer_amount', 'duration']
                                if field in float_fields and actual_value is not None and isinstance(actual_value, (int, float)):
                                    actual_value = round(float(actual_value), 2)
                                values.append(actual_value)
                        
                        if update_fields:
                            set_clause = ', '.join([f"{field} = ?" for field in update_fields])
                            values.extend([record.get('phone'), record.get('run_date')])
                            
                            sql = f"""
                                UPDATE history_records SET {set_clause}
                                WHERE phone = ? AND run_date = ?
                            """
                            cursor.execute(sql, values)
                            print(f"[数据库] 更新记录: {record.get('phone')} - {record.get('run_date')} (更新了 {len(update_fields)} 个字段: {', '.join(update_fields)})")
                        else:
                            print(f"[数据库] 跳过更新: {record.get('phone')} - {record.get('run_date')} (所有字段都是None)")
                    else:
                        # 插入新记录 - 使用字典映射，并对浮点数进行精度控制
                        fields_str = ', '.join(self.HISTORY_RECORD_FIELDS)
                        placeholders = ', '.join(['?'] * len(self.HISTORY_RECORD_FIELDS))
                        
                        # 定义需要精度控制的浮点数字段
                        float_fields = ['balance_before', 'balance_after', 'vouchers', 
                                       'checkin_reward', 'checkin_balance_after', 
                                       'transfer_amount', 'duration']
                        
                        # 处理值：对浮点数字段进行精度控制
                        values = []
                        for field in self.HISTORY_RECORD_FIELDS:
                            value = record.get(field)
                            # 对浮点数字段进行精度控制（保留2位小数）
                            if field in float_fields and value is not None and isinstance(value, (int, float)):
                                value = round(float(value), 2)
                            values.append(value)
                        
                        sql = f"""
                            INSERT INTO history_records ({fields_str})
                            VALUES ({placeholders})
                        """
                        cursor.execute(sql, values)
                        print(f"[数据库] 插入记录: {record.get('phone')} - {record.get('run_date')}")
                    
                    conn.commit()
                    
                    # 保存后验证（读取刚保存的记录）
                    cursor.execute("""
                        SELECT status, duration, checkin_reward FROM history_records
                        WHERE phone = ? AND run_date = ?
                    """, (record.get('phone'), record.get('run_date')))
                    
                    saved = cursor.fetchone()
                    if saved:
                        saved_status, saved_duration, saved_checkin_reward = saved
                        # 验证关键字段
                        if record.get('status') and saved_status != record.get('status'):
                            print(f"[数据库] ⚠️ 警告: status不匹配 (期望: {record.get('status')}, 实际: {saved_status})")
                        if record.get('duration') and abs(saved_duration - record.get('duration')) > 0.01:
                            print(f"[数据库] ⚠️ 警告: duration不匹配 (期望: {record.get('duration')}, 实际: {saved_duration})")
                    
                    conn.close()
                    return True
                    
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    # 数据库锁定，等待后重试
                    print(f"[数据库] 数据库锁定，{retry_delay}秒后重试 (尝试 {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 指数退避
                    continue
                else:
                    print(f"[数据库] UPSERT 失败: {e}")
                    import traceback
                    traceback.print_exc()
                    return False
            except Exception as e:
                print(f"[数据库] UPSERT 失败: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        return False
    
    def get_history_records(self, phone: str = None, start_date: str = None, 
                           end_date: str = None, limit: int = 100) -> list:
        """查询历史记录（只返回已完成的记录）
        
        Args:
            phone: 手机号（可选，用于筛选）
            start_date: 开始日期（可选，格式：YYYY-MM-DD）
            end_date: 结束日期（可选，格式：YYYY-MM-DD）
            limit: 返回记录数量限制
            
        Returns:
            历史记录列表（只包含"成功"和"失败"状态）
        """
        try:
            with self._lock:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                # 构建查询条件
                conditions = []
                params = []
                
                # 只查询已完成的记录（成功或失败，支持带emoji的状态）
                conditions.append("(status LIKE '%成功%' OR status LIKE '%失败%')")
                
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
                
                query = f"""
                    SELECT phone, nickname, user_id, balance_before, points, vouchers, coupons,
                           checkin_reward, checkin_total_times, checkin_balance_after,
                           balance_after, transfer_amount, transfer_recipient, duration, status, login_method, run_date, created_at, owner
                    FROM history_records
                    WHERE {where_clause}
                    ORDER BY created_at DESC
                    LIMIT ?
                """
                
                params.append(limit)
                cursor.execute(query, params)
                
                rows = cursor.fetchall()
                conn.close()
                
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
                
                return records
                
        except Exception as e:
            print(f"查询历史记录失败: {e}")
            return []
    
    def get_latest_record_by_phone(self, phone: str) -> Optional[Dict[str, Any]]:
        """获取指定手机号的最新记录
        
        Args:
            phone: 手机号
            
        Returns:
            最新的历史记录，如果不存在返回 None
        """
        records = self.get_history_records(phone=phone, limit=1)
        return records[0] if records else None
    
    def delete_old_records(self, days: int = 365) -> int:
        """删除指定天数之前的历史记录
        
        Args:
            days: 保留最近多少天的记录（默认365天，即一年）
            
        Returns:
            删除的记录数量
        """
        try:
            with self._lock:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
                
                cursor.execute("""
                    DELETE FROM history_records
                    WHERE run_date < ?
                """, (cutoff_date,))
                
                deleted_count = cursor.rowcount
                conn.commit()
                conn.close()
                
                return deleted_count
                
        except Exception as e:
            print(f"删除旧记录失败: {e}")
            return 0
    
    def get_database_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息
        
        Returns:
            统计信息字典，包含：
            - total_records: 总记录数
            - unique_phones: 不同手机号数量
            - date_range: 日期范围
            - db_size_mb: 数据库文件大小（MB）
        """
        try:
            with self._lock:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                # 总记录数
                cursor.execute("SELECT COUNT(*) FROM history_records")
                total_records = cursor.fetchone()[0]
                
                # 不同手机号数量
                cursor.execute("SELECT COUNT(DISTINCT phone) FROM history_records")
                unique_phones = cursor.fetchone()[0]
                
                # 日期范围
                cursor.execute("SELECT MIN(run_date), MAX(run_date) FROM history_records")
                date_range = cursor.fetchone()
                
                conn.close()
                
                # 数据库文件大小
                db_size_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0
                db_size_mb = db_size_bytes / (1024 * 1024)
                
                return {
                    'total_records': total_records,
                    'unique_phones': unique_phones,
                    'date_range': date_range,
                    'db_size_mb': round(db_size_mb, 2)
                }
                
        except Exception as e:
            print(f"获取数据库统计失败: {e}")
            return {
                'total_records': 0,
                'unique_phones': 0,
                'date_range': (None, None),
                'db_size_mb': 0
            }
    
    def get_account_summary(self, phone: str) -> Optional[Dict[str, Any]]:
        """获取指定账号的汇总统计
        
        Args:
            phone: 手机号
            
        Returns:
            账号汇总信息，包含：
            - phone: 手机号
            - nickname: 昵称（最新）
            - user_id: 用户ID（最新）
            - total_records: 总记录数
            - success_count: 成功次数
            - fail_count: 失败次数
            - success_rate: 成功率（%）
            - total_checkin_reward: 总签到奖励
            - avg_checkin_reward: 平均签到奖励
            - latest_balance: 最新余额
            - first_date: 首次记录日期
            - latest_date: 最新记录日期
        """
        try:
            with self._lock:
                conn = sqlite3.connect(str(self.db_path))
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
                    conn.close()
                    return None
                
                # 获取最新记录的昵称、用户ID和余额
                cursor.execute("""
                    SELECT nickname, user_id, balance_after
                    FROM history_records
                    WHERE phone = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                """, (phone,))
                
                latest = cursor.fetchone()
                conn.close()
                
                total = row[0]
                success = row[1] or 0
                fail = row[2] or 0
                success_rate = (success / total * 100) if total > 0 else 0
                
                return {
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
                
        except Exception as e:
            print(f"获取账号汇总失败: {e}")
            return None
    
    def get_date_summary(self, date: str) -> Dict[str, Any]:
        """获取指定日期的汇总统计
        
        Args:
            date: 日期（格式：YYYY-MM-DD）
            
        Returns:
            日期汇总信息，包含：
            - date: 日期
            - total_accounts: 总账号数
            - success_count: 成功数
            - fail_count: 失败数
            - success_rate: 成功率（%）
            - total_checkin_reward: 总签到奖励
            - avg_checkin_reward: 平均签到奖励
            - total_duration: 总耗时（秒）
            - avg_duration: 平均耗时（秒）
        """
        try:
            with self._lock:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN status = '成功' THEN 1 ELSE 0 END) as success,
                        SUM(CASE WHEN status = '失败' THEN 1 ELSE 0 END) as fail,
                        SUM(checkin_reward) as total_reward,
                        AVG(checkin_reward) as avg_reward,
                        SUM(duration) as total_duration,
                        AVG(duration) as avg_duration
                    FROM history_records
                    WHERE run_date = ?
                """, (date,))
                
                row = cursor.fetchone()
                conn.close()
                
                if not row or row[0] == 0:
                    return {
                        'date': date,
                        'total_accounts': 0,
                        'success_count': 0,
                        'fail_count': 0,
                        'success_rate': 0,
                        'total_checkin_reward': 0,
                        'avg_checkin_reward': 0,
                        'total_duration': 0,
                        'avg_duration': 0
                    }
                
                total = row[0]
                success = row[1] or 0
                fail = row[2] or 0
                success_rate = (success / total * 100) if total > 0 else 0
                
                return {
                    'date': date,
                    'total_accounts': total,
                    'success_count': success,
                    'fail_count': fail,
                    'success_rate': round(success_rate, 2),
                    'total_checkin_reward': round(row[3] or 0, 2),
                    'avg_checkin_reward': round(row[4] or 0, 2),
                    'total_duration': round(row[5] or 0, 1),
                    'avg_duration': round(row[6] or 0, 1)
                }
                
        except Exception as e:
            print(f"获取日期汇总失败: {e}")
            return {}
    
    def get_month_summary(self, year: int, month: int) -> Dict[str, Any]:
        """获取指定月份的汇总统计
        
        Args:
            year: 年份
            month: 月份（1-12）
            
        Returns:
            月份汇总信息
        """
        try:
            with self._lock:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                # 构建日期范围
                start_date = f"{year}-{month:02d}-01"
                if month == 12:
                    end_date = f"{year + 1}-01-01"
                else:
                    end_date = f"{year}-{month + 1:02d}-01"
                
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(DISTINCT phone) as unique_phones,
                        SUM(CASE WHEN status = '成功' THEN 1 ELSE 0 END) as success,
                        SUM(CASE WHEN status = '失败' THEN 1 ELSE 0 END) as fail,
                        SUM(checkin_reward) as total_reward,
                        AVG(checkin_reward) as avg_reward,
                        SUM(duration) as total_duration
                    FROM history_records
                    WHERE run_date >= ? AND run_date < ?
                """, (start_date, end_date))
                
                row = cursor.fetchone()
                conn.close()
                
                if not row or row[0] == 0:
                    return {
                        'year': year,
                        'month': month,
                        'total_records': 0,
                        'unique_phones': 0,
                        'success_count': 0,
                        'fail_count': 0,
                        'success_rate': 0,
                        'total_checkin_reward': 0,
                        'avg_checkin_reward': 0,
                        'total_duration': 0
                    }
                
                total = row[0]
                success = row[2] or 0
                fail = row[3] or 0
                success_rate = (success / total * 100) if total > 0 else 0
                
                return {
                    'year': year,
                    'month': month,
                    'total_records': total,
                    'unique_phones': row[1],
                    'success_count': success,
                    'fail_count': fail,
                    'success_rate': round(success_rate, 2),
                    'total_checkin_reward': round(row[4] or 0, 2),
                    'avg_checkin_reward': round(row[5] or 0, 2),
                    'total_duration': round(row[6] or 0, 1)
                }
                
        except Exception as e:
            print(f"获取月份汇总失败: {e}")
            return {}
    
    def get_all_accounts_summary(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """获取所有账号的汇总列表
        
        Args:
            limit: 返回账号数量限制
            
        Returns:
            账号汇总列表，按最新记录时间排序
        """
        try:
            with self._lock:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                # 先获取每个账号的基本统计
                cursor.execute("""
                    SELECT 
                        phone,
                        COUNT(*) as total_records,
                        SUM(CASE WHEN status = '成功' THEN 1 ELSE 0 END) as success_count,
                        SUM(CASE WHEN status = '失败' THEN 1 ELSE 0 END) as fail_count,
                        SUM(checkin_reward) as total_checkin_reward,
                        AVG(checkin_reward) as avg_checkin_reward,
                        MIN(run_date) as first_date,
                        MAX(run_date) as latest_date
                    FROM history_records
                    GROUP BY phone
                    ORDER BY latest_date DESC
                    LIMIT ?
                """, (limit,))
                
                stats_rows = cursor.fetchall()
                
                # 对每个账号，获取最新记录的昵称、用户ID和余额（使用和get_account_summary相同的查询）
                summaries = []
                for stats_row in stats_rows:
                    phone = stats_row[0]
                    
                    # 使用和get_account_summary相同的查询方式
                    cursor.execute("""
                        SELECT nickname, user_id, balance_after
                        FROM history_records
                        WHERE phone = ?
                        ORDER BY created_at DESC
                        LIMIT 1
                    """, (phone,))
                    
                    latest = cursor.fetchone()
                    
                    total = stats_row[1]
                    success = stats_row[2] or 0
                    fail = stats_row[3] or 0
                    success_rate = (success / total * 100) if total > 0 else 0
                    
                    summaries.append({
                        'phone': phone,
                        'nickname': latest[0] if latest else None,
                        'user_id': latest[1] if latest else None,
                        'total_records': total,
                        'success_count': success,
                        'fail_count': fail,
                        'success_rate': round(success_rate, 2),
                        'total_checkin_reward': round(stats_row[4] or 0, 2),
                        'avg_checkin_reward': round(stats_row[5] or 0, 2),
                        'latest_balance': round(latest[2], 2) if latest and latest[2] else None,
                        'first_date': stats_row[6],
                        'latest_date': stats_row[7]
                    })
                
                conn.close()
                return summaries
                
        except Exception as e:
            print(f"获取所有账号汇总失败: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def export_to_csv(self, output_path: str, phone: str = None, 
                     start_date: str = None, end_date: str = None) -> bool:
        """导出历史记录到CSV文件
        
        Args:
            output_path: 输出文件路径
            phone: 手机号（可选）
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）
            
        Returns:
            是否成功
        """
        try:
            import csv
            
            records = self.get_history_records(phone, start_date, end_date, limit=100000)
            
            if not records:
                print("没有数据可导出")
                return False
            
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
                fieldnames = [
                    '手机号', '昵称', '用户ID', '余额前', '积分', '抵扣券', '优惠券',
                    '签到奖励', '签到总次数', '余额', '耗时(秒)',
                    '状态', '登录方式', '运行日期', '创建时间'
                ]
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for record in records:
                    writer.writerow({
                        '手机号': record['phone'],
                        '昵称': record['nickname'] or 'N/A',
                        '用户ID': record['user_id'] or 'N/A',
                        '余额前': f"{record['balance_before']:.2f}" if record['balance_before'] else 'N/A',
                        '积分': record['points'] or 'N/A',
                        '抵扣券': record['vouchers'] or 'N/A',
                        '优惠券': record['coupons'] or 'N/A',
                        '签到奖励': f"{round(record['checkin_reward'], 3)}",
                        '签到总次数': record['checkin_total_times'] or 'N/A',
                        '余额': f"{record['balance_after']:.2f}" if record['balance_after'] else 'N/A',
                        '耗时(秒)': f"{record['duration']:.1f}" if record['duration'] else 'N/A',
                        '状态': record['status'],
                        '登录方式': record['login_method'] or 'N/A',
                        '运行日期': record['run_date'],
                        '创建时间': record['created_at']
                    })
            
            print(f"✅ 成功导出 {len(records)} 条记录到: {output_path}")
            return True
            
        except Exception as e:
            print(f"导出CSV失败: {e}")
            return False
    
    def get_all_history_records(self) -> List[Dict[str, Any]]:
        """获取所有历史记录（用于GUI加载，只返回已完成的记录）
        
        Returns:
            所有历史记录列表（只包含"成功"和"失败"状态），按创建时间倒序排列
            返回的字典同时包含中文键名和英文键名，以保持兼容性
        """
        try:
            with self._lock:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                # 只查询已完成的记录（成功或失败，支持带emoji的状态）
                cursor.execute("""
                    SELECT phone, nickname, user_id, balance_before, points, vouchers, coupons,
                           checkin_reward, checkin_total_times, checkin_balance_after,
                           balance_after, transfer_amount, transfer_recipient, duration, status, login_method, run_date, created_at, owner
                    FROM history_records
                    WHERE status LIKE '%成功%' OR status LIKE '%失败%'
                    ORDER BY created_at DESC
                """)
                
                rows = cursor.fetchall()
                conn.close()
                
                # 转换为字典列表，同时包含中文键名和英文键名
                records = []
                for row in rows:
                    record = {
                        # 中文键名（用于GUI显示）
                        '手机号': row[0],
                        '昵称': row[1] or '待处理',
                        '用户ID': row[2] or '待处理',
                        '余额前(元)': row[3] if row[3] is not None else '-',
                        '积分': row[4] if row[4] is not None else '-',
                        '抵扣券(张)': row[5] if row[5] is not None else '-',
                        '优惠券(张)': row[6] if row[6] is not None else '-',
                        '签到奖励(元)': row[7] if row[7] is not None else '-',
                        '签到总次数': row[8] if row[8] is not None else '-',
                        '余额(元)': row[10] if row[10] is not None else '-',
                        '转账金额(元)': row[11] if row[11] is not None else 0.0,
                        '转账收款人': row[12] or '',
                        '耗时(秒)': row[13] if row[13] is not None else '-',
                        '状态': row[14] or '待处理',
                        '登录方式': row[15] or '-',
                        '运行日期': row[16],
                        '创建时间': row[17],
                        '管理员': row[18] or '-',
                        # 英文键名（用于代码访问）
                        'phone': row[0],
                        'nickname': row[1] or '待处理',
                        'user_id': row[2] or '待处理',
                        'balance_before': row[3],
                        'points': row[4],
                        'vouchers': row[5],
                        'coupons': row[6],
                        'checkin_reward': row[7],
                        'checkin_total_times': row[8],
                        'checkin_balance_after': row[9],
                        'balance_after': row[10],
                        'transfer_amount': row[11] if row[11] is not None else 0.0,
                        'transfer_recipient': row[12] or '',
                        'duration': row[13],
                        'status': row[14] or '待处理',
                        'login_method': row[15] or '-',
                        'run_date': row[16],
                        'created_at': row[17],
                        'owner': row[18] or '-'
                    }
                    records.append(record)
                
                return records
                
        except Exception as e:
            print(f"获取所有历史记录失败: {e}")
            return []
            return []
    
    def update_account_owner(self, phone: str, owner: str) -> bool:
        """更新账号的管理员
        
        Args:
            phone: 手机号
            owner: 管理员名称
            
        Returns:
            是否成功
        """
        try:
            with self._lock:
                conn = sqlite3.connect(str(self.db_path), timeout=15.0)
                cursor = conn.cursor()
                
                # 更新该手机号的所有记录
                cursor.execute("""
                    UPDATE history_records 
                    SET owner = ?
                    WHERE phone = ?
                """, (owner, phone))
                
                affected_rows = cursor.rowcount
                conn.commit()
                conn.close()
                
                print(f"[数据库] 更新管理员: {phone} → {owner} (影响 {affected_rows} 条记录)")
                return True
                
        except Exception as e:
            print(f"[数据库] 更新管理员失败: {e}")
            return False
    
    def batch_update_account_owner(self, phones: List[str], owner: str) -> int:
        """批量更新账号的管理员
        
        Args:
            phones: 手机号列表
            owner: 管理员名称
            
        Returns:
            成功更新的账号数量
        """
        count = 0
        for phone in phones:
            if self.update_account_owner(phone, owner):
                count += 1
        return count
    
    def get_all_account_owners(self) -> Dict[str, str]:
        """获取所有账号的管理员映射
        
        Returns:
            字典 {手机号: 管理员名称}
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # 查询所有有管理员的账号（从每个账号的最新记录获取）
            cursor.execute("""
                SELECT phone, owner
                FROM history_records
                WHERE owner IS NOT NULL AND owner != ''
                AND (phone, run_date) IN (
                    SELECT phone, MAX(run_date)
                    FROM history_records
                    WHERE owner IS NOT NULL AND owner != ''
                    GROUP BY phone
                )
            """)
            
            result = {}
            for row in cursor.fetchall():
                phone = row[0]
                owner = row[1]
                result[phone] = owner
            
            return result
            
        except Exception as e:
            print(f"获取账号管理员映射失败: {e}")
            return {}

    def get_top_accounts(self, limit: int = 10, order_by: str = 'total_reward') -> List[Dict[str, Any]]:
        """获取排行榜（按指定指标排序）
        
        Args:
            limit: 返回数量
            order_by: 排序字段，可选：
                - 'total_reward': 总签到奖励
                - 'success_rate': 成功率
                - 'total_records': 总记录数
                
        Returns:
            账号排行列表
        """
        try:
            summaries = self.get_all_accounts_summary(limit=10000)
            
            if order_by == 'total_reward':
                summaries.sort(key=lambda x: x['total_checkin_reward'], reverse=True)
            elif order_by == 'success_rate':
                summaries.sort(key=lambda x: x['success_rate'], reverse=True)
            elif order_by == 'total_records':
                summaries.sort(key=lambda x: x['total_records'], reverse=True)
            
            return summaries[:limit]
            
        except Exception as e:
            print(f"获取排行榜失败: {e}")
            return []
