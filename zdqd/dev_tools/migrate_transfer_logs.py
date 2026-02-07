#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""将数据库中的转账记录迁移到日志文件"""

import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# 连接数据库
conn = sqlite3.connect('runtime_data/license.db')
cursor = conn.cursor()

# 查询所有转账记录
cursor.execute("""
    SELECT 
        id, sender_phone, sender_name, recipient_phone, recipient_name,
        amount, strategy, success, error_message, timestamp
    FROM transfer_history
    ORDER BY timestamp ASC
""")

records = cursor.fetchall()
conn.close()

if not records:
    print("❌ 没有找到转账记录")
    exit(1)

print(f"找到 {len(records)} 条转账记录，开始迁移到日志文件...\n")

# 按日期分组记录
records_by_date = defaultdict(list)
for record in records:
    timestamp = record[9]  # timestamp字段
    # 解析时间戳，提取日期
    try:
        dt = datetime.fromisoformat(timestamp.replace('T', ' ').split('.')[0])
        date_str = dt.strftime('%Y%m%d')
        records_by_date[date_str].append((record, dt))
    except Exception as e:
        print(f"⚠️ 解析时间戳失败: {timestamp}, 错误: {e}")
        continue

# 创建logs目录
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)

# 为每个日期创建日志文件
total_success = 0
total_failure = 0

for date_str, date_records in sorted(records_by_date.items()):
    # 创建该日期的日志记录器
    transfer_logger = logging.getLogger(f"transfer_{date_str}")
    transfer_logger.setLevel(logging.INFO)
    transfer_logger.handlers.clear()
    transfer_logger.propagate = False
    
    failure_logger = logging.getLogger(f"failure_{date_str}")
    failure_logger.setLevel(logging.ERROR)
    failure_logger.handlers.clear()
    failure_logger.propagate = False
    
    # 转账日志文件
    transfer_log_file = log_dir / f"transfer_{date_str}.log"
    transfer_handler = logging.FileHandler(transfer_log_file, encoding='utf-8', mode='a')
    transfer_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    transfer_handler.setFormatter(formatter)
    transfer_logger.addHandler(transfer_handler)
    
    # 失败日志文件
    failure_log_file = log_dir / f"transfer_failure_{date_str}.log"
    failure_handler = logging.FileHandler(failure_log_file, encoding='utf-8', mode='a')
    failure_handler.setLevel(logging.ERROR)
    failure_handler.setFormatter(formatter)
    failure_logger.addHandler(failure_handler)
    
    # 写入该日期的所有记录
    day_success = 0
    day_failure = 0
    
    for record, dt in date_records:
        id, sender_phone, sender_name, recipient_phone, recipient_name, amount, strategy, success, error_msg, timestamp = record
        
        # 创建日志记录（使用原始时间戳）
        log_record = transfer_logger.makeRecord(
            transfer_logger.name, 
            logging.INFO if success else logging.ERROR,
            "", 0, "", (), None
        )
        log_record.created = dt.timestamp()
        log_record.msecs = (dt.timestamp() % 1) * 1000
        
        if success:
            log_record.msg = (
                f"✓ 转账成功 | 发送方: {sender_name}({sender_phone}) | "
                f"接收方: {recipient_name}({recipient_phone}) | "
                f"金额: {amount:.2f}元 | 策略: {strategy}"
            )
            transfer_handler.emit(log_record)
            day_success += 1
            total_success += 1
        else:
            log_record.msg = (
                f"❌ 转账失败 | 发送方: {sender_name}({sender_phone}) | "
                f"接收方: {recipient_name}({recipient_phone}) | "
                f"金额: {amount:.2f}元 | 策略: {strategy} | 错误: {error_msg}"
            )
            log_record.levelno = logging.ERROR
            transfer_handler.emit(log_record)
            
            # 同时写入失败日志
            failure_record = failure_logger.makeRecord(
                failure_logger.name,
                logging.ERROR,
                "", 0, "", (), None
            )
            failure_record.created = dt.timestamp()
            failure_record.msecs = (dt.timestamp() % 1) * 1000
            failure_record.msg = log_record.msg
            failure_handler.emit(failure_record)
            
            day_failure += 1
            total_failure += 1
    
    # 关闭处理器
    transfer_handler.close()
    failure_handler.close()
    
    print(f"✓ {date_str}: {day_success} 成功, {day_failure} 失败 → {transfer_log_file.name}")

print("\n" + "=" * 60)
print(f"迁移完成！")
print(f"  总记录数: {len(records)}")
print(f"  成功: {total_success}")
print(f"  失败: {total_failure}")
print(f"  日志文件数: {len(records_by_date)} 天")
print("=" * 60)
print("\n日志文件位置: logs/")
print("  - transfer_YYYYMMDD.log (所有转账记录)")
print("  - transfer_failure_YYYYMMDD.log (只有失败记录)")
