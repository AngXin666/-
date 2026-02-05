#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查转账记录"""

import sqlite3
from datetime import datetime, timedelta

# 连接数据库
conn = sqlite3.connect('runtime_data/license.db')
cursor = conn.cursor()

# 检查是否有transfer_history表
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='transfer_history'")
table_exists = cursor.fetchone()

if not table_exists:
    print("❌ transfer_history 表不存在")
    conn.close()
    exit(1)

print("✓ transfer_history 表存在\n")

# 查询最近的转账记录（最近7天）
seven_days_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')

cursor.execute("""
    SELECT 
        id, sender_phone, sender_name, recipient_phone, recipient_name,
        amount, strategy, success, error_message, timestamp
    FROM transfer_history
    WHERE timestamp >= ?
    ORDER BY timestamp DESC
    LIMIT 50
""", (seven_days_ago,))

records = cursor.fetchall()

if not records:
    print("⚠️ 最近7天没有转账记录")
else:
    print(f"📊 最近7天的转账记录（共 {len(records)} 条）：\n")
    print("=" * 120)
    
    for record in records:
        id, sender_phone, sender_name, recipient_phone, recipient_name, amount, strategy, success, error_msg, timestamp = record
        status = "✓ 成功" if success else "❌ 失败"
        
        print(f"ID: {id}")
        print(f"时间: {timestamp}")
        print(f"发送方: {sender_name} ({sender_phone})")
        print(f"接收方: {recipient_name} ({recipient_phone})")
        print(f"金额: {amount:.2f} 元")
        print(f"策略: {strategy}")
        print(f"状态: {status}")
        if not success and error_msg:
            print(f"错误: {error_msg}")
        print("-" * 120)

# 统计信息
cursor.execute("""
    SELECT 
        COUNT(*) as total,
        SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count,
        SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as fail_count,
        SUM(CASE WHEN success = 1 THEN amount ELSE 0 END) as total_amount
    FROM transfer_history
    WHERE timestamp >= ?
""", (seven_days_ago,))

stats = cursor.fetchone()
total, success_count, fail_count, total_amount = stats

print("\n📈 统计信息（最近7天）：")
print(f"  总记录数: {total}")
print(f"  成功: {success_count}")
print(f"  失败: {fail_count}")
print(f"  成功率: {(success_count/total*100 if total > 0 else 0):.1f}%")
print(f"  总金额: {total_amount:.2f} 元")

conn.close()
