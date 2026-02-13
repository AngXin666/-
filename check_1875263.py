#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查 ID 1875263 的转账记录"""

import sqlite3
from datetime import datetime

# 连接数据库
conn = sqlite3.connect('runtime_data/license.db')
cursor = conn.cursor()

# 查询最近的记录
cursor.execute('''
    SELECT phone, user_id, balance_before, balance_after, 
           checkin_balance_after, transfer_amount, transfer_recipient, 
           timestamp, success, error_message
    FROM account_results 
    WHERE user_id = "1875263"
    ORDER BY timestamp DESC 
    LIMIT 10
''')

rows = cursor.fetchall()

print("=" * 80)
print(f"ID 1875263 最近 {len(rows)} 条记录:")
print("=" * 80)

for i, row in enumerate(rows, 1):
    phone, user_id, balance_before, balance_after, checkin_balance_after, \
    transfer_amount, transfer_recipient, timestamp, success, error_message = row
    
    print(f"\n记录 {i}:")
    print(f"  手机号: {phone}")
    print(f"  用户ID: {user_id}")
    print(f"  时间: {timestamp}")
    print(f"  成功: {success}")
    print(f"  余额前: {balance_before}")
    print(f"  签到后余额: {checkin_balance_after}")
    print(f"  转账金额: {transfer_amount}")
    print(f"  收款人: {transfer_recipient}")
    print(f"  余额后: {balance_after}")
    if error_message:
        print(f"  错误信息: {error_message}")
    print("-" * 80)

conn.close()
