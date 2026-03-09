#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查数据库中的余额总和

检查今天的签到记录，计算总余额是否正确
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.local_db import LocalDatabase

def main():
    """检查余额总和"""
    db = LocalDatabase()
    
    # 获取今天的所有记录
    from datetime import datetime
    today = datetime.now().strftime('%Y-%m-%d')
    
    print(f"检查日期: {today}")
    print("=" * 80)
    
    # 使用 get_history_records 方法查询今天的所有记录
    records = db.get_history_records(start_date=today, end_date=today, limit=10000)
    
    if not records:
        print(f"今天没有签到记录")
        return
    
    print(f"今天共有 {len(records)} 条记录")
    print()
    
    # 按账号分组，只取每个账号的最新记录
    account_records = {}
    for record in records:
        phone = record['phone']
        if phone not in account_records:
            account_records[phone] = record
    
    print(f"去重后共有 {len(account_records)} 个账号")
    print()
    
    # 计算总余额
    total_balance_after = 0.0
    total_checkin_reward = 0.0
    success_count = 0
    
    print("账号余额明细：")
    print("-" * 80)
    print(f"{'账号':<15} {'余额前':<12} {'余额后':<12} {'签到奖励':<12} {'状态':<10}")
    print("-" * 80)
    
    for phone, record in sorted(account_records.items()):
        balance_before = record['balance_before'] if record['balance_before'] is not None else 0.0
        balance_after = record['balance_after'] if record['balance_after'] is not None else 0.0
        checkin_reward = record['checkin_reward'] if record['checkin_reward'] is not None else 0.0
        status = record['status']
        
        print(f"{phone:<15} {balance_before:<12.2f} {balance_after:<12.2f} {checkin_reward:<12.2f} {status:<10}")
        
        if status == '成功' or '成功' in status:
            success_count += 1
            total_balance_after += balance_after
            total_checkin_reward += checkin_reward
    
    print("-" * 80)
    print(f"{'总计':<15} {'':<12} {total_balance_after:<12.2f} {total_checkin_reward:<12.2f}")
    print("=" * 80)
    print()
    print(f"成功账号数: {success_count}")
    print(f"总余额（所有成功账号的余额后）: {total_balance_after:.2f} 元")
    print(f"总签到奖励: {total_checkin_reward:.2f} 元")
    print(f"平均每账号余额: {total_balance_after / success_count:.2f} 元" if success_count > 0 else "N/A")

if __name__ == '__main__':
    main()
