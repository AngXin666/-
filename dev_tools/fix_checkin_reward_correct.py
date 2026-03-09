"""
正确修复历史记录中的签到奖励

正确逻辑：
1. 余额前 = 前一天（或更早）的余额后
2. 签到奖励 = 当天余额后 - 余额前
3. 如果没有前一天记录，余额前 = 0.00
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.local_db import LocalDatabase
from datetime import datetime, timedelta

def fix_checkin_rewards():
    """正确修复签到奖励"""
    db = LocalDatabase()
    
    print("=" * 80)
    print("正确修复历史记录中的签到奖励")
    print("=" * 80)
    print()
    
    # 获取所有历史记录，按手机号和日期排序
    all_records = db.get_all_history_records()
    
    if not all_records:
        print("没有找到历史记录")
        return
    
    print(f"共找到 {len(all_records)} 条历史记录")
    print()
    
    # 按手机号分组
    records_by_phone = {}
    for record in all_records:
        phone = record.get('phone', '')
        if phone not in records_by_phone:
            records_by_phone[phone] = []
        records_by_phone[phone].append(record)
    
    # 对每个手机号的记录按日期排序
    for phone in records_by_phone:
        records_by_phone[phone].sort(key=lambda x: x.get('run_date', ''))
    
    print(f"共 {len(records_by_phone)} 个账号")
    print()
    
    error_count = 0
    fixed_count = 0
    
    # 处理每个账号的记录
    for phone, records in records_by_phone.items():
        print(f"处理账号: {phone}")
        
        for i, record in enumerate(records):
            run_date = record.get('run_date', '')
            balance_before = record.get('balance_before', 0) or 0
            balance_after = record.get('balance_after', 0) or 0
            checkin_reward = record.get('checkin_reward', 0) or 0
            
            # 获取前一天的余额（从前面的记录中查找）
            correct_balance_before = 0.00
            if i > 0:
                # 有前面的记录，使用前一条记录的 balance_after
                prev_record = records[i - 1]
                prev_balance_after = prev_record.get('balance_after', 0) or 0
                correct_balance_before = prev_balance_after
            else:
                # 第一条记录，余额前应该是 0.00
                correct_balance_before = 0.00
            
            # 获取转账收款人信息
            transfer_recipient = record.get('transfer_recipient', '') or ''
            
            # 计算正确的签到奖励
            # 如果余额前有数字，但余额后是0或更小，说明发生了转账
            # 签到奖励 = 0（因为余额被转走了）
            if correct_balance_before > 0 and balance_after == 0:
                # 发生了转账，签到奖励设为0
                correct_reward = 0.00
            else:
                # 正常情况：签到奖励 = 余额后 - 余额前
                correct_reward = round(balance_after - correct_balance_before, 2)
                # 签到奖励不应该是负数
                if correct_reward < 0:
                    correct_reward = 0.00
                # 签到奖励不应该大于10（除非是收款账号）
                if correct_reward > 10 and not transfer_recipient:
                    correct_reward = 0.00
            
            # 检查是否需要修复
            need_fix = False
            if abs(balance_before - correct_balance_before) > 0.01:
                need_fix = True
            if abs(checkin_reward - correct_reward) > 0.01:
                need_fix = True
            
            if need_fix:
                error_count += 1
                print(f"  ❌ 错误记录: {run_date}")
                print(f"     当前余额前: {balance_before:.2f}, 正确余额前: {correct_balance_before:.2f}")
                print(f"     当前余额后: {balance_after:.2f}")
                print(f"     当前签到奖励: {checkin_reward:.2f}, 正确签到奖励: {correct_reward:.2f}")
                
                # 直接使用SQL UPDATE语句修复记录（绕过upsert的智能判断）
                try:
                    import sqlite3
                    conn = sqlite3.connect('runtime_data/license.db')
                    cursor = conn.cursor()
                    
                    cursor.execute("""
                        UPDATE history_records
                        SET balance_before = ?, checkin_reward = ?
                        WHERE phone = ? AND run_date = ?
                    """, (correct_balance_before, correct_reward, phone, run_date))
                    
                    conn.commit()
                    conn.close()
                    
                    fixed_count += 1
                    print(f"     ✓ 已修复")
                except Exception as e:
                    print(f"     ✗ 修复失败: {e}")
        
        print()
    
    print("=" * 80)
    print(f"检查完成")
    print(f"错误记录数: {error_count}")
    print(f"已修复记录数: {fixed_count}")
    print("=" * 80)

if __name__ == '__main__':
    fix_checkin_rewards()
