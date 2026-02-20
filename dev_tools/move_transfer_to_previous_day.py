"""
将转账金额移到前一天

转账是在前一天发生的，应该记录在前一天的记录中

运行方式:
    python dev_tools/move_transfer_to_previous_day.py
"""

import sys
import os
from pathlib import Path

# 设置标准输出编码为 UTF-8
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.local_db import LocalDatabase


def move_transfer_to_previous_day():
    """将转账金额移到前一天"""
    
    print("=" * 80)
    print("将转账金额移到前一天")
    print("=" * 80)
    print()
    print("逻辑:")
    print("  如果当天余额突然下降（有转账），将转账金额移到前一天的记录")
    print("  前一天: balance_after = checkin_balance_after - transfer_amount")
    print("  当天: transfer_amount = 0")
    print()
    
    # 初始化数据库
    db = LocalDatabase()
    all_records = db.get_all_history_records()
    
    print(f"总记录数: {len(all_records)}")
    print()
    
    # 按账号分组
    records_by_phone = {}
    for record in all_records:
        phone = record.get('phone')
        if phone:
            if phone not in records_by_phone:
                records_by_phone[phone] = []
            records_by_phone[phone].append(record)
    
    # 对每个账号的记录按日期排序
    for phone in records_by_phone:
        records_by_phone[phone].sort(key=lambda r: r.get('run_date', ''))
    
    print(f"共 {len(records_by_phone)} 个账号")
    print()
    
    # 开始修复
    print("=" * 80)
    print("开始修复...")
    print("=" * 80)
    print()
    
    updated_count = 0
    error_count = 0
    
    for phone, records in records_by_phone.items():
        for idx in range(len(records)):
            current_record = records[idx]
            current_transfer = current_record.get('transfer_amount', 0.0) or 0.0
            
            # 如果当天有转账金额
            if current_transfer > 0:
                current_id = current_record.get('id')
                current_date = current_record.get('run_date')
                
                # 找到前一天的记录
                if idx > 0:
                    previous_record = records[idx - 1]
                    previous_id = previous_record.get('id')
                    previous_date = previous_record.get('run_date')
                    previous_transfer = previous_record.get('transfer_amount', 0.0) or 0.0
                    previous_checkin_balance = previous_record.get('checkin_balance_after')
                    
                    if previous_checkin_balance is not None:
                        try:
                            conn = db._get_connection()
                            cursor = conn.cursor()
                            
                            # 更新前一天的记录：添加转账金额，更新 balance_after
                            new_previous_transfer = previous_transfer + current_transfer
                            new_previous_balance_after = previous_checkin_balance - new_previous_transfer
                            
                            cursor.execute('''
                                UPDATE history_records 
                                SET transfer_amount = ?, balance_after = ?
                                WHERE id = ?
                            ''', (new_previous_transfer, new_previous_balance_after, previous_id))
                            
                            # 更新当天的记录：清除转账金额
                            cursor.execute('''
                                UPDATE history_records 
                                SET transfer_amount = 0.0
                                WHERE id = ?
                            ''', (current_id,))
                            
                            conn.commit()
                            conn.close()
                            
                            updated_count += 1
                            
                            # 显示前20条更新
                            if updated_count <= 20:
                                print(f"[{phone}]")
                                print(f"  前一天 [{previous_date}]:")
                                print(f"    签到后余额: {previous_checkin_balance:.2f}")
                                print(f"    转账金额: {previous_transfer:.2f} → {new_previous_transfer:.2f}")
                                print(f"    最终余额: → {new_previous_balance_after:.2f}")
                                print(f"  当天 [{current_date}]:")
                                print(f"    转账金额: {current_transfer:.2f} → 0.00")
                                print()
                        
                        except Exception as e:
                            print(f"❌ [{phone}] [{current_date}] 更新失败: {e}")
                            error_count += 1
    
    # 输出统计信息
    print()
    print("=" * 80)
    print("修复完成")
    print("=" * 80)
    print(f"已更新: {updated_count} 条")
    print(f"错误: {error_count} 条")
    print()


if __name__ == "__main__":
    try:
        move_transfer_to_previous_day()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
