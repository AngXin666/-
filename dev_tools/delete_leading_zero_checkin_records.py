"""
删除每个账号前面连续签到次数为0的记录

只删除账号开始签到前的记录，保留开始签到后的所有记录（包括签到次数为0的）

运行方式:
    python dev_tools/delete_leading_zero_checkin_records.py
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


def delete_leading_zero_checkin_records():
    """删除每个账号前面连续签到次数为0的记录"""
    
    print("=" * 80)
    print("删除每个账号前面连续签到次数为0的记录")
    print("=" * 80)
    print()
    
    # 初始化数据库
    db = LocalDatabase()
    all_records = db.get_all_history_records()
    
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
    
    print(f"总记录数: {len(all_records)}")
    print(f"共 {len(records_by_phone)} 个账号")
    print()
    
    # 找出每个账号前面连续为0的记录
    records_to_delete = []
    
    for phone, records in records_by_phone.items():
        # 找到第一个签到次数不为0的记录
        first_nonzero_idx = None
        for idx, record in enumerate(records):
            checkin_times = record.get('checkin_total_times') or 0
            if checkin_times > 0:
                first_nonzero_idx = idx
                break
        
        # 如果找到了第一个非0记录，删除它之前的所有记录
        if first_nonzero_idx is not None and first_nonzero_idx > 0:
            leading_zero_records = records[:first_nonzero_idx]
            records_to_delete.extend(leading_zero_records)
            
            print(f"账号 {phone}: 删除前 {len(leading_zero_records)} 条记录（签到次数为0）")
            for record in leading_zero_records:
                run_date = record.get('run_date')
                checkin_times = record.get('checkin_total_times') or 0
                print(f"  - 日期: {run_date}, 签到次数: {checkin_times}")
        elif first_nonzero_idx is None:
            # 所有记录的签到次数都是0，删除所有记录
            records_to_delete.extend(records)
            print(f"账号 {phone}: 删除所有 {len(records)} 条记录（签到次数全为0）")
    
    print()
    print("=" * 80)
    print(f"共需删除 {len(records_to_delete)} 条记录")
    print("=" * 80)
    print()
    
    if not records_to_delete:
        print("✓ 没有需要删除的记录")
        return
    
    # 删除这些记录
    try:
        import sqlite3
        record_ids = [record.get('id') for record in records_to_delete]
        
        with db._lock:
            conn = sqlite3.connect(str(db.db_path))
            cursor = conn.cursor()
            
            # 批量删除
            placeholders = ','.join(['?'] * len(record_ids))
            cursor.execute(f"""
                DELETE FROM history_records 
                WHERE id IN ({placeholders})
            """, record_ids)
            
            conn.commit()
            deleted_count = cursor.rowcount
            conn.close()
        
        print(f"✓ 成功删除 {deleted_count} 条记录")
        
    except Exception as e:
        print(f"❌ 删除失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        delete_leading_zero_checkin_records()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
