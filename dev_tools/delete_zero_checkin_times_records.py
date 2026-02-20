"""
删除所有签到次数为0的记录

这些记录是账号在开始签到前的记录，不应该被保留

运行方式:
    python dev_tools/delete_zero_checkin_times_records.py
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


def delete_zero_checkin_times_records():
    """删除所有签到次数为0的记录"""
    
    print("=" * 80)
    print("删除所有签到次数为0的记录")
    print("=" * 80)
    print()
    
    # 初始化数据库
    db = LocalDatabase()
    
    # 查询所有签到次数为0的记录
    import sqlite3
    with db._lock:
        conn = sqlite3.connect(str(db.db_path))
        cursor = conn.cursor()
        
        # 查询签到次数为0或NULL的记录
        cursor.execute("""
            SELECT id, phone, run_date, checkin_total_times, balance_after
            FROM history_records 
            WHERE checkin_total_times IS NULL OR checkin_total_times = 0
            ORDER BY phone, run_date
        """)
        
        columns = [description[0] for description in cursor.description]
        records = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
    
    if not records:
        print("✓ 没有找到签到次数为0的记录")
        return
    
    print(f"找到 {len(records)} 条签到次数为0的记录")
    print()
    
    # 按账号分组显示
    records_by_phone = {}
    for record in records:
        phone = record.get('phone')
        if phone not in records_by_phone:
            records_by_phone[phone] = []
        records_by_phone[phone].append(record)
    
    print(f"涉及 {len(records_by_phone)} 个账号")
    print()
    
    # 显示前10个账号的记录
    for idx, (phone, phone_records) in enumerate(list(records_by_phone.items())[:10], 1):
        print(f"[{idx}] 账号 {phone}: {len(phone_records)} 条记录")
        for record in phone_records[:3]:
            run_date = record.get('run_date')
            checkin_times = record.get('checkin_total_times') or 0
            balance = record.get('balance_after')
            balance_str = f"{balance:.2f}" if balance is not None else "None"
            print(f"    日期: {run_date}, 签到次数: {checkin_times}, 余额: {balance_str}")
        if len(phone_records) > 3:
            print(f"    ... 还有 {len(phone_records) - 3} 条记录")
    
    if len(records_by_phone) > 10:
        print(f"... 还有 {len(records_by_phone) - 10} 个账号")
    
    print()
    print("=" * 80)
    
    # 删除这些记录
    try:
        with db._lock:
            conn = sqlite3.connect(str(db.db_path))
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM history_records 
                WHERE checkin_total_times IS NULL OR checkin_total_times = 0
            """)
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
        delete_zero_checkin_times_records()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
