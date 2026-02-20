"""
检查2月4日的记录

运行方式:
    python dev_tools/check_feb4_records.py
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


def check_feb4_records():
    """检查2月4日的记录"""
    
    print("=" * 80)
    print("检查2月4日的记录")
    print("=" * 80)
    print()
    
    # 初始化数据库
    db = LocalDatabase()
    
    # 查询2月4日的所有记录
    import sqlite3
    with db._lock:
        conn = sqlite3.connect(str(db.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT phone, run_date, status, checkin_total_times, 
                   balance_before, checkin_reward, checkin_balance_after, 
                   balance_after, transfer_amount
            FROM history_records 
            WHERE run_date = '2026-02-04'
            ORDER BY phone
        """)
        
        columns = [description[0] for description in cursor.description]
        records = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
    
    if not records:
        print("❌ 没有找到2月4日的记录")
        return
    
    print(f"找到 {len(records)} 条2月4日的记录")
    print()
    
    # 统计状态
    status_count = {}
    for record in records:
        status = record.get('status', '未知')
        status_count[status] = status_count.get(status, 0) + 1
    
    print("状态统计:")
    for status, count in status_count.items():
        print(f"  {status}: {count} 条")
    print()
    
    # 显示所有记录
    print("=" * 80)
    print("详细记录:")
    print("=" * 80)
    print()
    
    for idx, record in enumerate(records, 1):
        phone = record.get('phone')
        status = record.get('status')
        checkin_times = record.get('checkin_total_times') or 0
        balance_before = record.get('balance_before')
        checkin_reward = record.get('checkin_reward', 0.0) or 0.0
        checkin_balance_after = record.get('checkin_balance_after')
        balance_after = record.get('balance_after')
        transfer_amount = record.get('transfer_amount', 0.0) or 0.0
        
        balance_before_str = f"{balance_before:.2f}" if balance_before is not None else 'None'
        checkin_balance_after_str = f"{checkin_balance_after:.2f}" if checkin_balance_after is not None else 'None'
        balance_after_str = f"{balance_after:.2f}" if balance_after is not None else 'None'
        
        print(f"[{idx}] 账号: {phone}")
        print(f"    状态: {status}")
        print(f"    签到次数: {checkin_times}")
        print(f"    余额前: {balance_before_str}")
        print(f"    签到奖励: {checkin_reward:.2f}")
        print(f"    签到后余额: {checkin_balance_after_str}")
        print(f"    最终余额: {balance_after_str}")
        print(f"    转账金额: {transfer_amount:.2f}")
        print()


if __name__ == "__main__":
    try:
        check_feb4_records()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
