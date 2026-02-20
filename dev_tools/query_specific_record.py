"""
查询特定记录的当前值
"""

import sys
import os
from pathlib import Path
import sqlite3

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def query_record():
    """查询特定记录"""
    
    phone = "15355570094"
    run_date = "2026-02-12"
    
    print("=" * 80)
    print(f"查询账号 {phone} 在 {run_date} 的记录")
    print("=" * 80)
    print()
    
    # 直接连接数据库
    db_path = project_root / "runtime_data" / "license.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # 查询记录
    cursor.execute("""
        SELECT id, phone, run_date, balance_before, balance_after, checkin_reward, transfer_amount
        FROM history_records
        WHERE phone = ? AND run_date = ?
    """, (phone, run_date))
    
    record = cursor.fetchone()
    
    if record:
        print(f"记录ID: {record[0]}")
        print(f"账号: {record[1]}")
        print(f"日期: {record[2]}")
        print(f"签到前余额: {record[3]}")
        print(f"签到后余额: {record[4]}")
        print(f"签到奖励: {record[5]}")
        print(f"转账金额: {record[6]}")
    else:
        print("未找到记录")
    
    conn.close()


if __name__ == "__main__":
    try:
        query_record()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
