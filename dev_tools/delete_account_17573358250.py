"""
删除账号17573358250的所有记录

运行方式:
    python dev_tools/delete_account_17573358250.py
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


def delete_account_records():
    """删除账号17573358250的所有记录"""
    
    phone = "17573358250"
    
    print("=" * 80)
    print(f"删除账号 {phone} 的所有记录")
    print("=" * 80)
    print()
    
    # 初始化数据库
    db = LocalDatabase()
    
    # 查询该账号的所有记录
    import sqlite3
    with db._lock:
        conn = sqlite3.connect(str(db.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM history_records WHERE phone = ? ORDER BY run_date", (phone,))
        columns = [description[0] for description in cursor.description]
        records = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
    
    if not records:
        print(f"❌ 账号 {phone} 没有记录")
        return
    
    print(f"找到 {len(records)} 条记录")
    print()
    
    # 显示记录
    for idx, record in enumerate(records, 1):
        run_date = record.get('run_date')
        balance_after = record.get('balance_after')
        checkin_reward = record.get('checkin_reward', 0.0) or 0.0
        print(f"[{idx}] 日期: {run_date}, 余额: {balance_after:.2f}, 签到奖励: {checkin_reward:.2f}")
    
    print()
    
    # 删除记录
    try:
        with db._lock:
            conn = sqlite3.connect(str(db.db_path))
            cursor = conn.cursor()
            cursor.execute("DELETE FROM history_records WHERE phone = ?", (phone,))
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
        delete_account_records()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
