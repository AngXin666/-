import sqlite3
from pathlib import Path

# 检查 dist 目录数据库
dist_db = Path("dist/runtime_data/license.db")
if dist_db.exists():
    print("=== dist 目录数据库 ===")
    conn = sqlite3.connect(str(dist_db))
    cursor = conn.cursor()
    
    # 查看表结构
    cursor.execute("PRAGMA table_info(history_records)")
    columns = cursor.fetchall()
    print(f"history_records 表结构:")
    for col in columns:
        print(f"  {col[1]} ({col[2]})")
    
    # 检查历史记录数量
    cursor.execute("SELECT COUNT(*) FROM history_records")
    count = cursor.fetchone()[0]
    print(f"\n历史记录总数: {count}")
    
    # 显示所有记录
    if count > 0:
        cursor.execute("SELECT phone, nickname, status, run_date FROM history_records WHERE status='成功' OR status='失败' ORDER BY created_at DESC LIMIT 10")
        print("\n最近10条已完成记录:")
        for row in cursor.fetchall():
            print(f"  {row[0]} | {row[1]} | {row[2]} | {row[3]}")
    else:
        print("\n没有历史记录")
    
    conn.close()
else:
    print("dist 目录数据库不存在")
