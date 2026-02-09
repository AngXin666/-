"""
删除数据库中的测试账号
"""
import sqlite3
from pathlib import Path

def delete_test_accounts():
    """删除测试账号"""
    db_path = Path("runtime_data") / "license.db"
    
    if not db_path.exists():
        print(f"❌ 数据库文件不存在: {db_path}")
        return
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # 先查看所有表
    print("=" * 80)
    print("数据库中的表:")
    print("=" * 80)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    for table in tables:
        print(f"  - {table[0]}")
    
    # 查找测试账号（手机号包含测试特征）
    test_patterns = [
        '13800138999%',  # 13800138999开头
        '13900139000%',  # 13900139000开头
        '13900139001%',  # 13900139001开头
        '13900139002%',  # 13900139002开头
        '13044226531%',  # 13044226531开头
        '%9999999%',     # 包含9999999
        '%0000000%',     # 包含0000000
        '%1111111%',     # 包含1111111
        '%8888888%',     # 包含8888888
    ]
    
    print("\n" + "=" * 80)
    print("查找测试账号...")
    print("=" * 80)
    
    # 查找所有可能的测试账号
    all_test_phones = set()
    for pattern in test_patterns:
        cursor.execute("SELECT DISTINCT phone FROM history_records WHERE phone LIKE ?", (pattern,))
        phones = [row[0] for row in cursor.fetchall()]
        all_test_phones.update(phones)
    
    if not all_test_phones:
        print("✓ 没有找到测试账号")
        conn.close()
        return
    
    print(f"找到 {len(all_test_phones)} 个测试账号:")
    for phone in sorted(all_test_phones):
        print(f"  - {phone}")
    
    # 删除测试账号数据
    print("\n" + "=" * 80)
    print("开始删除...")
    print("=" * 80)
    
    deleted_count = 0
    for phone in all_test_phones:
        # 删除 history_records 表中的记录
        cursor.execute("DELETE FROM history_records WHERE phone = ?", (phone,))
        deleted = cursor.rowcount
        
        if deleted > 0:
            print(f"✓ 删除账号: {phone} ({deleted} 条记录)")
            deleted_count += deleted
    
    # 提交更改
    conn.commit()
    
    print("\n" + "=" * 80)
    print(f"✓ 删除完成，共删除 {deleted_count} 条记录")
    print("=" * 80)
    
    # 验证删除结果
    print("\n验证删除结果...")
    for pattern in test_patterns:
        cursor.execute("SELECT COUNT(*) FROM history_records WHERE phone LIKE ?", (pattern,))
        count = cursor.fetchone()[0]
        if count > 0:
            print(f"⚠️ 仍有 {count} 条记录匹配模式: {pattern}")
    
    conn.close()

if __name__ == "__main__":
    delete_test_accounts()
