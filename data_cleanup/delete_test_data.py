"""删除数据库中的测试数据"""
from src.local_db import LocalDatabase

db = LocalDatabase()

# 测试账号的手机号列表
test_phones = [
    '13800138001',
    '13800138002',
    '13800138003',
    '13800138004'
]

print("\n开始删除测试数据...")
print("=" * 60)

# 删除测试数据
import sqlite3
conn = sqlite3.connect(str(db.db_path))
cursor = conn.cursor()

deleted_count = 0
for phone in test_phones:
    cursor.execute("DELETE FROM history_records WHERE phone = ?", (phone,))
    if cursor.rowcount > 0:
        print(f"✓ 已删除: {phone} ({cursor.rowcount} 条记录)")
        deleted_count += cursor.rowcount
    else:
        print(f"- 未找到: {phone}")

conn.commit()
conn.close()

print("=" * 60)
print(f"\n总共删除了 {deleted_count} 条测试记录")

# 验证删除结果
print("\n验证删除结果...")
records = db.get_all_history_records()
remaining_test = [r for r in records if r['手机号'] in test_phones]

if remaining_test:
    print(f"⚠️ 还有 {len(remaining_test)} 条测试记录未删除")
else:
    print("✓ 所有测试数据已成功删除")

print(f"\n当前数据库总记录数: {len(records)}")
