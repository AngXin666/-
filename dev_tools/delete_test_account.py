"""
删除测试账号的所有记录

对比账号文件,删除数据库中不存在于账号文件的测试账号记录

运行方式:
    python dev_tools/delete_test_account.py
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
from src.encrypted_accounts_file import EncryptedAccountsFile


def load_valid_accounts():
    """从账号文件加载有效账号列表"""
    try:
        accounts_file = EncryptedAccountsFile("data/accounts.txt")
        accounts = accounts_file.read_accounts()
        valid_phones = set()
        
        for phone, password in accounts:
            if phone:
                valid_phones.add(phone)
        
        return valid_phones
    except Exception as e:
        print(f"⚠️ 读取账号文件失败: {e}")
        return set()


def delete_test_accounts():
    """删除测试账号的所有记录"""
    
    print("=" * 80)
    print("删除测试账号记录")
    print("=" * 80)
    print()
    
    # 加载有效账号
    print("1. 读取账号文件...")
    valid_phones = load_valid_accounts()
    
    if not valid_phones:
        print("❌ 无法读取账号文件,终止操作")
        return
    
    print(f"   ✓ 账号文件中有 {len(valid_phones)} 个账号")
    print()
    
    # 初始化数据库
    print("2. 检查数据库记录...")
    db = LocalDatabase()
    all_records = db.get_all_history_records()
    
    # 找出数据库中存在但账号文件中不存在的账号
    db_phones = set(r.get('phone') for r in all_records if r.get('phone'))
    test_phones = db_phones - valid_phones
    
    if not test_phones:
        print("   ✓ 没有发现测试账号")
        return
    
    print(f"   ✓ 发现 {len(test_phones)} 个测试账号:")
    for phone in sorted(test_phones):
        print(f"      - {phone}")
    print()
    
    # 统计要删除的记录
    test_records = [r for r in all_records if r.get('phone') in test_phones]
    print(f"3. 找到 {len(test_records)} 条测试记录:")
    
    records_by_phone = {}
    for record in test_records:
        phone = record.get('phone')
        if phone not in records_by_phone:
            records_by_phone[phone] = []
        records_by_phone[phone].append(record)
    
    for phone in sorted(records_by_phone.keys()):
        records = records_by_phone[phone]
        print(f"   - {phone}: {len(records)} 条记录")
    print()
    
    # 删除记录
    print("4. 开始删除...")
    deleted_count = 0
    
    conn = db._get_connection()
    cursor = conn.cursor()
    
    for record in test_records:
        record_id = record.get('id')
        phone = record.get('phone')
        run_date = record.get('run_date')
        
        try:
            cursor.execute("DELETE FROM history_records WHERE id = ?", (record_id,))
            conn.commit()
            deleted_count += 1
            print(f"   ✓ 删除 {phone} - {run_date} (ID: {record_id})")
        except Exception as e:
            print(f"   ✗ 删除 {phone} - {run_date} (ID: {record_id}) 失败: {e}")
    
    conn.close()
    
    print()
    print("=" * 80)
    print(f"删除完成: 共删除 {deleted_count} 条记录")
    print("=" * 80)


if __name__ == "__main__":
    try:
        delete_test_accounts()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
