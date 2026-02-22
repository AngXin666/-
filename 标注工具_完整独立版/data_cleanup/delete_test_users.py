"""删除数据库中的测试用户数据"""
import sys
sys.path.insert(0, 'src')

from local_db import LocalDatabase


def delete_test_users():
    """删除测试用户数据"""
    print("=" * 60)
    print("删除测试用户数据")
    print("=" * 60)
    
    db = LocalDatabase()
    
    # 测试账号列表
    test_phones = [
        '13111111111',
        '13222222222',
        '13333333333'
    ]
    
    print(f"\n准备删除 {len(test_phones)} 个测试账号的数据...")
    
    for phone in test_phones:
        print(f"\n处理账号: {phone}")
        
        # 检查是否有数据
        records = db.get_history_records(phone)
        if not records:
            print(f"  ⚠️ 该账号没有历史记录")
            continue
        
        print(f"  找到 {len(records)} 条历史记录")
        
        # 删除该账号的所有历史记录
        try:
            conn = db._get_connection()
            cursor = conn.cursor()
            
            # 删除历史记录（表名是 history）
            cursor.execute("DELETE FROM history WHERE 手机号 = ?", (phone,))
            deleted_count = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            print(f"  ✓ 已删除 {deleted_count} 条记录")
        except Exception as e:
            print(f"  ✗ 删除失败: {e}")
    
    print("\n" + "=" * 60)
    print("删除完成")
    print("=" * 60)
    
    # 验证删除结果
    print("\n验证删除结果...")
    remaining = 0
    for phone in test_phones:
        records = db.get_history_records(phone)
        if records:
            print(f"  ⚠️ {phone} 还有 {len(records)} 条记录")
            remaining += len(records)
        else:
            print(f"  ✓ {phone} 已完全删除")
    
    if remaining == 0:
        print("\n✅ 所有测试用户数据已删除")
    else:
        print(f"\n⚠️ 还有 {remaining} 条记录未删除")


if __name__ == '__main__':
    # 确认删除
    print("⚠️  警告：此操作将永久删除以下测试账号的所有数据：")
    print("  - 13111111111")
    print("  - 13222222222")
    print("  - 13333333333")
    print()
    
    confirm = input("确认删除？(输入 yes 继续): ")
    if confirm.lower() == 'yes':
        delete_test_users()
    else:
        print("已取消删除")
