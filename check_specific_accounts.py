"""
检查特定账号的数据
"""

from src.local_db import LocalDatabase

def check_specific_accounts():
    """检查特定账号"""
    db = LocalDatabase()
    
    # 检查截图中显示有问题的账号
    problem_phones = ['13307857120', '13322736481']
    
    for phone in problem_phones:
        print(f"\n{'='*60}")
        print(f"检查账号: {phone}")
        print(f"{'='*60}")
        
        # 使用get_all_accounts_summary
        summaries = db.get_all_accounts_summary(limit=10000)
        summary = next((s for s in summaries if s['phone'] == phone), None)
        
        if summary:
            print(f"\n[get_all_accounts_summary] 结果:")
            print(f"  手机号: {summary.get('phone')}")
            print(f"  昵称: '{summary.get('nickname')}'")
            print(f"  用户ID: '{summary.get('user_id')}'")
        else:
            print(f"\n[get_all_accounts_summary] 未找到该账号")
        
        # 使用get_account_summary
        single = db.get_account_summary(phone)
        if single:
            print(f"\n[get_account_summary] 结果:")
            print(f"  手机号: {single.get('phone')}")
            print(f"  昵称: '{single.get('nickname')}'")
            print(f"  用户ID: '{single.get('user_id')}'")
        else:
            print(f"\n[get_account_summary] 未找到该账号")
        
        # 直接查询数据库
        import sqlite3
        conn = sqlite3.connect('runtime_data/license.db')
        cursor = conn.cursor()
        cursor.execute("""
            SELECT nickname, user_id, run_date, created_at
            FROM history_records
            WHERE phone = ?
            ORDER BY created_at DESC
            LIMIT 3
        """, (phone,))
        rows = cursor.fetchall()
        conn.close()
        
        print(f"\n[直接查询数据库] 最近3条记录:")
        for i, row in enumerate(rows, 1):
            print(f"  记录{i}: 昵称='{row[0]}', 用户ID='{row[1]}', 日期={row[2]}, 创建时间={row[3]}")


if __name__ == '__main__':
    check_specific_accounts()
