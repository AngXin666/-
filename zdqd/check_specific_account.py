"""
查询特定user_id的历史记录
"""
import sys
sys.path.insert(0, 'src')

from local_db import LocalDatabase

def check_account(user_id):
    """查询指定user_id的历史记录"""
    db = LocalDatabase()
    
    # 直接查询数据库
    conn = db._get_connection()
    cursor = conn.cursor()
    
    # 查询该user_id的所有记录
    cursor.execute("""
        SELECT * FROM history_records 
        WHERE user_id = ? 
        ORDER BY run_date DESC
    """, (user_id,))
    
    rows = cursor.fetchall()
    columns = [description[0] for description in cursor.description]
    
    print("=" * 80)
    print(f"用户ID {user_id} 的历史记录")
    print("=" * 80)
    print(f"找到 {len(rows)} 条记录\n")
    
    for i, row in enumerate(rows, 1):
        print(f"记录 {i}:")
        print("-" * 80)
        
        # 打印所有字段
        for col, val in zip(columns, row):
            print(f"  {col}: {val}")
        
        print()
    
    # 如果有记录，检查最新的一条
    if rows:
        latest_row = rows[0]
        latest = dict(zip(columns, latest_row))
        
        print("\n" + "=" * 80)
        print("最新记录详细分析:")
        print("=" * 80)
        
        # 检查关键字段
        phone = latest.get('phone')
        nickname = latest.get('nickname')
        user_id_field = latest.get('user_id')
        balance_before = latest.get('balance_before')
        balance_after = latest.get('balance_after')
        
        print(f"手机号: {phone}")
        print(f"昵称: {nickname}")
        print(f"用户ID: {user_id_field}")
        print(f"余额前: {balance_before}")
        print(f"余额后: {balance_after}")
        
        # 检查是否有字段错位
        print("\n字段错位检查:")
        if nickname and isinstance(nickname, str):
            # 检查昵称是否像余额（包含数字和小数点）
            if '.' in str(nickname) and any(c.isdigit() for c in str(nickname)):
                print(f"  ⚠️ 警告：昵称字段 '{nickname}' 看起来像余额数据！")
            else:
                print(f"  ✅ 昵称字段正常")
        
        if balance_before and isinstance(balance_before, str):
            # 检查余额前是否像昵称（包含中文或字母）
            if any('\u4e00' <= c <= '\u9fff' for c in str(balance_before)):
                print(f"  ⚠️ 警告：余额前字段 '{balance_before}' 看起来像昵称数据！")
            else:
                print(f"  ✅ 余额前字段正常")
    
    conn.close()

if __name__ == '__main__':
    check_account('1800829')
