"""
检查特定账号的签到奖励历史
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.local_db import LocalDatabase

def check_account_reward_history(phone):
    """检查特定账号的签到奖励历史"""
    db = LocalDatabase()
    
    print("=" * 80)
    print(f"检查账号 {phone} 的签到奖励历史")
    print("=" * 80)
    
    conn = db._get_connection()
    cursor = conn.cursor()
    
    # 查询该账号的所有记录
    cursor.execute("""
        SELECT run_date, checkin_reward, status, created_at
        FROM history_records
        WHERE phone = ?
        ORDER BY run_date DESC, created_at DESC
        LIMIT 30
    """, (phone,))
    
    records = cursor.fetchall()
    
    if records:
        print(f"\n找到 {len(records)} 条记录：")
        print("-" * 80)
        print(f"{'日期':<12} {'签到奖励':<12} {'状态':<10} {'创建时间':<20}")
        print("-" * 80)
        for run_date, reward, status, created_at in records:
            print(f"{run_date:<12} {reward:<12.2f} {status:<10} {created_at:<20}")
    else:
        print(f"\n未找到账号 {phone} 的记录")
    
    conn.close()
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    # 检查异常账号
    check_account_reward_history("17573358250")
