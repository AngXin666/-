"""
检查签到次数和奖励的关系
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.local_db import LocalDatabase

def check_checkin_times(phone):
    """检查签到次数和奖励的关系"""
    db = LocalDatabase()
    
    print("=" * 80)
    print(f"检查账号 {phone} 的签到次数和奖励")
    print("=" * 80)
    
    conn = db._get_connection()
    cursor = conn.cursor()
    
    # 查询该账号的记录
    cursor.execute("""
        SELECT run_date, checkin_reward, checkin_total_times
        FROM history_records
        WHERE phone = ?
        ORDER BY run_date DESC
        LIMIT 15
    """, (phone,))
    
    records = cursor.fetchall()
    
    if records:
        print(f"\n{'日期':<12} {'签到奖励':<12} {'签到次数':<10} {'平均奖励':<12}")
        print("-" * 80)
        for run_date, reward, times in records:
            times_str = str(times) if times else "-"
            avg = reward / times if times and times > 0 else 0
            avg_str = f"{avg:.2f}" if times else "-"
            print(f"{run_date:<12} {reward:<12.2f} {times_str:<10} {avg_str:<12}")
    else:
        print(f"\n未找到账号 {phone} 的记录")
    
    conn.close()
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    check_checkin_times("17573358250")
