"""
检查签到奖励累计问题

检查数据库中是否有签到奖励被错误累计的记录
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.local_db import LocalDatabase

def check_checkin_reward_accumulation():
    """检查签到奖励累计问题"""
    db = LocalDatabase()
    
    print("=" * 80)
    print("检查签到奖励累计问题")
    print("=" * 80)
    
    # 查询所有记录，按手机号和日期分组
    conn = db._get_connection()
    cursor = conn.cursor()
    
    # 查询每个账号每天的记录数
    cursor.execute("""
        SELECT phone, run_date, COUNT(*) as count, 
               GROUP_CONCAT(checkin_reward) as rewards,
               GROUP_CONCAT(id) as ids
        FROM history_records
        WHERE checkin_reward > 0
        GROUP BY phone, run_date
        HAVING COUNT(*) > 1
        ORDER BY run_date DESC, phone
    """)
    
    duplicate_records = cursor.fetchall()
    
    if duplicate_records:
        print(f"\n⚠️ 发现 {len(duplicate_records)} 个账号在同一天有多条记录：")
        print("-" * 80)
        for phone, run_date, count, rewards, ids in duplicate_records:
            print(f"手机号: {phone}")
            print(f"日期: {run_date}")
            print(f"记录数: {count}")
            print(f"签到奖励: {rewards}")
            print(f"记录ID: {ids}")
            print("-" * 80)
    else:
        print("\n✅ 没有发现同一天有多条记录的情况")
    
    # 查询签到奖励异常高的记录（可能是累计的）
    cursor.execute("""
        SELECT phone, run_date, checkin_reward, nickname
        FROM history_records
        WHERE checkin_reward > 10
        ORDER BY checkin_reward DESC
        LIMIT 20
    """)
    
    high_reward_records = cursor.fetchall()
    
    if high_reward_records:
        print(f"\n⚠️ 发现 {len(high_reward_records)} 条签到奖励异常高的记录（>10元）：")
        print("-" * 80)
        print(f"{'手机号':<15} {'日期':<12} {'签到奖励':<10} {'昵称':<15}")
        print("-" * 80)
        for phone, run_date, reward, nickname in high_reward_records:
            print(f"{phone:<15} {run_date:<12} {reward:<10.2f} {nickname or '-':<15}")
    else:
        print("\n✅ 没有发现签到奖励异常高的记录")
    
    # 统计签到奖励的分布
    cursor.execute("""
        SELECT 
            CASE 
                WHEN checkin_reward = 0 THEN '0元'
                WHEN checkin_reward > 0 AND checkin_reward <= 1 THEN '0-1元'
                WHEN checkin_reward > 1 AND checkin_reward <= 2 THEN '1-2元'
                WHEN checkin_reward > 2 AND checkin_reward <= 5 THEN '2-5元'
                WHEN checkin_reward > 5 AND checkin_reward <= 10 THEN '5-10元'
                ELSE '>10元'
            END as range,
            COUNT(*) as count
        FROM history_records
        WHERE checkin_reward IS NOT NULL
        GROUP BY range
        ORDER BY 
            CASE range
                WHEN '0元' THEN 1
                WHEN '0-1元' THEN 2
                WHEN '1-2元' THEN 3
                WHEN '2-5元' THEN 4
                WHEN '5-10元' THEN 5
                ELSE 6
            END
    """)
    
    distribution = cursor.fetchall()
    
    print("\n📊 签到奖励分布：")
    print("-" * 80)
    print(f"{'范围':<15} {'数量':<10}")
    print("-" * 80)
    for range_name, count in distribution:
        print(f"{range_name:<15} {count:<10}")
    
    conn.close()
    
    print("\n" + "=" * 80)
    print("检查完成")
    print("=" * 80)

if __name__ == "__main__":
    check_checkin_reward_accumulation()
