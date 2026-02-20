"""
分析时间计算问题
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.local_db import LocalDatabase

def analyze_time_calculation(phone):
    """分析时间计算问题"""
    db = LocalDatabase()
    
    print("=" * 80)
    print(f"分析账号 {phone} 的时间计算")
    print("=" * 80)
    
    conn = db._get_connection()
    cursor = conn.cursor()
    
    # 查询该账号的所有记录
    cursor.execute("""
        SELECT run_date, created_at, checkin_reward, checkin_total_times
        FROM history_records
        WHERE phone = ?
        ORDER BY created_at
    """, (phone,))
    
    records = cursor.fetchall()
    
    if not records:
        print(f"\n未找到账号 {phone} 的记录")
        conn.close()
        return
    
    print(f"\n找到 {len(records)} 条记录：")
    print("-" * 100)
    print(f"{'run_date':<12} {'created_at':<20} {'签到奖励':<12} {'签到次数':<10} {'时间差':<15}")
    print("-" * 100)
    
    for i, (run_date, created_at, reward, times) in enumerate(records):
        # 解析时间
        created_dt = datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S')
        run_dt = datetime.strptime(run_date, '%Y-%m-%d')
        
        # 计算时间差
        time_diff = (run_dt - created_dt.replace(hour=0, minute=0, second=0, microsecond=0)).days
        
        # 判断是否异常
        if time_diff == 0:
            time_diff_str = "✅ 同一天"
        elif time_diff == 1:
            time_diff_str = f"⚠️ 提前1天"
        elif time_diff == -1:
            time_diff_str = f"⚠️ 延后1天"
        else:
            time_diff_str = f"❌ 差{time_diff}天"
        
        times_str = str(times) if times else "-"
        
        print(f"{run_date:<12} {created_at:<20} {reward:<12.2f} {times_str:<10} {time_diff_str:<15}")
        
        # 分析累计逻辑
        if i > 0:
            prev_run_date, prev_created_at, prev_reward, prev_times = records[i-1]
            
            # 检查是否是同一个 run_date
            if run_date == prev_run_date:
                print(f"  ⚠️ 与上一条记录的 run_date 相同！这不应该发生（有唯一约束）")
            
            # 检查 created_at 的时间间隔
            prev_created_dt = datetime.strptime(prev_created_at, '%Y-%m-%d %H:%M:%S')
            created_interval = (created_dt - prev_created_dt).total_seconds() / 3600  # 小时
            
            if created_interval < 1:
                print(f"  ⚠️ 与上一条记录间隔 {created_interval:.1f} 小时（可能是同一次运行的多次更新）")
            elif created_interval < 12:
                print(f"  ⚠️ 与上一条记录间隔 {created_interval:.1f} 小时（同一天内多次运行）")
    
    conn.close()
    
    print("\n" + "=" * 100)
    print("分析结论：")
    print("=" * 100)
    
    # 统计时间差分布
    time_diff_count = {}
    for run_date, created_at, _, _ in records:
        created_dt = datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S')
        run_dt = datetime.strptime(run_date, '%Y-%m-%d')
        time_diff = (run_dt - created_dt.replace(hour=0, minute=0, second=0, microsecond=0)).days
        time_diff_count[time_diff] = time_diff_count.get(time_diff, 0) + 1
    
    print("\n时间差分布：")
    for diff, count in sorted(time_diff_count.items()):
        if diff == 0:
            print(f"  同一天: {count} 条记录 ✅")
        elif diff == 1:
            print(f"  提前1天: {count} 条记录 ⚠️ (run_date 比 created_at 早1天)")
        elif diff == -1:
            print(f"  延后1天: {count} 条记录 ⚠️ (run_date 比 created_at 晚1天)")
        else:
            print(f"  差{diff}天: {count} 条记录 ❌")
    
    print("\n" + "=" * 100)

if __name__ == "__main__":
    analyze_time_calculation("17573358250")
