"""
分析签到奖励累计问题
"""

import sys
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.local_db import LocalDatabase

def analyze_reward_accumulation(phone):
    """分析特定账号的签到奖励累计情况"""
    db = LocalDatabase()
    
    print("=" * 80)
    print(f"分析账号 {phone} 的签到奖励累计情况")
    print("=" * 80)
    
    conn = db._get_connection()
    cursor = conn.cursor()
    
    # 查询该账号的所有记录（按创建时间排序）
    cursor.execute("""
        SELECT id, run_date, checkin_reward, created_at
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
    print("-" * 80)
    print(f"{'ID':<8} {'日期':<12} {'签到奖励':<12} {'创建时间':<20} {'说明':<30}")
    print("-" * 80)
    
    # 分析每条记录
    date_records = {}  # {run_date: [(id, reward, created_at), ...]}
    
    for record_id, run_date, reward, created_at in records:
        if run_date not in date_records:
            date_records[run_date] = []
        date_records[run_date].append((record_id, reward, created_at))
    
    # 打印记录并分析
    for record_id, run_date, reward, created_at in records:
        # 检查是否是同一天的多条记录
        same_day_records = date_records[run_date]
        
        if len(same_day_records) > 1:
            # 找到这条记录在同一天中的位置
            index = next(i for i, (rid, _, _) in enumerate(same_day_records) if rid == record_id)
            if index == 0:
                note = f"⚠️ 同一天第1次运行"
            else:
                prev_reward = same_day_records[index-1][1]
                note = f"⚠️ 同一天第{index+1}次运行（前一次：{prev_reward:.2f}元）"
        else:
            note = "✅ 正常"
        
        print(f"{record_id:<8} {run_date:<12} {reward:<12.2f} {created_at:<20} {note:<30}")
    
    # 统计同一天多次运行的情况
    print("\n" + "=" * 80)
    print("同一天多次运行统计：")
    print("=" * 80)
    
    multi_run_dates = {date: recs for date, recs in date_records.items() if len(recs) > 1}
    
    if multi_run_dates:
        for run_date, recs in sorted(multi_run_dates.items()):
            print(f"\n日期: {run_date} - 运行了 {len(recs)} 次")
            for i, (record_id, reward, created_at) in enumerate(recs, 1):
                print(f"  第{i}次: {reward:.2f}元 (ID:{record_id}, 时间:{created_at})")
            
            # 计算累加情况
            if len(recs) >= 2:
                total_accumulated = sum(r[1] for r in recs)
                last_reward = recs[-1][1]
                print(f"  ⚠️ 如果累加：{total_accumulated:.2f}元")
                print(f"  ✅ 应该保留：{last_reward:.2f}元（最后一次运行的结果）")
    else:
        print("\n✅ 没有同一天多次运行的情况")
    
    conn.close()
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    # 分析异常账号
    analyze_reward_accumulation("17573358250")
