"""分析被跳过的记录
查看为什么这些记录的签到奖励为0
"""
import sys
import sqlite3
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def analyze_skipped_records():
    """分析被跳过的记录"""
    db_path = project_root / "runtime_data" / "license.db"
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    print("="*100)
    print("分析签到奖励为0的记录")
    print("="*100)
    print()
    
    # 查询所有签到奖励为0的记录
    cursor.execute("""
        SELECT run_date, phone, balance_before, balance_after, checkin_reward, status
        FROM history_records
        WHERE (checkin_reward IS NULL OR checkin_reward = 0 OR checkin_reward = 0.0)
        AND run_date >= '2026-02-02'
        ORDER BY run_date, phone
        LIMIT 50
    """)
    
    rows = cursor.fetchall()
    
    print(f"找到 {len(rows)} 条签到奖励为0的记录（显示前50条）")
    print()
    
    # 分类统计
    no_balance_data = 0  # 没有余额数据
    balance_no_change = 0  # 余额没有变化
    balance_decreased = 0  # 余额减少了
    balance_increased = 0  # 余额增加了但奖励为0
    
    print(f"{'日期':<15} {'手机号':<15} {'余额前':<10} {'余额后':<10} {'变化':<10} {'状态':<10} {'原因'}")
    print("-"*100)
    
    for row in rows:
        run_date, phone, balance_before, balance_after, checkin_reward, status = row
        
        reason = ""
        
        if balance_before is None or balance_after is None:
            reason = "缺少余额数据"
            no_balance_data += 1
        elif abs(balance_after - balance_before) < 0.001:
            reason = "余额无变化（今日已签到）"
            balance_no_change += 1
        elif balance_after < balance_before:
            reason = f"余额减少（可能转账）"
            balance_decreased += 1
        else:
            reason = "余额增加但奖励为0（异常）"
            balance_increased += 1
        
        balance_before_str = f"{balance_before:.2f}" if balance_before is not None else "None"
        balance_after_str = f"{balance_after:.2f}" if balance_after is not None else "None"
        
        if balance_before is not None and balance_after is not None:
            change = balance_after - balance_before
            change_str = f"{change:+.2f}"
        else:
            change_str = "N/A"
        
        print(f"{run_date:<15} {phone:<15} {balance_before_str:<10} {balance_after_str:<10} {change_str:<10} {status:<10} {reason}")
    
    print()
    print("="*100)
    print("统计分析")
    print("="*100)
    print(f"缺少余额数据: {no_balance_data} 条")
    print(f"余额无变化（今日已签到）: {balance_no_change} 条")
    print(f"余额减少（可能转账）: {balance_decreased} 条")
    print(f"余额增加但奖励为0（异常）: {balance_increased} 条")
    print()
    
    # 查询有签到奖励的记录统计
    cursor.execute("""
        SELECT COUNT(*) 
        FROM history_records 
        WHERE checkin_reward > 0
        AND run_date >= '2026-02-02'
    """)
    
    has_reward_count = cursor.fetchone()[0]
    
    print(f"有签到奖励的记录: {has_reward_count} 条")
    print()
    
    conn.close()

if __name__ == '__main__':
    analyze_skipped_records()
