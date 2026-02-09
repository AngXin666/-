"""批量修复历史签到奖励数据
通过与前一天余额对比来计算签到奖励
"""
import sys
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def get_all_dates_with_records(cursor):
    """获取所有有记录的日期"""
    cursor.execute("""
        SELECT DISTINCT run_date 
        FROM history_records 
        ORDER BY run_date
    """)
    
    dates = [row[0] for row in cursor.fetchall()]
    return dates

def fix_all_dates():
    """修复所有日期的签到奖励数据"""
    db_path = project_root / "runtime_data" / "license.db"
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # 获取所有有记录的日期
    all_dates = get_all_dates_with_records(cursor)
    
    if not all_dates:
        print("❌ 数据库中没有历史记录")
        conn.close()
        return
    
    print("="*100)
    print(f"批量修复签到奖励数据")
    print("="*100)
    print(f"找到 {len(all_dates)} 个日期需要检查")
    print(f"日期范围: {all_dates[0]} 到 {all_dates[-1]}")
    print()
    
    total_fixed = 0
    total_skipped = 0
    date_summary = []
    
    # 按日期顺序处理（从旧到新）
    for i, target_date in enumerate(all_dates):
        print(f"\n{'='*100}")
        print(f"处理日期: {target_date} ({i+1}/{len(all_dates)})")
        print(f"{'='*100}")
        
        # 计算前一天的日期
        target_dt = datetime.strptime(target_date, '%Y-%m-%d')
        previous_date = (target_dt - timedelta(days=1)).strftime('%Y-%m-%d')
        
        # 查询当天签到奖励为0或NULL的记录
        cursor.execute("""
            SELECT id, phone, balance_before, balance_after, checkin_reward
            FROM history_records
            WHERE run_date = ?
            AND (checkin_reward IS NULL OR checkin_reward = 0 OR checkin_reward = 0.0)
            AND balance_after IS NOT NULL
            ORDER BY phone
        """, (target_date,))
        
        target_records = cursor.fetchall()
        
        if not target_records:
            print(f"✓ 没有需要修复的记录")
            date_summary.append({
                'date': target_date,
                'fixed': 0,
                'skipped': 0,
                'total': 0
            })
            continue
        
        print(f"找到 {len(target_records)} 条需要检查的记录")
        
        date_fixed = 0
        date_skipped = 0
        
        for record in target_records:
            record_id, phone, balance_before, balance_after, checkin_reward = record
            
            # 方法1: 如果有 balance_before，直接用 balance_after - balance_before
            if balance_before is not None and balance_after is not None:
                calculated_reward = balance_after - balance_before
                
                if calculated_reward > 0.001:  # 大于0.001才认为有变化
                    # 更新签到奖励
                    cursor.execute("""
                        UPDATE history_records
                        SET checkin_reward = ?
                        WHERE id = ?
                    """, (round(calculated_reward, 2), record_id))
                    
                    print(f"  ✓ {phone}: 方法1计算 = {balance_after:.2f} - {balance_before:.2f} = {calculated_reward:.2f} 元")
                    date_fixed += 1
                    continue
                elif abs(calculated_reward) < 0.001:
                    # 余额没有变化，说明今日已签到或未签到
                    date_skipped += 1
                    continue
            
            # 方法2: 查询前一天的 balance_after
            cursor.execute("""
                SELECT balance_after
                FROM history_records
                WHERE phone = ? AND run_date = ?
                AND balance_after IS NOT NULL
                ORDER BY created_at DESC
                LIMIT 1
            """, (phone, previous_date))
            
            previous_record = cursor.fetchone()
            
            if previous_record and previous_record[0] is not None:
                previous_balance = previous_record[0]
                calculated_reward = balance_after - previous_balance
                
                if calculated_reward > 0.001:
                    # 更新签到奖励
                    cursor.execute("""
                        UPDATE history_records
                        SET checkin_reward = ?
                        WHERE id = ?
                    """, (round(calculated_reward, 2), record_id))
                    
                    print(f"  ✓ {phone}: 方法2计算 = {balance_after:.2f} - {previous_balance:.2f} = {calculated_reward:.2f} 元")
                    date_fixed += 1
                    continue
                elif abs(calculated_reward) < 0.001:
                    date_skipped += 1
                    continue
            
            # 如果两种方法都失败，跳过
            date_skipped += 1
        
        # 提交当天的更改
        conn.commit()
        
        print(f"\n{target_date} 处理完成:")
        print(f"  ✓ 成功修复: {date_fixed} 条")
        print(f"  ✗ 跳过: {date_skipped} 条")
        
        total_fixed += date_fixed
        total_skipped += date_skipped
        
        date_summary.append({
            'date': target_date,
            'fixed': date_fixed,
            'skipped': date_skipped,
            'total': len(target_records)
        })
    
    conn.close()
    
    # 显示总结
    print()
    print("="*100)
    print("批量修复完成")
    print("="*100)
    print(f"总计:")
    print(f"  ✓ 成功修复: {total_fixed} 条")
    print(f"  ✗ 跳过: {total_skipped} 条")
    print()
    
    # 显示每日统计
    print("每日统计:")
    print(f"{'日期':<15} {'需检查':<10} {'已修复':<10} {'已跳过':<10}")
    print("-"*100)
    
    for summary in date_summary:
        if summary['total'] > 0:
            print(f"{summary['date']:<15} {summary['total']:<10} {summary['fixed']:<10} {summary['skipped']:<10}")
    
    print()
    
    # 显示修复后的数据预览
    if total_fixed > 0:
        print("="*100)
        print("修复后的数据预览（最近20条有签到奖励的记录）:")
        print("="*100)
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT run_date, phone, balance_before, checkin_reward, balance_after
            FROM history_records
            WHERE checkin_reward > 0
            ORDER BY run_date DESC, phone
            LIMIT 20
        """)
        
        rows = cursor.fetchall()
        
        print(f"{'日期':<15} {'手机号':<15} {'余额前':<10} {'签到奖励':<12} {'余额后':<10}")
        print("-"*100)
        
        for row in rows:
            run_date, phone, balance_before, checkin_reward, balance_after = row
            balance_before_str = f"{balance_before:.2f}" if balance_before else "N/A"
            balance_after_str = f"{balance_after:.2f}" if balance_after else "N/A"
            print(f"{run_date:<15} {phone:<15} {balance_before_str:<10} {checkin_reward:<12.2f} {balance_after_str:<10}")
        
        conn.close()

if __name__ == '__main__':
    fix_all_dates()
