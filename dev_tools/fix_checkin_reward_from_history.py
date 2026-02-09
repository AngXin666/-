"""修复历史签到奖励数据
通过对比昨天和今天的余额来计算签到奖励
"""
import sys
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def fix_checkin_rewards(target_date=None):
    """修复签到奖励数据
    
    Args:
        target_date: 要修复的日期（格式：YYYY-MM-DD），默认为今天
    """
    db_path = project_root / "runtime_data" / "license.db"
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # 获取目标日期
    if target_date is None:
        target_date = datetime.now().strftime('%Y-%m-%d')
    
    # 计算前一天的日期
    target_dt = datetime.strptime(target_date, '%Y-%m-%d')
    previous_date = (target_dt - timedelta(days=1)).strftime('%Y-%m-%d')
    
    print("="*100)
    print(f"修复签到奖励数据 - {target_date}")
    print("="*100)
    print()
    
    # 查询目标日期签到奖励为0或NULL的记录
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
        print("✓ 没有需要修复的记录")
        conn.close()
        return
    
    print(f"找到 {len(target_records)} 条需要修复的记录")
    print()
    
    fixed_count = 0
    skipped_count = 0
    
    for record in target_records:
        record_id, phone, balance_before, balance_after, checkin_reward = record
        
        print(f"处理账号: {phone}")
        print(f"  当前数据: 余额前={balance_before}, 余额后={balance_after}, 签到奖励={checkin_reward}")
        
        # 方法1: 如果有 balance_before，直接用 balance_after - balance_before
        if balance_before is not None and balance_after is not None:
            calculated_reward = balance_after - balance_before
            
            if calculated_reward > 0.001:  # 大于0.001才认为有变化（避免浮点数精度问题）
                # 更新签到奖励
                cursor.execute("""
                    UPDATE history_records
                    SET checkin_reward = ?
                    WHERE id = ?
                """, (round(calculated_reward, 2), record_id))
                
                print(f"  ✓ 方法1: 使用当日数据计算")
                print(f"    签到奖励 = {balance_after:.2f} - {balance_before:.2f} = {calculated_reward:.2f} 元")
                fixed_count += 1
                continue
            elif abs(calculated_reward) < 0.001:
                # 余额没有变化，说明今日已签到或未签到
                print(f"  ⚠️ 方法1: 余额无变化，可能今日已签到，跳过")
                skipped_count += 1
                continue
            else:
                print(f"  ⚠️ 方法1: 计算结果 < 0 ({calculated_reward:.2f})，尝试方法2")
        
        # 方法2: 查询前一天的 balance_after，用当天的 balance_after - 前一天的 balance_after
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
                
                print(f"  ✓ 方法2: 使用前一天余额计算")
                print(f"    前一天余额: {previous_balance:.2f} 元")
                print(f"    签到奖励 = {balance_after:.2f} - {previous_balance:.2f} = {calculated_reward:.2f} 元")
                fixed_count += 1
                continue
            elif abs(calculated_reward) < 0.001:
                print(f"  ⚠️ 方法2: 余额无变化，可能今日已签到，跳过")
            else:
                print(f"  ⚠️ 方法2: 计算结果 < 0 ({calculated_reward:.2f})，可能余额减少了")
        else:
            print(f"  ⚠️ 方法2: 未找到前一天的记录")
        
        # 如果两种方法都失败，跳过
        print(f"  ✗ 跳过: 无法计算签到奖励")
        skipped_count += 1
        print()
    
    # 提交更改
    conn.commit()
    conn.close()
    
    print()
    print("="*100)
    print(f"修复完成")
    print(f"  ✓ 成功修复: {fixed_count} 条")
    print(f"  ✗ 跳过: {skipped_count} 条")
    print("="*100)
    
    # 显示修复后的数据
    if fixed_count > 0:
        print()
        print("修复后的数据预览:")
        print("-"*100)
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT phone, balance_before, checkin_reward, balance_after
            FROM history_records
            WHERE run_date = ?
            AND checkin_reward > 0
            ORDER BY phone
            LIMIT 20
        """, (target_date,))
        
        rows = cursor.fetchall()
        
        print(f"{'手机号':<15} {'余额前':<10} {'签到奖励':<12} {'余额后':<10}")
        print("-"*100)
        
        for row in rows:
            phone, balance_before, checkin_reward, balance_after = row
            print(f"{phone:<15} {balance_before if balance_before else 'N/A':<10} {checkin_reward:<12.2f} {balance_after if balance_after else 'N/A':<10}")
        
        conn.close()

if __name__ == '__main__':
    # 支持命令行参数指定日期
    if len(sys.argv) > 1:
        target_date = sys.argv[1]
        fix_checkin_rewards(target_date)
    else:
        # 默认修复今天的数据
        fix_checkin_rewards()
