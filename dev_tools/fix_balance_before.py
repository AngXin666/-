"""
修复签到前余额

将所有记录的 balance_before 设置为前一天的 balance_after
第一条记录的 balance_before 设为 0

运行方式:
    python dev_tools/fix_balance_before.py
"""

import sys
import os
from pathlib import Path

# 设置标准输出编码为 UTF-8
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.local_db import LocalDatabase


def fix_balance_before():
    """修复签到前余额"""
    
    print("=" * 80)
    print("修复签到前余额")
    print("=" * 80)
    print()
    print("规则:")
    print("  balance_before = 前一天的 balance_after")
    print("  第一条记录: balance_before = 0")
    print()
    
    # 初始化数据库
    db = LocalDatabase()
    all_records = db.get_all_history_records()
    
    print(f"总记录数: {len(all_records)}")
    print()
    
    # 按账号分组
    records_by_phone = {}
    for record in all_records:
        phone = record.get('phone')
        if phone:
            if phone not in records_by_phone:
                records_by_phone[phone] = []
            records_by_phone[phone].append(record)
    
    # 对每个账号的记录按日期排序
    for phone in records_by_phone:
        records_by_phone[phone].sort(key=lambda r: r.get('run_date', ''))
    
    print(f"共 {len(records_by_phone)} 个账号")
    print()
    
    # 开始修复
    print("=" * 80)
    print("开始修复...")
    print("=" * 80)
    print()
    
    updated_count = 0
    error_count = 0
    
    for phone, records in records_by_phone.items():
        previous_balance_after = None
        
        for idx, record in enumerate(records):
            record_id = record.get('id')
            run_date = record.get('run_date')
            old_balance_before = record.get('balance_before')
            balance_after = record.get('balance_after')
            
            # 确定新的 balance_before
            if idx == 0:
                # 第一条记录，balance_before = 0
                new_balance_before = 0.0
            else:
                # 非第一条记录，balance_before = 前一天的 balance_after
                if previous_balance_after is not None:
                    new_balance_before = previous_balance_after
                else:
                    # 前一天没有 balance_after，跳过
                    previous_balance_after = balance_after
                    continue
            
            # 检查是否需要更新
            if old_balance_before is None or abs(new_balance_before - old_balance_before) > 0.001:
                try:
                    conn = db._get_connection()
                    cursor = conn.cursor()
                    cursor.execute('''
                        UPDATE history_records 
                        SET balance_before = ?
                        WHERE id = ?
                    ''', (new_balance_before, record_id))
                    conn.commit()
                    conn.close()
                    
                    updated_count += 1
                    
                    # 显示前20条更新
                    if updated_count <= 20:
                        old_bal_str = f"{old_balance_before:.2f}" if old_balance_before is not None else 'None'
                        print(f"[{phone}] [{run_date}]")
                        print(f"  balance_before: {old_bal_str} → {new_balance_before:.2f}")
                        if idx == 0:
                            print(f"  (第一条记录)")
                        else:
                            print(f"  (前一天 balance_after: {previous_balance_after:.2f})")
                        print()
                
                except Exception as e:
                    print(f"❌ [{phone}] [{run_date}] 更新失败: {e}")
                    error_count += 1
            
            # 更新前一天的 balance_after
            previous_balance_after = balance_after
    
    # 输出统计信息
    print()
    print("=" * 80)
    print("修复完成")
    print("=" * 80)
    print(f"已更新: {updated_count} 条")
    print(f"错误: {error_count} 条")
    print()
    
    # 验证修复结果
    print("=" * 80)
    print("验证修复结果...")
    print("=" * 80)
    print()
    
    all_records = db.get_all_history_records()
    
    # 按账号分组
    records_by_phone = {}
    for record in all_records:
        phone = record.get('phone')
        if phone:
            if phone not in records_by_phone:
                records_by_phone[phone] = []
            records_by_phone[phone].append(record)
    
    # 对每个账号的记录按日期排序
    for phone in records_by_phone:
        records_by_phone[phone].sort(key=lambda r: r.get('run_date', ''))
    
    # 检查是否有不一致的情况
    problem_cases = 0
    
    for phone, records in records_by_phone.items():
        previous_balance_after = None
        
        for idx, record in enumerate(records):
            balance_before = record.get('balance_before')
            balance_after = record.get('balance_after')
            
            if idx == 0:
                # 第一条记录，balance_before 应该是 0
                if balance_before is not None and abs(balance_before - 0.0) > 0.001:
                    problem_cases += 1
            else:
                # 非第一条记录，balance_before 应该等于前一天的 balance_after
                if previous_balance_after is not None and balance_before is not None:
                    if abs(balance_before - previous_balance_after) > 0.001:
                        problem_cases += 1
            
            previous_balance_after = balance_after
    
    print(f"balance_before 不一致的记录: {problem_cases} 条")
    
    if problem_cases > 0:
        print("⚠️ 仍有部分记录的 balance_before 不一致")
    else:
        print("✓ 所有记录的 balance_before 都已正确")
    
    print()


if __name__ == "__main__":
    try:
        fix_balance_before()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
