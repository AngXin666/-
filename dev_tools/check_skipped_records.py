"""
检查被跳过的记录

分析无法计算签到奖励的记录,检查转账记录等问题

运行方式:
    python dev_tools/check_skipped_records.py
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


def check_skipped_records():
    """检查被跳过的记录"""
    
    print("=" * 80)
    print("检查被跳过的记录")
    print("=" * 80)
    print()
    
    # 初始化数据库
    db = LocalDatabase()
    
    # 获取所有记录
    all_records = db.get_all_history_records()
    
    if not all_records:
        print("❌ 数据库中没有记录")
        return
    
    print(f"数据库共有 {len(all_records)} 条记录")
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
    
    # 统计信息
    first_record_count = 0  # 第一条记录
    no_balance_count = 0  # 无余额数据
    balance_decrease_count = 0  # 余额减少(可能转账)
    
    skipped_records = []
    
    # 遍历每个账号的记录
    for phone, records in records_by_phone.items():
        previous_balance_after = None
        
        for idx, record in enumerate(records):
            record_id = record.get('id')
            run_date = record.get('run_date')
            balance_before = record.get('balance_before')
            balance_after = record.get('balance_after')
            checkin_balance_after = record.get('checkin_balance_after')
            transfer_amount = record.get('transfer_amount', 0.0) or 0.0
            
            # 检查跳过原因
            skip_reason = None
            
            # 原因1: 无余额数据
            if balance_after is None:
                skip_reason = "无余额数据"
                no_balance_count += 1
            # 原因2: 第一条记录
            elif previous_balance_after is None:
                skip_reason = "第一条记录"
                first_record_count += 1
            # 原因3: 没有签到后余额且余额减少
            elif checkin_balance_after is None:
                balance_diff = balance_after - previous_balance_after
                if balance_diff < 0:
                    skip_reason = f"余额减少(无签到后余额): {balance_diff:.2f}元"
                    balance_decrease_count += 1
            
            if skip_reason:
                skipped_records.append({
                    'phone': phone,
                    'run_date': run_date,
                    'reason': skip_reason,
                    'balance_before': balance_before,
                    'balance_after': balance_after,
                    'checkin_balance_after': checkin_balance_after,
                    'transfer_amount': transfer_amount,
                    'previous_balance': previous_balance_after
                })
            
            # 更新前一天余额
            if balance_after is not None:
                previous_balance_after = balance_after
    
    # 输出统计
    print("跳过记录统计:")
    print(f"  第一条记录: {first_record_count} 条")
    print(f"  无余额数据: {no_balance_count} 条")
    print(f"  余额减少(可能转账): {balance_decrease_count} 条")
    print(f"  总计: {len(skipped_records)} 条")
    print()
    
    # 详细显示余额减少的记录
    if balance_decrease_count > 0:
        print("=" * 80)
        print(f"余额减少的记录详情 (共{balance_decrease_count}条):")
        print("=" * 80)
        print()
        
        for record in skipped_records:
            if "余额减少" in record['reason']:
                print(f"账号: {record['phone']}")
                print(f"日期: {record['run_date']}")
                print(f"原因: {record['reason']}")
                print(f"  前一天余额: {record['previous_balance']:.2f if record['previous_balance'] is not None else 'None'}")
                print(f"  余额前: {record['balance_before']:.2f if record['balance_before'] is not None else 'None'}")
                print(f"  签到后余额: {record['checkin_balance_after'] if record['checkin_balance_after'] is not None else 'None'}")
                print(f"  最终余额: {record['balance_after']:.2f if record['balance_after'] is not None else 'None'}")
                print(f"  转账金额: {record['transfer_amount']:.2f}")
                
                # 分析可能的转账金额
                if record['previous_balance'] is not None and record['balance_after'] is not None:
                    balance_diff = record['balance_after'] - record['previous_balance']
                    print(f"  余额变化: {balance_diff:.2f} 元")
                    
                    # 如果有转账记录,检查是否合理
                    if record['transfer_amount'] > 0:
                        estimated_reward = balance_diff + record['transfer_amount']
                        print(f"  估算签到奖励: {estimated_reward:.2f} 元 (余额变化 + 转账金额)")
                        
                        if estimated_reward < 0 or estimated_reward > 10:
                            print(f"  ⚠️ 估算结果异常,转账金额可能不准确")
                    else:
                        print(f"  ⚠️ 无转账记录,但余额减少了 {abs(balance_diff):.2f} 元")
                
                print()
    
    # 保存到文件
    output_file = "dev_tools/skipped_records_report.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("被跳过的记录报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"总计: {len(skipped_records)} 条\n")
        f.write(f"  第一条记录: {first_record_count} 条\n")
        f.write(f"  无余额数据: {no_balance_count} 条\n")
        f.write(f"  余额减少(可能转账): {balance_decrease_count} 条\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("所有跳过记录详情:\n")
        f.write("=" * 80 + "\n\n")
        
        for record in skipped_records:
            f.write(f"账号: {record['phone']}\n")
            f.write(f"日期: {record['run_date']}\n")
            f.write(f"原因: {record['reason']}\n")
            
            prev_bal = f"{record['previous_balance']:.2f}" if record['previous_balance'] is not None else 'None'
            bal_before = f"{record['balance_before']:.2f}" if record['balance_before'] is not None else 'None'
            bal_after = f"{record['balance_after']:.2f}" if record['balance_after'] is not None else 'None'
            checkin_bal = f"{record['checkin_balance_after']:.2f}" if record['checkin_balance_after'] is not None else 'None'
            
            f.write(f"  前一天余额: {prev_bal}\n")
            f.write(f"  余额前: {bal_before}\n")
            f.write(f"  签到后余额: {checkin_bal}\n")
            f.write(f"  最终余额: {bal_after}\n")
            f.write(f"  转账金额: {record['transfer_amount']:.2f}\n")
            f.write("\n")
    
    print(f"详细报告已保存到: {output_file}")


if __name__ == "__main__":
    try:
        check_skipped_records()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
