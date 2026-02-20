"""
检查2月4日失败记录的详细情况

检查这些失败记录是否是无效数据

运行方式:
    python dev_tools/check_feb4_failed_records.py
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


def check_feb4_failed_records():
    """检查2月4日失败记录的详细情况"""
    
    print("=" * 80)
    print("检查2月4日失败记录")
    print("=" * 80)
    print()
    
    # 初始化数据库
    db = LocalDatabase()
    all_records = db.get_all_history_records()
    
    # 筛选2月4日的失败记录
    feb4_failed_records = []
    for record in all_records:
        run_date = record.get('run_date')
        status = record.get('status')
        if run_date == '2026-02-04' and status == '失败':
            feb4_failed_records.append(record)
    
    print(f"找到 {len(feb4_failed_records)} 条2月4日的失败记录")
    print()
    
    if not feb4_failed_records:
        print("✓ 没有找到2月4日的失败记录")
        return
    
    # 按账号分组
    records_by_phone = {}
    for record in feb4_failed_records:
        phone = record.get('phone')
        if phone not in records_by_phone:
            records_by_phone[phone] = []
        records_by_phone[phone].append(record)
    
    print(f"涉及 {len(records_by_phone)} 个账号")
    print()
    
    # 检查每个账号的前后记录
    print("=" * 80)
    print("详细分析:")
    print("=" * 80)
    print()
    
    invalid_records = []
    
    for phone, failed_records in records_by_phone.items():
        # 获取该账号的所有记录
        all_phone_records = [r for r in all_records if r.get('phone') == phone]
        all_phone_records.sort(key=lambda r: r.get('run_date', ''))
        
        for failed_record in failed_records:
            record_id = failed_record.get('id')
            run_date = failed_record.get('run_date')
            checkin_times = failed_record.get('checkin_total_times') or 0
            balance_before = failed_record.get('balance_before')
            checkin_reward = failed_record.get('checkin_reward', 0.0) or 0.0
            checkin_balance_after = failed_record.get('checkin_balance_after')
            transfer_amount = failed_record.get('transfer_amount', 0.0) or 0.0
            balance_after = failed_record.get('balance_after')
            
            # 找到前一天和后一天的记录
            idx = all_phone_records.index(failed_record)
            prev_record = all_phone_records[idx - 1] if idx > 0 else None
            next_record = all_phone_records[idx + 1] if idx < len(all_phone_records) - 1 else None
            
            print(f"账号: {phone}")
            print(f"  日期: {run_date}")
            print(f"  签到次数: {checkin_times}")
            print(f"  余额前: {balance_before}")
            print(f"  签到奖励: {checkin_reward:.2f}")
            print(f"  签到后余额: {checkin_balance_after}")
            print(f"  转账金额: {transfer_amount:.2f}")
            print(f"  最终余额: {balance_after}")
            
            if prev_record:
                prev_date = prev_record.get('run_date')
                prev_balance_after = prev_record.get('balance_after')
                print(f"  前一天({prev_date}): 最终余额 = {prev_balance_after}")
                
                # 检查是否连续
                if balance_before is not None and prev_balance_after is not None:
                    if abs(balance_before - prev_balance_after) > 0.01:
                        print(f"    ⚠️ 余额不连续: {prev_balance_after:.2f} → {balance_before:.2f}")
            
            if next_record:
                next_date = next_record.get('run_date')
                next_balance_before = next_record.get('balance_before')
                print(f"  后一天({next_date}): 余额前 = {next_balance_before}")
                
                # 检查是否连续
                if balance_after is not None and next_balance_before is not None:
                    if abs(balance_after - next_balance_before) > 0.01:
                        print(f"    ⚠️ 余额不连续: {balance_after:.2f} → {next_balance_before:.2f}")
            
            # 判断是否是无效数据
            is_invalid = False
            reasons = []
            
            # 检查1: 签到次数为0
            if checkin_times == 0:
                is_invalid = True
                reasons.append("签到次数为0")
            
            # 检查2: 余额数据缺失
            if balance_before is None or balance_after is None:
                is_invalid = True
                reasons.append("余额数据缺失")
            
            # 检查3: 签到后余额为0但签到奖励不为0
            if checkin_balance_after == 0 and checkin_reward > 0:
                is_invalid = True
                reasons.append("签到后余额为0但签到奖励不为0")
            
            if is_invalid:
                print(f"  ❌ 无效数据: {', '.join(reasons)}")
                invalid_records.append(record_id)
            else:
                print(f"  ✓ 数据完整")
            
            print()
    
    # 总结
    print("=" * 80)
    print("总结:")
    print("=" * 80)
    print(f"失败记录总数: {len(feb4_failed_records)}")
    print(f"无效记录数: {len(invalid_records)}")
    print(f"有效记录数: {len(feb4_failed_records) - len(invalid_records)}")
    print()
    
    if invalid_records:
        print("建议: 删除这些无效记录")
    else:
        print("建议: 所有失败记录的数据都是完整的，不建议删除")


if __name__ == "__main__":
    try:
        check_feb4_failed_records()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
