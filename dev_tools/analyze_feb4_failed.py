"""
分析2月4日失败记录

检查：
1. 数据是否有效（有完整的余额、签到次数等）
2. 是否有重复记录（同一账号同一天有多条记录）
3. 与前后记录对比合理性

运行方式:
    python dev_tools/analyze_feb4_failed.py
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


def analyze_feb4_failed():
    """分析2月4日失败记录"""
    
    print("=" * 80)
    print("分析2月4日失败记录")
    print("=" * 80)
    print()
    
    # 初始化数据库
    db = LocalDatabase()
    all_records = db.get_all_history_records()
    
    # 筛选2月4日的记录
    feb4_records = [r for r in all_records if r.get('run_date') == '2026-02-04']
    feb4_failed = [r for r in feb4_records if r.get('status') == '失败']
    feb4_success = [r for r in feb4_records if r.get('status') == '成功']
    
    print(f"2月4日总记录: {len(feb4_records)}")
    print(f"  失败: {len(feb4_failed)}")
    print(f"  成功: {len(feb4_success)}")
    print()
    
    # 检查是否有重复记录
    print("=" * 80)
    print("检查重复记录:")
    print("=" * 80)
    
    phone_count = {}
    for record in feb4_records:
        phone = record.get('phone')
        if phone not in phone_count:
            phone_count[phone] = []
        phone_count[phone].append(record)
    
    duplicates = {phone: records for phone, records in phone_count.items() if len(records) > 1}
    
    if duplicates:
        print(f"发现 {len(duplicates)} 个账号有重复记录:")
        for phone, records in duplicates.items():
            print(f"\n账号 {phone}: {len(records)} 条记录")
            for record in records:
                status = record.get('status')
                checkin_times = record.get('checkin_total_times')
                balance_after = record.get('balance_after')
                print(f"  状态: {status}, 签到次数: {checkin_times}, 最终余额: {balance_after}")
    else:
        print("✓ 没有重复记录")
    print()
    
    # 分析失败记录的数据完整性
    print("=" * 80)
    print("分析失败记录数据完整性:")
    print("=" * 80)
    
    valid_failed = []
    invalid_failed = []
    
    for record in feb4_failed:
        phone = record.get('phone')
        checkin_times = record.get('checkin_total_times') or 0
        balance_before = record.get('balance_before')
        checkin_reward = record.get('checkin_reward')
        checkin_balance_after = record.get('checkin_balance_after')
        balance_after = record.get('balance_after')
        
        # 判断数据是否完整
        is_valid = True
        reasons = []
        
        if checkin_times == 0:
            is_valid = False
            reasons.append("签到次数为0")
        
        if balance_before is None:
            is_valid = False
            reasons.append("缺少余额前")
        
        if balance_after is None:
            is_valid = False
            reasons.append("缺少最终余额")
        
        if checkin_balance_after is None:
            is_valid = False
            reasons.append("缺少签到后余额")
        
        if is_valid:
            valid_failed.append(record)
        else:
            invalid_failed.append({
                'phone': phone,
                'reasons': reasons
            })
    
    print(f"有效数据: {len(valid_failed)} 条")
    print(f"无效数据: {len(invalid_failed)} 条")
    
    if invalid_failed:
        print("\n无效数据详情:")
        for item in invalid_failed[:10]:
            print(f"  账号 {item['phone']}: {', '.join(item['reasons'])}")
    print()
    
    # 对比失败记录与成功记录的合理性
    print("=" * 80)
    print("对比失败与成功记录:")
    print("=" * 80)
    
    # 统计失败记录的特征
    failed_with_transfer = [r for r in feb4_failed if (r.get('transfer_amount') or 0) > 0]
    failed_without_transfer = [r for r in feb4_failed if (r.get('transfer_amount') or 0) == 0]
    
    success_with_transfer = [r for r in feb4_success if (r.get('transfer_amount') or 0) > 0]
    success_without_transfer = [r for r in feb4_success if (r.get('transfer_amount') or 0) == 0]
    
    print(f"失败记录:")
    print(f"  有转账: {len(failed_with_transfer)}")
    print(f"  无转账: {len(failed_without_transfer)}")
    print()
    print(f"成功记录:")
    print(f"  有转账: {len(success_with_transfer)}")
    print(f"  无转账: {len(success_without_transfer)}")
    print()
    
    # 建议
    print("=" * 80)
    print("建议:")
    print("=" * 80)
    
    if len(valid_failed) > 0:
        print(f"✓ 有 {len(valid_failed)} 条失败记录的数据是完整有效的")
        print(f"  建议：将这些记录的状态改为'成功'")
    
    if len(invalid_failed) > 0:
        print(f"⚠️ 有 {len(invalid_failed)} 条失败记录的数据不完整")
        print(f"  建议：检查这些记录是否需要删除或修复")
    
    if len(duplicates) > 0:
        print(f"⚠️ 有 {len(duplicates)} 个账号有重复记录")
        print(f"  建议：检查重复记录，保留合理的一条")
    
    print()
    
    return valid_failed


if __name__ == "__main__":
    try:
        analyze_feb4_failed()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
