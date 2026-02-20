"""
修复第一条记录的签到奖励

第一条记录: checkin_reward = checkin_balance_after - balance_before

运行方式:
    python dev_tools/fix_first_record_reward.py
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


def fix_first_record_reward():
    """修复第一条记录的签到奖励"""
    
    print("=" * 80)
    print("修复第一条记录的签到奖励")
    print("=" * 80)
    print()
    print("规则:")
    print("  checkin_reward = checkin_balance_after - balance_before")
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
        if records:
            # 只处理第一条记录
            first_record = records[0]
            record_id = first_record.get('id')
            run_date = first_record.get('run_date')
            balance_before = first_record.get('balance_before', 0.0) or 0.0
            checkin_balance_after = first_record.get('checkin_balance_after')
            old_checkin_reward = first_record.get('checkin_reward', 0.0) or 0.0
            
            if checkin_balance_after is not None:
                # 计算新的签到奖励
                new_checkin_reward = checkin_balance_after - balance_before
                
                # 检查是否需要更新
                if abs(new_checkin_reward - old_checkin_reward) > 0.001:
                    try:
                        db.update_checkin_reward(record_id, new_checkin_reward)
                        updated_count += 1
                        
                        # 显示前20条更新
                        if updated_count <= 20:
                            print(f"[{phone}] [{run_date}]")
                            print(f"  balance_before: {balance_before:.2f}")
                            print(f"  checkin_balance_after: {checkin_balance_after:.2f}")
                            print(f"  checkin_reward: {old_checkin_reward:.2f} → {new_checkin_reward:.2f}")
                            print()
                    
                    except Exception as e:
                        print(f"❌ [{phone}] [{run_date}] 更新失败: {e}")
                        error_count += 1
    
    # 输出统计信息
    print()
    print("=" * 80)
    print("修复完成")
    print("=" * 80)
    print(f"已更新: {updated_count} 条")
    print(f"错误: {error_count} 条")
    print()


if __name__ == "__main__":
    try:
        fix_first_record_reward()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
