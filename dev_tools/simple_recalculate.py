"""
简单重新计算签到奖励

只处理有完整数据的记录，数据不完整的标记出来

运行方式:
    python dev_tools/simple_recalculate.py
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


def simple_recalculate():
    """简单重新计算签到奖励"""
    
    print("=" * 80)
    print("简单重新计算签到奖励")
    print("=" * 80)
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
    
    # 统计
    complete_records = 0
    incomplete_records = []
    updated_records = 0
    
    # 遍历每个账号
    for phone, records in records_by_phone.items():
        previous_checkin_balance_after = None
        
        for idx, record in enumerate(records):
            record_id = record.get('id')
            run_date = record.get('run_date')
            checkin_balance_after = record.get('checkin_balance_after')
            old_checkin_reward = record.get('checkin_reward', 0.0) or 0.0
            
            # 检查数据完整性
            if checkin_balance_after is None:
                incomplete_records.append({
                    'phone': phone,
                    'date': run_date,
                    'reason': '缺少签到后余额'
                })
                continue
            
            # 第一条记录，前一天余额默认为0
            if idx == 0:
                previous_balance = 0.0
            else:
                if previous_checkin_balance_after is None:
                    incomplete_records.append({
                        'phone': phone,
                        'date': run_date,
                        'reason': '前一天签到后余额缺失'
                    })
                    previous_checkin_balance_after = checkin_balance_after
                    continue
                previous_balance = previous_checkin_balance_after
            
            # 计算签到奖励
            new_checkin_reward = checkin_balance_after - previous_balance
            
            # 第一条记录可能累积多天，不限制
            # 非第一条记录：异常值设为0
            if idx > 0:
                if new_checkin_reward > 10 or new_checkin_reward < 0:
                    new_checkin_reward = 0.0
            
            # 更新
            if abs(new_checkin_reward - old_checkin_reward) > 0.001:
                try:
                    db.update_checkin_reward(record_id, new_checkin_reward)
                    updated_records += 1
                    print(f"[{phone}] [{run_date}] {old_checkin_reward:.2f} → {new_checkin_reward:.2f}")
                except Exception as e:
                    print(f"❌ [{phone}] [{run_date}] 更新失败: {e}")
            
            complete_records += 1
            previous_checkin_balance_after = checkin_balance_after
    
    # 输出结果
    print()
    print("=" * 80)
    print("处理结果:")
    print("=" * 80)
    print(f"完整记录: {complete_records}")
    print(f"不完整记录: {len(incomplete_records)}")
    print(f"已更新: {updated_records}")
    print()
    
    if incomplete_records:
        print("=" * 80)
        print("不完整记录详情:")
        print("=" * 80)
        for item in incomplete_records[:20]:  # 只显示前20条
            print(f"[{item['phone']}] [{item['date']}] {item['reason']}")
        if len(incomplete_records) > 20:
            print(f"... 还有 {len(incomplete_records) - 20} 条")


if __name__ == "__main__":
    try:
        simple_recalculate()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
