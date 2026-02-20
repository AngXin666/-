"""
最终版本：重新计算签到奖励

按照简单逻辑处理：
1. 优先使用签到后余额计算
2. 没有就用最终余额计算
3. 负值检查转账
4. 都没有就设为0

运行方式:
    python dev_tools/final_recalculate.py
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


def final_recalculate():
    """最终版本：重新计算签到奖励"""
    
    print("=" * 80)
    print("重新计算签到奖励 - 最终版本")
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
    updated_count = 0
    
    # 遍历每个账号
    for phone, records in records_by_phone.items():
        for idx, record in enumerate(records):
            record_id = record.get('id')
            run_date = record.get('run_date')
            old_checkin_reward = record.get('checkin_reward', 0.0) or 0.0
            checkin_balance_after = record.get('checkin_balance_after')
            balance_after = record.get('balance_after')
            transfer_amount = record.get('transfer_amount', 0.0) or 0.0
            
            # 获取前一天的余额
            if idx == 0:
                # 第一条记录，前一天余额默认为0
                prev_balance = 0.0
            else:
                prev_record = records[idx - 1]
                # 优先使用前一天的签到后余额
                prev_balance = prev_record.get('checkin_balance_after')
                if prev_balance is None:
                    # 没有就用最终余额
                    prev_balance = prev_record.get('balance_after')
                if prev_balance is None:
                    # 都没有，设为0
                    prev_balance = 0.0
            
            # 计算签到奖励
            new_checkin_reward = None
            
            # 1. 优先使用签到后余额
            if checkin_balance_after is not None:
                new_checkin_reward = checkin_balance_after - prev_balance
            
            # 2. 没有签到后余额，用最终余额
            elif balance_after is not None:
                balance_diff = balance_after - prev_balance
                
                # 3. 如果是负值，检查转账
                if balance_diff < 0:
                    # 加上转账金额看看
                    if transfer_amount > 0:
                        new_checkin_reward = balance_diff + transfer_amount
                    else:
                        new_checkin_reward = 0.0
                else:
                    new_checkin_reward = balance_diff
            
            # 4. 都没有，设为0
            else:
                new_checkin_reward = 0.0
            
            # 第一条记录可能累积多天，不限制
            # 非第一条记录：异常值设为0
            if idx > 0:
                if new_checkin_reward > 10 or new_checkin_reward < 0:
                    new_checkin_reward = 0.0
            
            # 更新
            if abs(new_checkin_reward - old_checkin_reward) > 0.001:
                try:
                    db.update_checkin_reward(record_id, new_checkin_reward)
                    updated_count += 1
                    print(f"[{phone}] [{run_date}] {old_checkin_reward:.2f} → {new_checkin_reward:.2f}")
                except Exception as e:
                    print(f"❌ [{phone}] [{run_date}] 更新失败: {e}")
    
    # 输出结果
    print()
    print("=" * 80)
    print("处理完成")
    print("=" * 80)
    print(f"总记录数: {len(all_records)}")
    print(f"已更新: {updated_count}")
    print()


if __name__ == "__main__":
    try:
        final_recalculate()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
