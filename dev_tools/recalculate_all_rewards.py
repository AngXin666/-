"""
重新计算所有签到奖励

从头开始，使用正确的逻辑重新计算所有记录的签到奖励
"""

import sys
import os
from pathlib import Path

# 设置标准输出编码为 UTF-8（解决 Windows CMD 乱码问题）
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.local_db import LocalDatabase


def recalculate_all_rewards():
    """重新计算所有签到奖励"""
    
    print("=" * 80)
    print("重新计算所有签到奖励")
    print("=" * 80)
    print()
    
    # 初始化数据库
    db = LocalDatabase()
    
    # 获取所有记录
    all_records = db.get_all_history_records()
    
    print(f"共找到 {len(all_records)} 条记录")
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
    total_fixed = 0
    total_unchanged = 0
    total_errors = 0
    
    # 遍历每个账号的记录
    for phone, records in records_by_phone.items():
        print(f"处理账号: {phone}")
        print("-" * 80)
        
        previous_balance_after = None
        
        for record in records:
            record_id = record.get('id')
            run_date = record.get('run_date')
            balance_before = record.get('balance_before')
            checkin_balance_after = record.get('checkin_balance_after')  # 签到后余额（转账前）
            balance_after = record.get('balance_after')  # 最终余额（转账后）
            transfer_amount = record.get('transfer_amount', 0.0) or 0.0
            old_checkin_reward = record.get('checkin_reward', 0.0) or 0.0
            
            # 跳过没有余额数据的记录
            if balance_after is None:
                print(f"  [{run_date}] 跳过：无余额数据")
                total_unchanged += 1
                continue
            
            # 首先检查签到是否成功（余额是否变化）
            # 使用 checkin_balance_after（如果有）或 balance_after 来判断
            actual_checkin_balance = checkin_balance_after if checkin_balance_after is not None else balance_after
            
            if balance_before is not None and abs(actual_checkin_balance - balance_before) < 0.001:
                # 签到失败：余额没有变化
                new_checkin_reward = 0.0
                scenario = "签到失败"
                
                if abs(new_checkin_reward - old_checkin_reward) > 0.001:
                    try:
                        db.update_checkin_reward(record_id, new_checkin_reward)
                        print(f"  [{run_date}] ✓ 修复 ({scenario})")
                        print(f"    - 旧签到奖励: {old_checkin_reward:.2f} 元")
                        print(f"    - 新签到奖励: {new_checkin_reward:.2f} 元")
                        total_fixed += 1
                    except Exception as e:
                        print(f"  [{run_date}] ❌ 更新失败: {e}")
                        total_errors += 1
                else:
                    # 不输出"无需修复"信息，减少噪音
                    total_unchanged += 1
                
                previous_balance_after = balance_after
                continue
            
            # 确定基准余额
            if previous_balance_after is not None:
                base_balance = previous_balance_after
            elif balance_before is not None:
                base_balance = balance_before
            else:
                print(f"  [{run_date}] 跳过：无基准余额")
                total_unchanged += 1
                previous_balance_after = balance_after
                continue
            
            # 计算签到奖励：使用 checkin_balance_after（如果有）
            if checkin_balance_after is not None:
                # 有签到后余额数据，直接计算
                new_checkin_reward = checkin_balance_after - base_balance
                scenario = "使用签到后余额"
            else:
                # 没有签到后余额数据，使用最终余额计算
                balance_diff = balance_after - base_balance
                
                if balance_diff >= 0:
                    # 差额为正：正常签到
                    new_checkin_reward = balance_diff
                    scenario = "正常场景"
                else:
                    # 差额为负：转账场景
                    new_checkin_reward = balance_diff + transfer_amount
                    scenario = "转账场景"
            
            # 检查计算结果是否合理
            if new_checkin_reward > 10:
                # 奖励大于10，设为0
                print(f"  [{run_date}] ⚠️ 计算结果大于10元 ({new_checkin_reward:.2f})，设为0")
                new_checkin_reward = 0.0
                scenario = "无法准确计算"
            elif new_checkin_reward < 0:
                # 奖励为负值，设为0
                print(f"  [{run_date}] ⚠️ 计算结果为负值 ({new_checkin_reward:.2f})，设为0")
                new_checkin_reward = 0.0
                scenario = "数据异常"
            
            # 强制更新所有记录（因为之前的逻辑可能是错误的）
            try:
                db.update_checkin_reward(record_id, new_checkin_reward)
                
                # 只在奖励值发生变化时输出详细信息
                if abs(new_checkin_reward - old_checkin_reward) > 0.001:
                    print(f"  [{run_date}] ✓ 修复 ({scenario})")
                    print(f"    - 基准余额: {base_balance:.2f} 元")
                    if checkin_balance_after is not None:
                        print(f"    - 签到后余额: {checkin_balance_after:.2f} 元")
                    else:
                        print(f"    - 最终余额: {balance_after:.2f} 元")
                    if transfer_amount > 0:
                        print(f"    - 转账金额: {transfer_amount:.2f} 元")
                    print(f"    - 旧签到奖励: {old_checkin_reward:.2f} 元")
                    print(f"    - 新签到奖励: {new_checkin_reward:.2f} 元")
                
                total_fixed += 1
            except Exception as e:
                print(f"  [{run_date}] ❌ 更新失败: {e}")
                total_errors += 1
            
            # 更新前一天余额
            previous_balance_after = balance_after
        
        print()
    
    # 输出统计信息
    print("=" * 80)
    print("修复完成")
    print("=" * 80)
    print(f"总记录数: {len(all_records)}")
    print(f"已修复: {total_fixed}")
    print(f"无需修复: {total_unchanged}")
    print(f"错误: {total_errors}")
    print()


if __name__ == "__main__":
    try:
        recalculate_all_rewards()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
