"""
修复数据库中的签到奖励数据 - 使用转账金额修正逻辑

使用新的计算逻辑重新计算所有记录的签到奖励：
- 如果签到后余额 < 前一天余额 且 前一天余额 > 0，则加上转账金额
- 否则正常计算

运行方式：
    python dev_tools/fix_checkin_reward_with_transfer.py
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.local_db import LocalDatabase


def fix_checkin_rewards():
    """修复数据库中的签到奖励数据"""
    
    print("=" * 80)
    print("修复数据库中的签到奖励数据")
    print("=" * 80)
    print()
    
    # 初始化数据库
    db = LocalDatabase()
    
    # 获取所有记录，按手机号和日期排序
    print("正在读取数据库记录...")
    all_records = db.get_all_history_records()
    
    if not all_records:
        print("❌ 数据库中没有记录")
        return
    
    print(f"✓ 共找到 {len(all_records)} 条记录")
    print()
    
    # 按手机号分组
    records_by_phone = {}
    for record in all_records:
        phone = record.get('phone')
        if phone:
            if phone not in records_by_phone:
                records_by_phone[phone] = []
            records_by_phone[phone].append(record)
    
    # 对每个手机号的记录按日期排序
    for phone in records_by_phone:
        records_by_phone[phone].sort(key=lambda r: r.get('run_date', ''))
    
    print(f"✓ 共 {len(records_by_phone)} 个账号")
    print()
    
    # 统计信息
    total_fixed = 0
    total_unchanged = 0
    total_errors = 0
    
    # 遍历每个账号的记录
    for phone, records in records_by_phone.items():
        print(f"处理账号: {phone}")
        print("-" * 80)
        
        previous_balance_after = None  # 前一天的最终余额
        
        for i, record in enumerate(records):
            record_id = record.get('id')
            run_date = record.get('run_date')
            balance_before = record.get('balance_before')
            balance_after = record.get('balance_after')
            transfer_amount = record.get('transfer_amount', 0.0) or 0.0
            old_checkin_reward = record.get('checkin_reward', 0.0) or 0.0
            
            # 跳过没有余额数据的记录
            if balance_after is None:
                print(f"  [{run_date}] 跳过：无余额数据")
                total_unchanged += 1
                continue
            
            # 首先检查签到是否成功（余额是否变化）
            # 这个判断必须放在最前面，因为签到失败时不需要考虑基准余额
            if balance_before is not None and abs(balance_after - balance_before) < 0.001:
                # 签到失败：余额没有变化，签到奖励应该是 0
                new_checkin_reward = 0.0
                scenario = "签到失败"
                
                # 检查是否需要更新
                if abs(new_checkin_reward - old_checkin_reward) > 0.001:
                    try:
                        db.update_checkin_reward(record_id, new_checkin_reward)
                        print(f"  [{run_date}] ✓ 修复 ({scenario})")
                        print(f"    - 签到前余额: {balance_before:.2f} 元")
                        print(f"    - 签到后余额: {balance_after:.2f} 元")
                        print(f"    - 旧签到奖励: {old_checkin_reward:.2f} 元")
                        print(f"    - 新签到奖励: {new_checkin_reward:.2f} 元")
                        total_fixed += 1
                    except Exception as e:
                        print(f"  [{run_date}] ❌ 更新失败: {e}")
                        total_errors += 1
                else:
                    print(f"  [{run_date}] - 无需修复 (奖励: {old_checkin_reward:.2f} 元)")
                    total_unchanged += 1
                
                # 更新前一天余额
                previous_balance_after = balance_after
                continue
            
            # 确定基准余额（用于签到成功的情况）
            if previous_balance_after is not None:
                base_balance = previous_balance_after
            elif balance_before is not None:
                base_balance = balance_before
            else:
                print(f"  [{run_date}] 跳过：无基准余额")
                total_unchanged += 1
                previous_balance_after = balance_after
                continue
            
            # 计算前后两天的余额差额
            balance_diff = balance_after - base_balance
            
            # 根据差额判断场景
            if balance_diff >= 0:
                # 差额为正：正常签到场景
                new_checkin_reward = balance_diff
                scenario = "正常场景"
            else:
                # 差额为负：可能是转账场景
                # 加上转账金额后再计算
                new_checkin_reward = balance_diff + transfer_amount
                
                if transfer_amount > 0:
                    scenario = "转账场景"
                else:
                    scenario = "余额减少（无转账记录）"
            
            # 检查计算结果是否合理（签到奖励不应该大于10元）
            if new_checkin_reward > 10:
                # 奖励大于10，无法准确计算，设为0
                print(f"  [{run_date}] ⚠️ 计算结果大于10元 ({new_checkin_reward:.2f})，设为0")
                new_checkin_reward = 0.0
                scenario = "无法准确计算"
            elif new_checkin_reward < 0:
                # 奖励为负值，数据异常，设为0
                print(f"  [{run_date}] ⚠️ 计算结果为负值 ({new_checkin_reward:.2f})，设为0")
                new_checkin_reward = 0.0
                scenario = "数据异常"
            
            # 检查是否需要更新
            if abs(new_checkin_reward - old_checkin_reward) > 0.001:  # 允许0.001的误差
                # 需要更新
                try:
                    db.update_checkin_reward(record_id, new_checkin_reward)
                    print(f"  [{run_date}] ✓ 修复 ({scenario})")
                    print(f"    - 基准余额: {base_balance:.2f} 元")
                    print(f"    - 签到后余额: {balance_after:.2f} 元")
                    if transfer_amount > 0:
                        print(f"    - 转账金额: {transfer_amount:.2f} 元")
                    print(f"    - 旧签到奖励: {old_checkin_reward:.2f} 元")
                    print(f"    - 新签到奖励: {new_checkin_reward:.2f} 元")
                    total_fixed += 1
                except Exception as e:
                    print(f"  [{run_date}] ❌ 更新失败: {e}")
                    total_errors += 1
            else:
                # 不需要更新
                print(f"  [{run_date}] - 无需修复 (奖励: {old_checkin_reward:.2f} 元)")
                total_unchanged += 1
            
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
        fix_checkin_rewards()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
