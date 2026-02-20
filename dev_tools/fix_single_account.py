"""
修复单个账号的签到奖励数据
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.local_db import LocalDatabase


def fix_single_account():
    """修复单个账号的签到奖励数据"""
    
    target_phone = "15355570094"
    
    print("=" * 80)
    print(f"修复账号 {target_phone} 的签到奖励数据")
    print("=" * 80)
    print()
    
    # 初始化数据库
    db = LocalDatabase()
    
    # 获取所有记录
    all_records = db.get_all_history_records()
    
    # 筛选该账号的记录
    records = [r for r in all_records if r.get('phone') == target_phone]
    records.sort(key=lambda r: r.get('run_date', ''))
    
    print(f"共找到 {len(records)} 条记录")
    print()
    
    # 统计信息
    total_fixed = 0
    total_unchanged = 0
    total_errors = 0
    
    previous_balance_after = None
    
    for record in records:
        record_id = record.get('id')
        run_date = record.get('run_date')
        balance_before = record.get('balance_before')
        balance_after = record.get('balance_after')
        transfer_amount = record.get('transfer_amount', 0.0) or 0.0
        old_checkin_reward = record.get('checkin_reward', 0.0) or 0.0
        
        print(f"[{run_date}] 记录ID: {record_id}")
        print(f"  签到前余额: {balance_before}")
        print(f"  签到后余额: {balance_after}")
        print(f"  旧签到奖励: {old_checkin_reward}")
        
        # 跳过没有余额数据的记录
        if balance_after is None:
            print(f"  跳过：无余额数据")
            total_unchanged += 1
            print()
            continue
        
        # 首先检查签到是否成功（余额是否变化）
        if balance_before is not None and abs(balance_after - balance_before) < 0.001:
            # 签到失败：余额没有变化，签到奖励应该是 0
            new_checkin_reward = 0.0
            scenario = "签到失败"
            
            print(f"  判断：{scenario}")
            print(f"  新签到奖励: {new_checkin_reward}")
            
            # 检查是否需要更新
            if abs(new_checkin_reward - old_checkin_reward) > 0.001:
                print(f"  需要更新")
                try:
                    result = db.update_checkin_reward(record_id, new_checkin_reward)
                    if result:
                        print(f"  ✓ 修复成功")
                        total_fixed += 1
                    else:
                        print(f"  ❌ 修复失败（update返回False）")
                        total_errors += 1
                except Exception as e:
                    print(f"  ❌ 更新失败: {e}")
                    total_errors += 1
            else:
                print(f"  无需修复")
                total_unchanged += 1
            
            # 更新前一天余额
            previous_balance_after = balance_after
            print()
            continue
        
        # 确定基准余额（用于签到成功的情况）
        if previous_balance_after is not None:
            base_balance = previous_balance_after
        elif balance_before is not None:
            base_balance = balance_before
        else:
            print(f"  跳过：无基准余额")
            total_unchanged += 1
            previous_balance_after = balance_after
            print()
            continue
        
        # 使用新逻辑计算签到奖励（签到成功的情况）
        if balance_after < base_balance and base_balance > 0:
            # 转账场景
            new_checkin_reward = (balance_after + transfer_amount) - base_balance
            scenario = "转账场景"
        else:
            # 正常场景
            new_checkin_reward = balance_after - base_balance
            scenario = "正常场景"
        
        print(f"  判断：{scenario}")
        print(f"  基准余额: {base_balance}")
        print(f"  新签到奖励: {new_checkin_reward}")
        
        # 检查是否需要更新
        if abs(new_checkin_reward - old_checkin_reward) > 0.001:
            print(f"  需要更新")
            try:
                result = db.update_checkin_reward(record_id, new_checkin_reward)
                if result:
                    print(f"  ✓ 修复成功")
                    total_fixed += 1
                else:
                    print(f"  ❌ 修复失败（update返回False）")
                    total_errors += 1
            except Exception as e:
                print(f"  ❌ 更新失败: {e}")
                total_errors += 1
        else:
            print(f"  无需修复")
            total_unchanged += 1
        
        # 更新前一天余额
        previous_balance_after = balance_after
        print()
    
    # 输出统计信息
    print("=" * 80)
    print("修复完成")
    print("=" * 80)
    print(f"总记录数: {len(records)}")
    print(f"已修复: {total_fixed}")
    print(f"无需修复: {total_unchanged}")
    print(f"错误: {total_errors}")
    print()


if __name__ == "__main__":
    try:
        fix_single_account()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
