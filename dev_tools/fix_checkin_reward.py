"""
修复历史记录中的签到奖励错误

问题：签到奖励 = 余额后 - 余额前，但数据库中的计算可能有误
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.local_db import LocalDatabase

def check_and_fix_checkin_rewards():
    """检查并修复签到奖励"""
    db = LocalDatabase()
    
    print("=" * 80)
    print("检查历史记录中的签到奖励计算")
    print("=" * 80)
    print()
    
    # 获取所有历史记录
    all_records = db.get_all_history_records()
    
    if not all_records:
        print("没有找到历史记录")
        return
    
    print(f"共找到 {len(all_records)} 条历史记录")
    print()
    
    # 检查每条记录
    error_count = 0
    fixed_count = 0
    
    for record in all_records:
        phone = record.get('phone', '')
        run_date = record.get('run_date', '')
        balance_before = record.get('balance_before', 0) or 0
        balance_after = record.get('balance_after', 0) or 0
        checkin_reward = record.get('checkin_reward', 0) or 0
        
        # 计算正确的签到奖励
        correct_reward = round(balance_after - balance_before, 2)
        
        # 检查是否有误差（允许0.01的浮点误差）
        if abs(checkin_reward - correct_reward) > 0.01:
            error_count += 1
            print(f"❌ 错误记录 #{error_count}")
            print(f"   手机号: {phone}")
            print(f"   日期: {run_date}")
            print(f"   余额前: {balance_before}")
            print(f"   余额后: {balance_after}")
            print(f"   记录的签到奖励: {checkin_reward}")
            print(f"   正确的签到奖励: {correct_reward}")
            print(f"   差异: {checkin_reward - correct_reward}")
            print()
            
            # 修复记录
            update_record = {
                'phone': phone,
                'run_date': run_date,
                'checkin_reward': correct_reward
            }
            
            if db.upsert_history_record(update_record):
                fixed_count += 1
                print(f"   ✓ 已修复")
            else:
                print(f"   ✗ 修复失败")
            print()
    
    print("=" * 80)
    print(f"检查完成")
    print(f"错误记录数: {error_count}")
    print(f"已修复记录数: {fixed_count}")
    print("=" * 80)

if __name__ == '__main__':
    check_and_fix_checkin_rewards()
