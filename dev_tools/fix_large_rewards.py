"""
修复签到奖励大于10的记录

签到奖励不可能大于10元，所有大于10的都是错误数据
将这些记录的签到奖励设置为0（表示无法准确计算）
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.local_db import LocalDatabase


def fix_large_rewards():
    """修复签到奖励大于10的记录"""
    
    print("=" * 80)
    print("修复签到奖励大于10的记录")
    print("=" * 80)
    print()
    
    # 初始化数据库
    db = LocalDatabase()
    
    # 获取所有记录
    all_records = db.get_all_history_records()
    
    # 找出大于10的奖励
    large_rewards = []
    for record in all_records:
        checkin_reward = record.get('checkin_reward', 0.0) or 0.0
        if checkin_reward > 10:
            large_rewards.append(record)
    
    print(f"共找到 {len(large_rewards)} 条签到奖励大于10的记录")
    print()
    
    if not large_rewards:
        print("✅ 没有找到签到奖励大于10的记录")
        return
    
    # 统计信息
    total_fixed = 0
    total_errors = 0
    
    # 修复每条记录
    for record in large_rewards:
        record_id = record.get('id')
        phone = record.get('phone')
        run_date = record.get('run_date')
        old_reward = record.get('checkin_reward', 0.0) or 0.0
        
        print(f"[{phone}] {run_date}")
        print(f"  旧签到奖励: {old_reward:.2f} 元")
        print(f"  新签到奖励: 0.00 元（无法准确计算）")
        
        try:
            result = db.update_checkin_reward(record_id, 0.0)
            if result:
                print(f"  ✓ 修复成功")
                total_fixed += 1
            else:
                print(f"  ❌ 修复失败（update返回False）")
                total_errors += 1
        except Exception as e:
            print(f"  ❌ 更新失败: {e}")
            total_errors += 1
        
        print()
    
    # 输出统计信息
    print("=" * 80)
    print("修复完成")
    print("=" * 80)
    print(f"总记录数: {len(large_rewards)}")
    print(f"已修复: {total_fixed}")
    print(f"错误: {total_errors}")
    print()


if __name__ == "__main__":
    try:
        fix_large_rewards()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
