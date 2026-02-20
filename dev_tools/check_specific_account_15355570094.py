"""
检查特定账号的所有记录
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.local_db import LocalDatabase


def check_account():
    """检查特定账号的所有记录"""
    
    phone = "15355570094"
    
    print("=" * 80)
    print(f"检查账号 {phone} 的所有记录")
    print("=" * 80)
    print()
    
    # 初始化数据库
    db = LocalDatabase()
    
    # 获取所有记录
    all_records = db.get_all_history_records()
    
    # 筛选该账号的记录
    account_records = [r for r in all_records if r.get('phone') == phone]
    account_records.sort(key=lambda r: r.get('run_date', ''))
    
    print(f"共找到 {len(account_records)} 条记录")
    print()
    
    previous_balance_after = None
    
    for record in account_records:
        run_date = record.get('run_date')
        balance_before = record.get('balance_before')
        balance_after = record.get('balance_after')
        checkin_reward = record.get('checkin_reward', 0.0) or 0.0
        transfer_amount = record.get('transfer_amount', 0.0) or 0.0
        
        print(f"日期: {run_date}")
        print(f"  签到前余额: {balance_before}")
        print(f"  签到后余额: {balance_after}")
        print(f"  签到奖励: {checkin_reward}")
        print(f"  转账金额: {transfer_amount}")
        print(f"  前一天余额: {previous_balance_after}")
        
        # 判断应该使用的基准余额
        if previous_balance_after is not None:
            base_balance = previous_balance_after
        elif balance_before is not None:
            base_balance = balance_before
        else:
            base_balance = None
        
        print(f"  基准余额: {base_balance}")
        
        # 判断签到是否成功
        if balance_before is not None and balance_after is not None:
            diff = abs(balance_after - balance_before)
            if diff < 0.001:
                print(f"  ✓ 签到失败（余额没变化，差异: {diff:.10f}）")
                expected_reward = 0.0
            elif balance_after < base_balance and base_balance > 0:
                print(f"  ✓ 转账场景")
                expected_reward = (balance_after + transfer_amount) - base_balance
            else:
                print(f"  ✓ 正常场景")
                expected_reward = balance_after - base_balance
            
            print(f"  预期奖励: {expected_reward}")
            
            if abs(expected_reward - checkin_reward) > 0.001:
                print(f"  ⚠️ 奖励不匹配！")
        
        print()
        
        if balance_after is not None:
            previous_balance_after = balance_after


if __name__ == "__main__":
    try:
        check_account()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
