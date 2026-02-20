"""
测试单个账号的签到奖励计算

提取一个账号的所有记录，使用正确的计算方式展示每条记录的签到奖励
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


def test_account_calculation(phone: str):
    """测试单个账号的签到奖励计算
    
    Args:
        phone: 手机号
    """
    
    print("=" * 80)
    print(f"测试账号 {phone} 的签到奖励计算")
    print("=" * 80)
    print()
    
    # 初始化数据库
    db = LocalDatabase()
    
    # 获取该账号的所有记录
    all_records = db.get_all_history_records()
    account_records = [r for r in all_records if r.get('phone') == phone]
    
    if not account_records:
        print(f"❌ 未找到账号 {phone} 的记录")
        return
    
    # 按日期排序
    account_records.sort(key=lambda r: r.get('run_date', ''))
    
    print(f"共找到 {len(account_records)} 条记录")
    print()
    
    previous_balance_after = None
    
    for i, record in enumerate(account_records):
        run_date = record.get('run_date')
        balance_before = record.get('balance_before')
        checkin_balance_after = record.get('checkin_balance_after')
        balance_after = record.get('balance_after')
        transfer_amount = record.get('transfer_amount', 0.0) or 0.0
        current_reward = record.get('checkin_reward', 0.0) or 0.0
        
        print(f"[{i+1}] {run_date}")
        print(f"  签到前余额: {balance_before}")
        print(f"  签到后余额(转账前): {checkin_balance_after}")
        print(f"  最终余额(转账后): {balance_after}")
        print(f"  转账金额: {transfer_amount}")
        print(f"  当前签到奖励: {current_reward}")
        
        # 跳过没有余额数据的记录
        if balance_after is None:
            print(f"  ⚠️ 无余额数据，跳过")
            print()
            continue
        
        # 检查签到是否成功
        actual_checkin_balance = checkin_balance_after if checkin_balance_after is not None else balance_after
        
        if balance_before is not None and abs(actual_checkin_balance - balance_before) < 0.001:
            # 签到失败
            correct_reward = 0.0
            print(f"  ✓ 签到失败（余额没变化）")
            print(f"  → 正确的签到奖励: {correct_reward:.2f} 元")
        else:
            # 签到成功，计算奖励
            # 确定基准余额
            if previous_balance_after is not None:
                base_balance = previous_balance_after
            elif balance_before is not None:
                base_balance = balance_before
            else:
                print(f"  ⚠️ 无基准余额，跳过")
                print()
                previous_balance_after = balance_after
                continue
            
            # 使用 checkin_balance_after 计算（如果有）
            if checkin_balance_after is not None:
                correct_reward = checkin_balance_after - base_balance
                print(f"  ✓ 使用签到后余额计算")
                print(f"  → 基准余额: {base_balance:.2f} 元")
                print(f"  → 签到后余额: {checkin_balance_after:.2f} 元")
            else:
                # 使用最终余额计算
                balance_diff = balance_after - base_balance
                
                if balance_diff >= 0:
                    correct_reward = balance_diff
                    print(f"  ✓ 正常场景（差额为正）")
                else:
                    correct_reward = balance_diff + transfer_amount
                    print(f"  ✓ 转账场景（差额为负）")
                
                print(f"  → 基准余额: {base_balance:.2f} 元")
                print(f"  → 最终余额: {balance_after:.2f} 元")
                if transfer_amount > 0:
                    print(f"  → 转账金额: {transfer_amount:.2f} 元")
            
            # 检查计算结果是否合理
            if correct_reward > 10:
                print(f"  ⚠️ 计算结果大于10元 ({correct_reward:.2f})，应设为0")
                correct_reward = 0.0
            elif correct_reward < 0:
                print(f"  ⚠️ 计算结果为负值 ({correct_reward:.2f})，应设为0")
                correct_reward = 0.0
            
            print(f"  → 正确的签到奖励: {correct_reward:.2f} 元")
        
        # 对比当前值
        if abs(correct_reward - current_reward) > 0.001:
            print(f"  ❌ 当前值错误！应该是 {correct_reward:.2f} 元，实际是 {current_reward:.2f} 元")
        else:
            print(f"  ✅ 当前值正确")
        
        print()
        
        # 更新前一天余额
        previous_balance_after = balance_after


if __name__ == "__main__":
    # 选择一个有转账记录的账号进行测试
    test_phone = "13544171311"  # 这个账号有转账记录
    
    try:
        test_account_calculation(test_phone)
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
