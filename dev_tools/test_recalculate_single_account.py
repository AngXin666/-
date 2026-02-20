"""
测试重新计算单个账号的签到奖励

通过前后两天的余额差计算签到奖励
先测试一个账号,确认逻辑正确后再批量处理

运行方式:
    python dev_tools/test_recalculate_single_account.py
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


def test_recalculate_single_account(phone: str = None):
    """测试重新计算单个账号的签到奖励
    
    Args:
        phone: 手机号,如果为None则选择第一个有记录的账号
    """
    
    # 从转账配置读取最小转账金额(用于估算缺失的转账记录)
    MIN_TRANSFER_AMOUNT = 30.0  # 默认值
    try:
        import json
        transfer_config_path = project_root / "transfer_config.json"
        if transfer_config_path.exists():
            with open(transfer_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                MIN_TRANSFER_AMOUNT = config.get('min_transfer_amount', 30.0)
                print(f"从配置读取最小转账金额: {MIN_TRANSFER_AMOUNT} 元")
    except Exception as e:
        print(f"⚠️ 读取转账配置失败: {e}, 使用默认值 {MIN_TRANSFER_AMOUNT} 元")
    
    print("=" * 80)
    print("测试重新计算单个账号的签到奖励")
    print("=" * 80)
    print()
    
    # 初始化数据库
    db = LocalDatabase()
    
    # 获取所有记录
    all_records = db.get_all_history_records()
    
    if not all_records:
        print("❌ 数据库中没有记录")
        return
    
    print(f"数据库共有 {len(all_records)} 条记录")
    print()
    
    # 按账号分组
    records_by_phone = {}
    for record in all_records:
        p = record.get('phone')
        if p:
            if p not in records_by_phone:
                records_by_phone[p] = []
            records_by_phone[p].append(record)
    
    # 对每个账号的记录按日期排序
    for p in records_by_phone:
        records_by_phone[p].sort(key=lambda r: r.get('run_date', ''))
    
    print(f"共 {len(records_by_phone)} 个账号")
    print()
    
    # 如果没有指定手机号,选择第一个有多条记录的账号
    if phone is None:
        for p, records in records_by_phone.items():
            if len(records) >= 2:  # 至少要有2条记录才能计算余额差
                phone = p
                break
        
        if phone is None:
            print("❌ 没有找到有多条记录的账号")
            return
    
    # 检查账号是否存在
    if phone not in records_by_phone:
        print(f"❌ 账号 {phone} 不存在")
        return
    
    records = records_by_phone[phone]
    
    print(f"测试账号: {phone}")
    print(f"记录数量: {len(records)}")
    print("=" * 80)
    print()
    
    # 遍历该账号的所有记录,计算签到奖励
    previous_balance_after = None
    
    for i, record in enumerate(records):
        record_id = record.get('id')
        run_date = record.get('run_date')
        balance_before = record.get('balance_before')
        balance_after = record.get('balance_after')
        checkin_balance_after = record.get('checkin_balance_after')
        transfer_amount = record.get('transfer_amount', 0.0) or 0.0
        old_checkin_reward = record.get('checkin_reward', 0.0) or 0.0
        
        print(f"记录 {i+1}: {run_date}")
        print("-" * 80)
        
        # 跳过没有余额数据的记录
        if balance_after is None:
            print(f"  ⚠️ 跳过: 无余额数据")
            print()
            continue
        
        # 显示关键数据
        print(f"  数据:")
        if previous_balance_after is not None:
            print(f"    - 前天余额: {previous_balance_after:.2f}")
        else:
            print(f"    - 前天余额: None")
        
        if checkin_balance_after is not None:
            print(f"    - 签到后余额: {checkin_balance_after:.2f}")
        else:
            print(f"    - 签到后余额: None")
        
        if transfer_amount > 0:
            print(f"    - 转账金额: {transfer_amount:.2f}")
        print()
        
        # 计算新的签到奖励
        # 核心逻辑: 只使用前后两天的余额差,不考虑balance_before
        new_checkin_reward = None
        calculation_method = None
        
        # 如果没有前一天的余额,跳过第一条记录
        if previous_balance_after is None:
            print(f"  ⚠️ 跳过: 第一条记录,无前一天余额")
            print()
            previous_balance_after = balance_after
            continue
        
        # 使用前一天的最终余额作为基准
        base_balance = previous_balance_after
        
        # 计算签到奖励:只看余额变化,不考虑转账金额
        if checkin_balance_after is not None:
            # 有签到后余额(转账前的余额),直接计算 - 最准确
            new_checkin_reward = checkin_balance_after - base_balance
            calculation_method = "签到后余额 - 前天余额"
        else:
            # 没有签到后余额,使用最终余额计算
            balance_diff = balance_after - base_balance
            
            if balance_diff >= 0:
                # 差额为正:可以使用
                new_checkin_reward = balance_diff
                calculation_method = "最终余额 - 前天余额 (无签到后余额)"
            else:
                # 差额为负:可能有转账,但转账金额不可靠,无法准确计算
                print(f"  ⚠️ 无签到后余额且最终余额减少({balance_diff:.2f}),无法准确计算,设为0")
                new_checkin_reward = 0.0
                calculation_method = "无法准确计算(缺少签到后余额)"
        
        # 检查计算结果是否合理
        if new_checkin_reward is not None:
            if new_checkin_reward > 10:
                print(f"  ⚠️ 签到奖励大于10元 ({new_checkin_reward:.2f}), 设为0 (数据异常)")
                new_checkin_reward = 0.0
                calculation_method += " → 异常(>10元)"
            elif new_checkin_reward < 0:
                print(f"  ⚠️ 签到奖励为负值 ({new_checkin_reward:.2f}), 设为0 (数据异常)")
                new_checkin_reward = 0.0
                calculation_method += " → 异常(<0元)"
        
        # 显示计算结果
        print(f"  计算结果:")
        print(f"    - 计算方法: {calculation_method}")
        print(f"    - 签到奖励: {new_checkin_reward:.2f} 元")
        
        if abs(new_checkin_reward - old_checkin_reward) > 0.001:
            print(f"    - 需要更新: ✓")
        else:
            print(f"    - 需要更新: ✗")
        
        print()
        
        # 更新前一天余额
        previous_balance_after = balance_after
    
    print("=" * 80)
    print("测试完成")
    print("=" * 80)
    print()
    print("提示: 这只是测试计算,没有实际更新数据库")
    print("如果计算结果正确,可以运行批量更新脚本")
    print()


if __name__ == "__main__":
    try:
        # 可以指定手机号测试,或者留空自动选择
        test_phone = None  # 例如: "13044226531"
        
        test_recalculate_single_account(test_phone)
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
