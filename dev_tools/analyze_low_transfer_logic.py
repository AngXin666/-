"""
分析低于最小转账金额的转账记录逻辑

详细检查每条记录的计算逻辑是否合理

运行方式:
    python dev_tools/analyze_low_transfer_logic.py
"""

import sys
import os
from pathlib import Path
import json

# 设置标准输出编码为 UTF-8
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.local_db import LocalDatabase


def analyze_low_transfer_logic():
    """分析低于最小转账金额的转账记录逻辑"""
    
    # 读取转账配置
    MIN_TRANSFER_AMOUNT = 30.0
    try:
        transfer_config_path = project_root / "transfer_config.json"
        if transfer_config_path.exists():
            with open(transfer_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                MIN_TRANSFER_AMOUNT = config.get('min_transfer_amount', 30.0)
    except Exception as e:
        print(f"⚠️ 读取转账配置失败: {e}, 使用默认值 {MIN_TRANSFER_AMOUNT} 元")
    
    print("=" * 80)
    print("分析低于最小转账金额的转账记录逻辑")
    print("=" * 80)
    print(f"最小转账金额: {MIN_TRANSFER_AMOUNT} 元")
    print()
    
    # 初始化数据库
    db = LocalDatabase()
    all_records = db.get_all_history_records()
    
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
    
    # 查找低于最小转账金额的记录并详细分析
    suspicious_records = []
    
    for phone, records in records_by_phone.items():
        for idx, record in enumerate(records):
            transfer_amount = record.get('transfer_amount', 0.0) or 0.0
            
            # 只检查有转账且低于最小金额的记录
            if 0 < transfer_amount < MIN_TRANSFER_AMOUNT:
                # 获取前一天余额
                if idx > 0:
                    prev_record = records[idx - 1]
                    prev_balance = prev_record.get('checkin_balance_after')
                    if prev_balance is None:
                        prev_balance = prev_record.get('balance_after')
                else:
                    prev_balance = 0.0
                
                if prev_balance is None:
                    prev_balance = 0.0
                
                # 获取当前记录的数据
                old_checkin_reward = record.get('checkin_reward', 0.0) or 0.0
                balance_after = record.get('balance_after')
                checkin_balance_after = record.get('checkin_balance_after')
                
                # 计算：前一天余额 + 旧签到奖励
                total_before_transfer = prev_balance + old_checkin_reward
                
                # 如果前一天余额+旧签到奖励 >= 30，但转账金额 < 30，这是可疑的
                if total_before_transfer >= MIN_TRANSFER_AMOUNT:
                    suspicious_records.append({
                        'id': record.get('id'),
                        'phone': phone,
                        'date': record.get('run_date'),
                        'prev_balance': prev_balance,
                        'old_checkin_reward': old_checkin_reward,
                        'total_before_transfer': total_before_transfer,
                        'transfer_amount': transfer_amount,
                        'balance_after': balance_after,
                        'checkin_balance_after': checkin_balance_after,
                        'should_transfer': total_before_transfer >= MIN_TRANSFER_AMOUNT
                    })
    
    print(f"找到 {len(suspicious_records)} 条可疑记录（前一天余额+旧签到奖励 >= {MIN_TRANSFER_AMOUNT}，但转账 < {MIN_TRANSFER_AMOUNT}）")
    print()
    
    if not suspicious_records:
        print("✓ 没有找到可疑记录")
        return
    
    # 显示详情
    print("=" * 80)
    print("可疑记录详情:")
    print("=" * 80)
    
    for item in suspicious_records:
        print(f"\n账号: {item['phone']}, 日期: {item['date']}")
        print(f"  前一天余额: {item['prev_balance']:.2f}")
        print(f"  旧签到奖励: {item['old_checkin_reward']:.2f}")
        print(f"  前一天余额+旧签到奖励: {item['total_before_transfer']:.2f}")
        print(f"  实际转账金额: {item['transfer_amount']:.2f}")
        print(f"  转账后最终余额: {item['balance_after']:.2f}" if item['balance_after'] is not None else "  转账后最终余额: None")
        print(f"  签到后余额: {item['checkin_balance_after']:.2f}" if item['checkin_balance_after'] is not None else "  签到后余额: None")
        
        # 分析
        if item['total_before_transfer'] >= MIN_TRANSFER_AMOUNT:
            print(f"  ⚠️ 问题: 前一天余额+旧签到奖励 = {item['total_before_transfer']:.2f} >= {MIN_TRANSFER_AMOUNT}，应该转账 >= {MIN_TRANSFER_AMOUNT}")
            print(f"  ⚠️ 但实际只转账了 {item['transfer_amount']:.2f} 元")
            
            # 验证计算
            if item['balance_after'] is not None:
                calculated_transfer = item['prev_balance'] + item['old_checkin_reward'] - item['balance_after']
                print(f"  计算验证: {item['prev_balance']:.2f} + {item['old_checkin_reward']:.2f} - {item['balance_after']:.2f} = {calculated_transfer:.2f}")
                if abs(calculated_transfer - item['transfer_amount']) < 0.01:
                    print(f"  ✓ 转账金额计算正确")
                else:
                    print(f"  ❌ 转账金额计算错误，应该是 {calculated_transfer:.2f}")
    
    # 总结
    print()
    print("=" * 80)
    print("总结:")
    print("=" * 80)
    print(f"可疑记录数: {len(suspicious_records)}")
    print()
    print("这些记录的前一天余额+旧签到奖励 >= 30元，理论上应该转账 >= 30元")
    print("但实际转账金额 < 30元，可能存在以下情况：")
    print("1. 旧的签到奖励记录本身就是错误的（已经被重新计算过）")
    print("2. 转账逻辑在当时执行时使用的是重新计算后的签到奖励，而不是旧的")
    print("3. 需要检查这些记录的签到奖励是否已经被重新计算过")


if __name__ == "__main__":
    try:
        analyze_low_transfer_logic()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
