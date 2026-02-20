"""
检查低于最小转账金额的转账记录

检查转账金额 < 30元的记录，看是否因为旧的签到奖励导致
正确的转账金额 = 前天余额 + 旧签到奖励 - 最终余额

运行方式:
    python dev_tools/check_low_transfer.py
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


def check_low_transfer():
    """检查低于最小转账金额的转账记录"""
    
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
    print("检查低于最小转账金额的转账记录")
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
    
    # 查找低于最小转账金额的记录
    low_transfer_records = []
    
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
                
                if balance_after is not None:
                    # 计算正确的转账金额
                    # 转账金额 = 前天余额 + 旧签到奖励 - 最终余额
                    correct_transfer = prev_balance + old_checkin_reward - balance_after
                    
                    low_transfer_records.append({
                        'id': record.get('id'),
                        'phone': phone,
                        'date': record.get('run_date'),
                        'prev_balance': prev_balance,
                        'old_checkin_reward': old_checkin_reward,
                        'balance_after': balance_after,
                        'current_transfer': transfer_amount,
                        'correct_transfer': correct_transfer,
                        'difference': correct_transfer - transfer_amount
                    })
    
    print(f"找到 {len(low_transfer_records)} 条低于最小转账金额的记录")
    print()
    
    if not low_transfer_records:
        print("✓ 没有找到低于最小转账金额的记录")
        return
    
    # 显示详情
    print("=" * 80)
    print("详细信息:")
    print("=" * 80)
    
    need_fix = []
    
    for item in low_transfer_records:
        print(f"\n账号: {item['phone']}, 日期: {item['date']}")
        print(f"  前天余额: {item['prev_balance']:.2f}")
        print(f"  旧签到奖励: {item['old_checkin_reward']:.2f}")
        print(f"  最终余额: {item['balance_after']:.2f}")
        print(f"  当前转账金额: {item['current_transfer']:.2f}")
        print(f"  正确转账金额: {item['correct_transfer']:.2f}")
        print(f"  差异: {item['difference']:.2f}")
        
        # 如果差异超过0.01元，需要修复
        if abs(item['difference']) > 0.01:
            print(f"  ⚠️ 需要修复")
            need_fix.append(item)
        else:
            print(f"  ✓ 金额正确")
    
    # 总结
    print()
    print("=" * 80)
    print("总结:")
    print("=" * 80)
    print(f"低于最小转账金额的记录: {len(low_transfer_records)}")
    print(f"需要修复的记录: {len(need_fix)}")
    print()
    
    if need_fix:
        print("建议: 运行修复脚本更新这些记录的转账金额")
    
    return need_fix


if __name__ == "__main__":
    try:
        check_low_transfer()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
