"""
修复缺失的转账金额

根据余额变化推算转账金额

运行方式:
    python dev_tools/fix_missing_transfer_amount.py
"""

import sys
import os
from pathlib import Path

# 设置标准输出编码为 UTF-8
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.local_db import LocalDatabase


def fix_missing_transfer_amount():
    """修复缺失的转账金额"""
    
    # 从转账配置读取最小转账金额
    MIN_TRANSFER_AMOUNT = 30.0
    try:
        import json
        transfer_config_path = project_root / "transfer_config.json"
        if transfer_config_path.exists():
            with open(transfer_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                MIN_TRANSFER_AMOUNT = config.get('min_transfer_amount', 30.0)
    except Exception as e:
        print(f"⚠️ 读取转账配置失败: {e}, 使用默认值 {MIN_TRANSFER_AMOUNT} 元")
    
    print("=" * 80)
    print("修复缺失的转账金额")
    print("=" * 80)
    print(f"最小转账金额: {MIN_TRANSFER_AMOUNT} 元")
    print()
    print("根据余额变化推算转账金额:")
    print("  转账金额 = 前一天签到后余额 - 当天签到后余额 + 当天签到奖励")
    print()
    
    # 初始化数据库
    db = LocalDatabase()
    all_records = db.get_all_history_records()
    
    print(f"总记录数: {len(all_records)}")
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
    
    # 开始修复
    print("=" * 80)
    print("开始修复...")
    print("=" * 80)
    print()
    
    updated_count = 0
    error_count = 0
    
    for phone, records in records_by_phone.items():
        previous_checkin_balance_after = None
        
        for idx, record in enumerate(records):
            record_id = record.get('id')
            run_date = record.get('run_date')
            checkin_balance_after = record.get('checkin_balance_after')
            checkin_reward = record.get('checkin_reward', 0.0) or 0.0
            old_transfer_amount = record.get('transfer_amount', 0.0) or 0.0
            
            # 跳过没有签到后余额的记录
            if checkin_balance_after is None:
                previous_checkin_balance_after = None
                continue
            
            # 如果有前一天的签到后余额
            if previous_checkin_balance_after is not None:
                # 计算余额变化
                balance_change = checkin_balance_after - previous_checkin_balance_after
                
                # 如果余额下降超过20元（可能是转账）
                if balance_change < -20:
                    # 推算转账金额 = 前一天余额 - 当天余额（忽略签到奖励，因为可能被错误地设为0）
                    calculated_transfer = previous_checkin_balance_after - checkin_balance_after
                    
                    # 只要推算的转账金额 > 0 就更新（不要求 >= 最小转账金额，因为有些可能不足30元）
                    if calculated_transfer > 0:
                        try:
                            conn = db._get_connection()
                            cursor = conn.cursor()
                            cursor.execute('''
                                UPDATE history_records 
                                SET transfer_amount = ?
                                WHERE id = ?
                            ''', (calculated_transfer, record_id))
                            conn.commit()
                            conn.close()
                            
                            updated_count += 1
                            
                            # 显示前20条更新
                            if updated_count <= 20:
                                print(f"[{phone}] [{run_date}]")
                                print(f"  前一天签到后余额: {previous_checkin_balance_after:.2f}")
                                print(f"  当天签到后余额: {checkin_balance_after:.2f}")
                                print(f"  推算转账金额: {calculated_transfer:.2f}")
                                print(f"  转账金额: {old_transfer_amount:.2f} → {calculated_transfer:.2f} 元")
                                print()
                        
                        except Exception as e:
                            print(f"❌ [{phone}] [{run_date}] 更新失败: {e}")
                            error_count += 1
            
            # 更新前一天的签到后余额
            previous_checkin_balance_after = checkin_balance_after
    
    # 输出统计信息
    print()
    print("=" * 80)
    print("修复完成")
    print("=" * 80)
    print(f"已更新: {updated_count} 条")
    print(f"错误: {error_count} 条")
    print()
    
    # 验证修复结果
    print("=" * 80)
    print("验证修复结果...")
    print("=" * 80)
    print()
    
    # 重新分析
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
    
    # 找出余额突然大幅下降但没有转账记录的情况
    problem_cases = 0
    
    for phone, records in records_by_phone.items():
        previous_checkin_balance_after = None
        
        for record in records:
            checkin_balance_after = record.get('checkin_balance_after')
            transfer_amount = record.get('transfer_amount', 0.0) or 0.0
            
            if checkin_balance_after is None:
                previous_checkin_balance_after = None
                continue
            
            if previous_checkin_balance_after is not None:
                balance_change = checkin_balance_after - previous_checkin_balance_after
                
                # 余额下降超过20元但没有转账记录（不要求 >= 最小转账金额）
                if balance_change < -20 and transfer_amount == 0:
                    problem_cases += 1
            
            previous_checkin_balance_after = checkin_balance_after
    
    print(f"余额突然下降但没有转账记录: {problem_cases} 条")
    
    if problem_cases > 0:
        print("⚠️ 仍有部分记录的转账金额缺失")
    else:
        print("✓ 所有转账记录都已修复")
    
    print()


if __name__ == "__main__":
    try:
        fix_missing_transfer_amount()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
