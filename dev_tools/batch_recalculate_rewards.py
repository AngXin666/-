"""
批量重新计算所有账号的签到奖励

通过前后两天的余额差计算签到奖励
使用正确的逻辑重新计算所有记录

运行方式:
    python dev_tools/batch_recalculate_rewards.py
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


def batch_recalculate_rewards():
    """批量重新计算所有账号的签到奖励"""
    
    # 从转账配置读取最小转账金额(用于参考)
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
    print("批量重新计算所有账号的签到奖励")
    print("=" * 80)
    print()
    
    # 创建日志文件
    from datetime import datetime
    log_filename = f"dev_tools/recalculate_rewards_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_file = open(log_filename, 'w', encoding='utf-8')
    
    def log(msg):
        """同时输出到控制台和文件"""
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()
    
    log(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"最小转账金额: {MIN_TRANSFER_AMOUNT} 元")
    log("=" * 80)
    log("")
    
    # 初始化数据库
    db = LocalDatabase()
    
    # 获取所有记录
    all_records = db.get_all_history_records()
    
    if not all_records:
        log("❌ 数据库中没有记录")
        log_file.close()
        return
    
    log(f"数据库共有 {len(all_records)} 条记录")
    log("")
    
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
    
    log(f"共 {len(records_by_phone)} 个账号")
    log("")
    
    # 统计信息
    total_updated = 0
    total_unchanged = 0
    total_skipped = 0
    total_errors = 0
    
    # 遍历每个账号的记录
    for phone_idx, (phone, records) in enumerate(records_by_phone.items(), 1):
        log(f"[{phone_idx}/{len(records_by_phone)}] 处理账号: {phone} ({len(records)}条记录)")
        
        previous_checkin_balance_after = None
        account_updated = 0
        
        for record in records:
            record_id = record.get('id')
            run_date = record.get('run_date')
            balance_after = record.get('balance_after')
            checkin_balance_after = record.get('checkin_balance_after')
            old_checkin_reward = record.get('checkin_reward', 0.0) or 0.0
            
            # 没有余额数据的记录,设置签到奖励为0
            if balance_after is None:
                if abs(old_checkin_reward - 0.0) > 0.001:
                    try:
                        db.update_checkin_reward(record_id, 0.0)
                        total_updated += 1
                        account_updated += 1
                        log(f"  [{run_date}] 更新: {old_checkin_reward:.2f} → 0.00 元 (无余额数据)")
                    except Exception as e:
                        log(f"  ❌ [{run_date}] 更新失败: {e}")
                        total_errors += 1
                else:
                    total_unchanged += 1
                total_skipped += 1
                continue
            
            # 使用前一天的签到后余额作为基准,第一条记录默认为0
            base_balance = previous_checkin_balance_after if previous_checkin_balance_after is not None else 0.0
            
            # 计算签到奖励
            if checkin_balance_after is not None:
                # 有签到后余额,直接计算 - 最准确
                new_checkin_reward = checkin_balance_after - base_balance
                method = "签到后余额 - 前天余额"
            else:
                # 没有签到后余额,使用最终余额计算
                balance_diff = balance_after - base_balance
                
                if balance_diff >= 0:
                    # 差额为正:可以使用
                    new_checkin_reward = balance_diff
                    method = "最终余额 - 前天余额"
                else:
                    # 差额为负:无法准确计算
                    new_checkin_reward = 0.0
                    method = "无法准确计算(余额减少)"
            
            # 检查计算结果是否合理
            # 第一条记录可能累积了多天签到，超过10元是正常的，不应用限制
            is_first_record = (previous_checkin_balance_after is None or previous_checkin_balance_after == 0.0)
            
            if not is_first_record and new_checkin_reward > 10:
                new_checkin_reward = 0.0
                method += " → 异常(>10元)"
            elif new_checkin_reward < 0:
                new_checkin_reward = 0.0
                method += " → 异常(<0元)"
            
            # 检查是否需要更新
            if abs(new_checkin_reward - old_checkin_reward) > 0.001:
                try:
                    db.update_checkin_reward(record_id, new_checkin_reward)
                    total_updated += 1
                    account_updated += 1
                    
                    # 记录详细信息
                    log(f"  [{run_date}] 更新: {old_checkin_reward:.2f} → {new_checkin_reward:.2f} 元")
                    log(f"    前天余额: {base_balance:.2f}, 签到后余额: {checkin_balance_after if checkin_balance_after is not None else '无'}")
                    log(f"    计算方法: {method}")
                except Exception as e:
                    log(f"  ❌ [{run_date}] 更新失败: {e}")
                    total_errors += 1
            else:
                total_unchanged += 1
            
            # 更新前一天的签到后余额（用于下一条记录计算）
            # 优先使用签到后余额，如果没有则使用最终余额
            if checkin_balance_after is not None:
                previous_checkin_balance_after = checkin_balance_after
            else:
                previous_checkin_balance_after = balance_after
        
        if account_updated > 0:
            log(f"  ✓ 更新了 {account_updated} 条记录")
        log("")
    
    # 输出统计信息
    log("")
    log("=" * 80)
    log("批量更新完成")
    log("=" * 80)
    log(f"总记录数: {len(all_records)}")
    log(f"已更新: {total_updated}")
    log(f"无需更新: {total_unchanged}")
    log(f"跳过: {total_skipped}")
    log(f"错误: {total_errors}")
    log("")
    log(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"日志文件: {log_filename}")
    
    log_file.close()
    print(f"\n详细日志已保存到: {log_filename}")


if __name__ == "__main__":
    try:
        batch_recalculate_rewards()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
