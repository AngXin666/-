"""
正确修复低于最小转账金额的转账记录

逻辑：
1. 如果前一天余额 + 签到奖励 >= 30元，应该全部转走
2. 转账金额 = checkin_balance_after（签到后余额）
3. 转账后最终余额 = 0

运行方式:
    python dev_tools/fix_low_transfer_correct.py
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


def fix_low_transfer_correct():
    """正确修复低于最小转账金额的转账记录"""
    
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
    print("正确修复低于最小转账金额的转账记录")
    print("=" * 80)
    print(f"最小转账金额: {MIN_TRANSFER_AMOUNT} 元")
    print()
    print("修复逻辑:")
    print("1. 保留余额功能未启动，转账会全部转走")
    print("2. 如果签到后余额 >= 30元，应该全部转走")
    print("3. 转账金额 = 签到后余额")
    print("4. 转账后最终余额 = 0")
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
    
    # 查找需要修复的记录
    need_fix = []
    
    for phone, records in records_by_phone.items():
        for idx, record in enumerate(records):
            transfer_amount = record.get('transfer_amount', 0.0) or 0.0
            
            # 检查所有有转账的记录
            if transfer_amount > 0:
                checkin_balance_after = record.get('checkin_balance_after')
                balance_after = record.get('balance_after')
                
                # 如果签到后余额 >= 30，说明应该全部转走
                if checkin_balance_after is not None and checkin_balance_after >= MIN_TRANSFER_AMOUNT:
                    # 检查是否需要修复
                    # 1. 转账金额不等于签到后余额
                    # 2. 转账后最终余额不为0
                    if (abs(transfer_amount - checkin_balance_after) > 0.01 or 
                        (balance_after is not None and abs(balance_after) > 0.01)):
                        need_fix.append({
                            'id': record.get('id'),
                            'phone': phone,
                            'date': record.get('run_date'),
                            'checkin_balance_after': checkin_balance_after,
                            'current_transfer': transfer_amount,
                            'correct_transfer': checkin_balance_after,
                            'current_balance_after': balance_after,
                            'correct_balance_after': 0.0
                        })
    
    print(f"找到 {len(need_fix)} 条需要修复的记录")
    print()
    
    if not need_fix:
        print("✓ 没有需要修复的记录")
        return
    
    # 显示前10条详情
    print("=" * 80)
    print(f"前10条记录详情:")
    print("=" * 80)
    
    for item in need_fix[:10]:
        print(f"\n账号: {item['phone']}, 日期: {item['date']}")
        print(f"  签到后余额: {item['checkin_balance_after']:.2f}")
        print(f"  当前转账金额: {item['current_transfer']:.2f} → 正确: {item['correct_transfer']:.2f}")
        print(f"  当前最终余额: {item['current_balance_after']:.2f} → 正确: {item['correct_balance_after']:.2f}")
    
    if len(need_fix) > 10:
        print(f"\n... 还有 {len(need_fix) - 10} 条记录")
    
    # 开始修复记录
    print()
    print("=" * 80)
    print("开始修复...")
    print("=" * 80)
    fixed_count = 0
    
    for item in need_fix:
        try:
            import sqlite3
            with db._lock:
                conn = sqlite3.connect(str(db.db_path))
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE history_records 
                    SET transfer_amount = ?, balance_after = ?
                    WHERE id = ?
                """, (item['correct_transfer'], item['correct_balance_after'], item['id']))
                conn.commit()
                conn.close()
            
            fixed_count += 1
            
        except Exception as e:
            print(f"❌ 修复失败 - 账号 {item['phone']}, 日期 {item['date']}: {e}")
    
    # 总结
    print()
    print("=" * 80)
    print("修复完成")
    print("=" * 80)
    print(f"需要修复: {len(need_fix)}")
    print(f"已修复: {fixed_count}")
    print()


if __name__ == "__main__":
    try:
        fix_low_transfer_correct()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
