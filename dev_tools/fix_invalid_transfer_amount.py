"""
修复无效的转账金额

将转账金额 < 最小转账金额(30元) 的记录的 transfer_amount 设为 0
因为这些转账金额是错误的（应该是0或>=30）

运行方式:
    python dev_tools/fix_invalid_transfer_amount.py
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


def fix_invalid_transfer_amount():
    """修复无效的转账金额"""
    
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
    print("修复无效的转账金额")
    print("=" * 80)
    print(f"最小转账金额: {MIN_TRANSFER_AMOUNT} 元")
    print()
    print("将转账金额 > 0 但 < 30元 的记录的 transfer_amount 设为 0")
    print()
    
    # 初始化数据库
    db = LocalDatabase()
    all_records = db.get_all_history_records()
    
    print(f"总记录数: {len(all_records)}")
    print()
    
    # 找出转账金额 > 0 但 < 最小转账金额的记录
    invalid_transfer_records = []
    
    for record in all_records:
        transfer_amount = record.get('transfer_amount', 0.0) or 0.0
        if 0 < transfer_amount < MIN_TRANSFER_AMOUNT:
            invalid_transfer_records.append(record)
    
    print(f"转账金额 > 0 但 < {MIN_TRANSFER_AMOUNT} 元的记录: {len(invalid_transfer_records)} 条")
    print()
    
    if not invalid_transfer_records:
        print("✓ 没有需要修复的记录")
        return
    
    # 开始修复
    print("=" * 80)
    print("开始修复...")
    print("=" * 80)
    print()
    
    updated_count = 0
    error_count = 0
    
    for record in invalid_transfer_records:
        record_id = record.get('id')
        phone = record.get('phone')
        run_date = record.get('run_date')
        old_transfer_amount = record.get('transfer_amount', 0.0) or 0.0
        
        try:
            conn = db._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE history_records 
                SET transfer_amount = 0.0
                WHERE id = ?
            ''', (record_id,))
            conn.commit()
            conn.close()
            
            updated_count += 1
            
            # 显示前20条更新
            if updated_count <= 20:
                print(f"[{phone}] [{run_date}]")
                print(f"  转账金额: {old_transfer_amount:.2f} → 0.00 元")
                print()
        
        except Exception as e:
            print(f"❌ [{phone}] [{run_date}] 更新失败: {e}")
            error_count += 1
    
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
    
    all_records = db.get_all_history_records()
    
    # 检查是否还有无效转账金额
    invalid_transfer_records = []
    for record in all_records:
        transfer_amount = record.get('transfer_amount', 0.0) or 0.0
        if 0 < transfer_amount < MIN_TRANSFER_AMOUNT:
            invalid_transfer_records.append(record)
    
    print(f"转账金额 > 0 但 < {MIN_TRANSFER_AMOUNT} 元的记录: {len(invalid_transfer_records)} 条")
    
    if invalid_transfer_records:
        print()
        print("仍有无效转账金额的记录:")
        for record in invalid_transfer_records[:10]:
            phone = record.get('phone')
            run_date = record.get('run_date')
            transfer_amount = record.get('transfer_amount', 0.0) or 0.0
            print(f"  [{phone}] [{run_date}] transfer_amount: {transfer_amount:.2f}")
    else:
        print("✓ 所有记录的转账金额都已修复")
    
    print()


if __name__ == "__main__":
    try:
        fix_invalid_transfer_amount()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
