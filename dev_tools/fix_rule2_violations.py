"""
修复规则2违反的记录

规则2: checkin_balance_after = balance_before + checkin_reward

对于签到后余额为0的记录，根据规则计算正确的值

运行方式:
    python dev_tools/fix_rule2_violations.py
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


def fix_rule2_violations():
    """修复规则2违反的记录"""
    
    print("=" * 80)
    print("修复规则2违反的记录")
    print("=" * 80)
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
    
    # 找出规则2违反的记录
    violations = []
    
    for phone, records in records_by_phone.items():
        for record in records:
            balance_before = record.get('balance_before')
            checkin_reward = record.get('checkin_reward', 0.0) or 0.0
            checkin_balance_after = record.get('checkin_balance_after')
            
            if balance_before is not None and checkin_balance_after is not None:
                expected = balance_before + checkin_reward
                if abs(checkin_balance_after - expected) > 0.01:
                    violations.append({
                        'id': record.get('id'),
                        'phone': phone,
                        'date': record.get('run_date'),
                        'balance_before': balance_before,
                        'checkin_reward': checkin_reward,
                        'checkin_balance_after': checkin_balance_after,
                        'balance_after': record.get('balance_after'),
                        'expected': expected
                    })
    
    print(f"发现 {len(violations)} 条违反规则2的记录")
    print()
    
    if not violations:
        print("✓ 没有发现问题记录")
        return
    
    # 修复所有违反记录
    fixed_count = 0
    
    for v in violations:
        print(f"修复账号 {v['phone']}, 日期 {v['date']}")
        print(f"  balance_before: {v['balance_before']:.2f}")
        print(f"  checkin_reward: {v['checkin_reward']:.2f}")
        print(f"  checkin_balance_after: {v['checkin_balance_after']:.2f} → {v['expected']:.2f}")
        
        # 如果签到后余额为0，说明数据缺失，需要修复
        # 修复策略：checkin_balance_after = balance_before + checkin_reward
        try:
            import sqlite3
            with db._lock:
                conn = sqlite3.connect(str(db.db_path))
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE history_records 
                    SET checkin_balance_after = ?
                    WHERE id = ?
                """, (v['expected'], v['id']))
                conn.commit()
                conn.close()
            
            fixed_count += 1
            print(f"  ✓ 修复成功")
            
        except Exception as e:
            print(f"  ❌ 修复失败: {e}")
        
        print()
    
    print("=" * 80)
    print(f"修复完成: {fixed_count}/{len(violations)} 条记录")
    print("=" * 80)


if __name__ == "__main__":
    try:
        fix_rule2_violations()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
