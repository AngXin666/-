"""
修复2月4日失败记录的状态

将数据完整有效的失败记录改为成功

运行方式:
    python dev_tools/fix_feb4_status.py
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


def fix_feb4_status():
    """修复2月4日失败记录的状态"""
    
    print("=" * 80)
    print("修复2月4日失败记录的状态")
    print("=" * 80)
    print()
    
    # 初始化数据库
    db = LocalDatabase()
    all_records = db.get_all_history_records()
    
    # 筛选2月4日的失败记录
    feb4_failed = [r for r in all_records 
                   if r.get('run_date') == '2026-02-04' and r.get('status') == '失败']
    
    print(f"找到 {len(feb4_failed)} 条2月4日的失败记录")
    print()
    
    if not feb4_failed:
        print("✓ 没有需要修复的记录")
        return
    
    # 检查数据完整性并修复
    fixed_count = 0
    
    for record in feb4_failed:
        record_id = record.get('id')
        phone = record.get('phone')
        checkin_times = record.get('checkin_total_times') or 0
        balance_before = record.get('balance_before')
        balance_after = record.get('balance_after')
        checkin_balance_after = record.get('checkin_balance_after')
        
        # 判断数据是否完整
        is_valid = (
            checkin_times > 0 and
            balance_before is not None and
            balance_after is not None and
            checkin_balance_after is not None
        )
        
        if is_valid:
            # 数据完整，修改状态为成功
            try:
                import sqlite3
                with db._lock:
                    conn = sqlite3.connect(str(db.db_path))
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE history_records 
                        SET status = '成功'
                        WHERE id = ?
                    """, (record_id,))
                    conn.commit()
                    conn.close()
                
                fixed_count += 1
                print(f"✓ [{phone}] 状态已改为'成功'")
                
            except Exception as e:
                print(f"❌ [{phone}] 修复失败: {e}")
        else:
            print(f"⚠️ [{phone}] 数据不完整，跳过")
    
    # 输出结果
    print()
    print("=" * 80)
    print("修复完成")
    print("=" * 80)
    print(f"总记录数: {len(feb4_failed)}")
    print(f"已修复: {fixed_count}")
    print()


if __name__ == "__main__":
    try:
        fix_feb4_status()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
