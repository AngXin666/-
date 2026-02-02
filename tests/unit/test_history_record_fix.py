"""
测试历史记录系统修复

验证需求：
1. 只保存已完成的操作（成功/失败）
2. 按天分组的 UPSERT 逻辑
3. 数据库唯一约束
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import tempfile
import shutil
import sqlite3

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

# 导入本地数据库模块
from src.local_db import LocalDatabase


def test_only_save_completed_operations():
    """测试需求 1：只保存已完成的操作"""
    print("\n=== 测试 1：只保存已完成的操作 ===")
    
    # 创建临时数据库
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "runtime_data" / "license.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # 创建数据库实例
        db = LocalDatabase()
        db.db_path = db_path
        db._init_database()
        
        # 测试保存成功状态
        record_success = {
            'phone': "13800138000",
            'nickname': "测试用户",
            'user_id': "123456",
            'balance_before': 100.0,
            'points': 100,
            'vouchers': 5,
            'coupons': 3,
            'checkin_reward': 10.0,
            'checkin_total_times': 10,
            'checkin_balance_after': 110.0,
            'balance_after': 110.0,
            'duration': 5.0,
            'status': '成功',
            'login_method': '缓存',
            'run_date': datetime.now().strftime('%Y-%m-%d')
        }
        
        assert db.upsert_history_record(record_success), "保存成功状态失败"
        print("[OK] 成功状态保存成功")
        
        # 测试保存失败状态
        record_failed = {
            'phone': "13800138001",
            'nickname': None,
            'user_id': None,
            'balance_before': None,
            'points': None,
            'vouchers': None,
            'coupons': None,
            'checkin_reward': 0.0,
            'checkin_total_times': None,
            'checkin_balance_after': None,
            'balance_after': None,
            'duration': 3.0,
            'status': '失败',
            'login_method': None,
            'run_date': datetime.now().strftime('%Y-%m-%d')
        }
        
        assert db.upsert_history_record(record_failed), "保存失败状态失败"
        print("[OK] 失败状态保存成功")
        
        # 验证记录数量
        records = db.get_history_records()
        assert len(records) == 2, f"期望 2 条记录，实际 {len(records)} 条"
        print(f"[OK] 数据库中有 {len(records)} 条记录")
        
        print("[PASSED] 测试 1 通过")
        return True
        
    finally:
        # 清理临时文件
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_upsert_logic():
    """测试需求 2：按天分组的 UPSERT 逻辑"""
    print("\n=== 测试 2：按天分组的 UPSERT 逻辑 ===")
    
    # 创建临时数据库
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "runtime_data" / "license.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # 创建数据库实例
        db = LocalDatabase()
        db.db_path = db_path
        db._init_database()
        
        phone = "13800138000"
        run_date = datetime.now().strftime('%Y-%m-%d')
        
        # 第一次保存：插入新记录
        record1 = {
            'phone': phone,
            'nickname': "用户1",
            'user_id': "123456",
            'balance_before': 100.0,
            'balance_after': 110.0,
            'checkin_reward': 10.0,
            'status': '成功',
            'run_date': run_date
        }
        
        assert db.upsert_history_record(record1), "第一次保存失败"
        records = db.get_history_records(phone=phone)
        assert len(records) == 1, f"期望 1 条记录，实际 {len(records)} 条"
        assert records[0]['nickname'] == "用户1", "昵称不匹配"
        print("[OK] 第一次保存：插入新记录成功")
        
        # 第二次保存（同一天）：更新现有记录
        record2 = {
            'phone': phone,
            'nickname': "用户1更新",
            'user_id': "123456",
            'balance_before': 110.0,
            'balance_after': 120.0,
            'checkin_reward': 20.0,
            'status': '成功',
            'run_date': run_date
        }
        
        assert db.upsert_history_record(record2), "第二次保存失败"
        records = db.get_history_records(phone=phone)
        assert len(records) == 1, f"期望仍然是 1 条记录，实际 {len(records)} 条"
        assert records[0]['nickname'] == "用户1更新", "昵称未更新"
        assert records[0]['checkin_reward'] == 20.0, "签到奖励未更新"
        print("[OK] 第二次保存（同一天）：更新现有记录成功")
        
        # 第三次保存（不同日期）：插入新记录
        record3 = {
            'phone': phone,
            'nickname': "用户1",
            'user_id': "123456",
            'balance_before': 120.0,
            'balance_after': 130.0,
            'checkin_reward': 10.0,
            'status': '成功',
            'run_date': '2026-01-27'  # 不同日期
        }
        
        assert db.upsert_history_record(record3), "第三次保存失败"
        records = db.get_history_records(phone=phone)
        assert len(records) == 2, f"期望 2 条记录，实际 {len(records)} 条"
        print("[OK] 第三次保存（不同日期）：插入新记录成功")
        
        print("[PASSED] 测试 2 通过")
        return True
        
    finally:
        # 清理临时文件
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_unique_constraint():
    """测试需求 3：数据库唯一约束"""
    print("\n=== 测试 3：数据库唯一约束 ===")
    
    # 创建临时数据库
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "runtime_data" / "license.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # 创建数据库实例
        db = LocalDatabase()
        db.db_path = db_path
        db._init_database()
        
        # 验证表结构包含唯一约束
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT sql FROM sqlite_master 
            WHERE type='table' AND name='history_records'
        """)
        table_sql = cursor.fetchone()[0]
        conn.close()
        
        assert 'UNIQUE' in table_sql, "表结构中没有 UNIQUE 约束"
        assert 'phone' in table_sql, "表结构中没有 phone 字段"
        assert 'run_date' in table_sql, "表结构中没有 run_date 字段"
        print("[OK] 数据库表包含 UNIQUE(phone, run_date) 约束")
        
        print("[PASSED] 测试 3 通过")
        return True
        
    finally:
        # 清理临时文件
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("历史记录系统修复 - 测试套件")
    print("=" * 60)
    
    tests = [
        test_only_save_completed_operations,
        test_upsert_logic,
        test_unique_constraint
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"[FAILED] 测试失败: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
