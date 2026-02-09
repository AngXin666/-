"""
数据库浮点数精度测试

测试目标：
1. 验证插入记录时浮点数精度正确（保留2位小数）
2. 验证更新记录时浮点数精度正确（保留2位小数）
3. 验证累计计算时精度正确（如签到奖励、转账金额）
4. 验证读取记录时精度正确

测试场景：
- 场景1: 插入带精度误差的浮点数（如 18.86999999）
- 场景2: 更新带精度误差的浮点数
- 场景3: 累计计算（签到奖励、转账金额）
- 场景4: 混合整数和浮点数
- 场景5: 边界值测试（0, 0.01, 999999.99）
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import tempfile
import shutil

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from local_db import LocalDatabase


def test_insert_float_precision():
    """测试插入记录时的浮点数精度"""
    print("\n" + "="*60)
    print("测试场景1: 插入带精度误差的浮点数")
    print("="*60)
    
    # 创建临时数据库
    temp_dir = tempfile.mkdtemp()
    original_db_path = Path("runtime_data") / "license.db"
    temp_db_path = Path(temp_dir) / "license.db"
    
    # 备份原数据库路径
    db = LocalDatabase()
    original_path = db.db_path
    db.db_path = temp_db_path
    db._init_database()
    
    test_cases = [
        {
            'name': '精度误差测试1',
            'data': {
                'phone': '13800000001',
                'balance_before': 18.86999999,  # 应该保存为 18.87
                'balance_after': 17.70000000,   # 应该保存为 17.70
                'checkin_reward': 0.15999999,   # 应该保存为 0.16
                'vouchers': 5.49999999,         # 应该保存为 5.50
                'transfer_amount': 1.00000001,  # 应该保存为 1.00
                'duration': 45.123456,          # 应该保存为 45.12
                'status': '成功',
                'run_date': '2026-02-08'
            },
            'expected': {
                'balance_before': 18.87,
                'balance_after': 17.70,
                'checkin_reward': 0.16,
                'vouchers': 5.50,
                'transfer_amount': 1.00,
                'duration': 45.12
            }
        },
        {
            'name': '精度误差测试2',
            'data': {
                'phone': '13800000002',
                'balance_before': 99.99999999,
                'balance_after': 100.00000001,
                'checkin_reward': 0.01000001,
                'vouchers': 0.99999999,
                'transfer_amount': 50.50500001,
                'duration': 30.999999,
                'status': '成功',
                'run_date': '2026-02-08'
            },
            'expected': {
                'balance_before': 100.00,
                'balance_after': 100.00,
                'checkin_reward': 0.01,
                'vouchers': 1.00,
                'transfer_amount': 50.51,
                'duration': 31.00
            }
        }
    ]
    
    passed = 0
    failed = 0
    
    for test_case in test_cases:
        try:
            # 插入记录
            success = db.upsert_history_record(test_case['data'])
            if not success:
                print(f"❌ {test_case['name']}: 插入失败")
                failed += 1
                continue
            
            # 读取记录验证
            records = db.get_history_records(phone=test_case['data']['phone'], limit=1)
            if not records:
                print(f"❌ {test_case['name']}: 读取失败")
                failed += 1
                continue
            
            record = records[0]
            
            # 验证每个浮点数字段
            all_correct = True
            for field, expected_value in test_case['expected'].items():
                actual_value = record.get(field)
                if actual_value != expected_value:
                    print(f"❌ {test_case['name']}: {field} 不匹配")
                    print(f"   期望: {expected_value}, 实际: {actual_value}")
                    all_correct = False
            
            if all_correct:
                print(f"✅ {test_case['name']}: 通过")
                passed += 1
            else:
                failed += 1
                
        except Exception as e:
            print(f"❌ {test_case['name']}: 异常 - {e}")
            failed += 1
    
    # 恢复原数据库路径
    db.db_path = original_path
    
    # 清理临时目录
    shutil.rmtree(temp_dir)
    
    print(f"\n测试结果: {passed} 通过, {failed} 失败")
    return failed == 0


def test_update_float_precision():
    """测试更新记录时的浮点数精度"""
    print("\n" + "="*60)
    print("测试场景2: 更新带精度误差的浮点数")
    print("="*60)
    
    # 创建临时数据库
    temp_dir = tempfile.mkdtemp()
    temp_db_path = Path(temp_dir) / "license.db"
    
    db = LocalDatabase()
    original_path = db.db_path
    db.db_path = temp_db_path
    db._init_database()
    
    test_cases = [
        {
            'name': '更新精度测试1',
            'initial': {
                'phone': '13800000003',
                'balance_before': 10.00,
                'balance_after': 10.00,
                'status': '成功',
                'run_date': '2026-02-08'
            },
            'update': {
                'phone': '13800000003',
                'balance_before': 20.86999999,  # 应该更新为 20.87
                'balance_after': 19.70000000,   # 应该更新为 19.70
                'status': '成功',
                'run_date': '2026-02-08'
            },
            'expected': {
                'balance_before': 20.87,
                'balance_after': 19.70
            }
        },
        {
            'name': '更新精度测试2',
            'initial': {
                'phone': '13800000004',
                'vouchers': 5.00,
                'transfer_amount': 0.00,
                'status': '成功',
                'run_date': '2026-02-08'
            },
            'update': {
                'phone': '13800000004',
                'vouchers': 10.49999999,        # 应该更新为 10.50
                'transfer_amount': 5.50500001,  # 应该更新为 5.51
                'status': '成功',
                'run_date': '2026-02-08'
            },
            'expected': {
                'vouchers': 10.50,
                'transfer_amount': 5.51
            }
        }
    ]
    
    passed = 0
    failed = 0
    
    for test_case in test_cases:
        try:
            # 插入初始记录
            db.upsert_history_record(test_case['initial'])
            
            # 更新记录
            success = db.upsert_history_record(test_case['update'])
            if not success:
                print(f"❌ {test_case['name']}: 更新失败")
                failed += 1
                continue
            
            # 读取记录验证
            records = db.get_history_records(phone=test_case['update']['phone'], limit=1)
            if not records:
                print(f"❌ {test_case['name']}: 读取失败")
                failed += 1
                continue
            
            record = records[0]
            
            # 验证每个浮点数字段
            all_correct = True
            for field, expected_value in test_case['expected'].items():
                actual_value = record.get(field)
                if actual_value != expected_value:
                    print(f"❌ {test_case['name']}: {field} 不匹配")
                    print(f"   期望: {expected_value}, 实际: {actual_value}")
                    all_correct = False
            
            if all_correct:
                print(f"✅ {test_case['name']}: 通过")
                passed += 1
            else:
                failed += 1
                
        except Exception as e:
            print(f"❌ {test_case['name']}: 异常 - {e}")
            failed += 1
    
    # 恢复原数据库路径
    db.db_path = original_path
    
    # 清理临时目录
    shutil.rmtree(temp_dir)
    
    print(f"\n测试结果: {passed} 通过, {failed} 失败")
    return failed == 0


def test_accumulation_precision():
    """测试累计计算时的精度（签到奖励、转账金额）"""
    print("\n" + "="*60)
    print("测试场景3: 累计计算精度测试")
    print("="*60)
    
    # 创建临时数据库
    temp_dir = tempfile.mkdtemp()
    temp_db_path = Path(temp_dir) / "license.db"
    
    db = LocalDatabase()
    original_path = db.db_path
    db.db_path = temp_db_path
    db._init_database()
    
    test_cases = [
        {
            'name': '签到奖励累计',
            'phone': '13800000005',
            'updates': [
                {'checkin_reward': 0.15999999, 'status': '成功', 'run_date': '2026-02-08'},  # 第1次：0.16
                {'checkin_reward': 0.15999999, 'status': '成功', 'run_date': '2026-02-08'},  # 第2次：累计 0.32
                {'checkin_reward': 0.15999999, 'status': '成功', 'run_date': '2026-02-08'},  # 第3次：累计 0.48
            ],
            'expected_checkin_reward': 0.48  # 0.16 * 3
        },
        {
            'name': '转账金额累计',
            'phone': '13800000006',
            'updates': [
                {'transfer_amount': 1.00000001, 'status': '成功', 'run_date': '2026-02-08'},  # 第1次：1.00
                {'transfer_amount': 2.50500001, 'status': '成功', 'run_date': '2026-02-08'},  # 第2次：累计 3.51
                {'transfer_amount': 0.49999999, 'status': '成功', 'run_date': '2026-02-08'},  # 第3次：累计 4.01
            ],
            'expected_transfer_amount': 4.01  # 1.00 + 2.51 + 0.50
        }
    ]
    
    passed = 0
    failed = 0
    
    for test_case in test_cases:
        try:
            # 执行多次更新
            for update_data in test_case['updates']:
                update_data['phone'] = test_case['phone']
                db.upsert_history_record(update_data)
            
            # 读取记录验证
            records = db.get_history_records(phone=test_case['phone'], limit=1)
            if not records:
                print(f"❌ {test_case['name']}: 读取失败")
                failed += 1
                continue
            
            record = records[0]
            
            # 验证累计值
            if 'expected_checkin_reward' in test_case:
                actual = record.get('checkin_reward')
                expected = test_case['expected_checkin_reward']
                if actual == expected:
                    print(f"✅ {test_case['name']}: 通过 (累计值: {actual})")
                    passed += 1
                else:
                    print(f"❌ {test_case['name']}: 累计值不匹配")
                    print(f"   期望: {expected}, 实际: {actual}")
                    failed += 1
            
            if 'expected_transfer_amount' in test_case:
                actual = record.get('transfer_amount')
                expected = test_case['expected_transfer_amount']
                if actual == expected:
                    print(f"✅ {test_case['name']}: 通过 (累计值: {actual})")
                    passed += 1
                else:
                    print(f"❌ {test_case['name']}: 累计值不匹配")
                    print(f"   期望: {expected}, 实际: {actual}")
                    failed += 1
                
        except Exception as e:
            print(f"❌ {test_case['name']}: 异常 - {e}")
            failed += 1
    
    # 恢复原数据库路径
    db.db_path = original_path
    
    # 清理临时目录
    shutil.rmtree(temp_dir)
    
    print(f"\n测试结果: {passed} 通过, {failed} 失败")
    return failed == 0


def test_boundary_values():
    """测试边界值（0, 0.01, 999999.99）"""
    print("\n" + "="*60)
    print("测试场景4: 边界值测试")
    print("="*60)
    
    # 创建临时数据库
    temp_dir = tempfile.mkdtemp()
    temp_db_path = Path(temp_dir) / "license.db"
    
    db = LocalDatabase()
    original_path = db.db_path
    db.db_path = temp_db_path
    db._init_database()
    
    test_cases = [
        {
            'name': '零值测试',
            'data': {
                'phone': '13800000007',
                'balance_before': 0.00,
                'balance_after': 0.00,
                'checkin_reward': 0.00,
                'vouchers': 0.00,
                'transfer_amount': 0.00,
                'duration': 0.00,
                'status': '成功',
                'run_date': '2026-02-08'
            },
            'expected': {
                'balance_before': 0.00,
                'balance_after': 0.00,
                'checkin_reward': 0.00,
                'vouchers': 0.00,
                'transfer_amount': 0.00,
                'duration': 0.00
            }
        },
        {
            'name': '最小值测试',
            'data': {
                'phone': '13800000008',
                'balance_before': 0.01,
                'balance_after': 0.01,
                'checkin_reward': 0.01,
                'vouchers': 0.01,
                'transfer_amount': 0.01,
                'duration': 0.01,
                'status': '成功',
                'run_date': '2026-02-08'
            },
            'expected': {
                'balance_before': 0.01,
                'balance_after': 0.01,
                'checkin_reward': 0.01,
                'vouchers': 0.01,
                'transfer_amount': 0.01,
                'duration': 0.01
            }
        },
        {
            'name': '大值测试',
            'data': {
                'phone': '13800000009',
                'balance_before': 999999.99,
                'balance_after': 999999.99,
                'checkin_reward': 999.99,
                'vouchers': 999.99,
                'transfer_amount': 999999.99,
                'duration': 9999.99,
                'status': '成功',
                'run_date': '2026-02-08'
            },
            'expected': {
                'balance_before': 999999.99,
                'balance_after': 999999.99,
                'checkin_reward': 999.99,
                'vouchers': 999.99,
                'transfer_amount': 999999.99,
                'duration': 9999.99
            }
        }
    ]
    
    passed = 0
    failed = 0
    
    for test_case in test_cases:
        try:
            # 插入记录
            success = db.upsert_history_record(test_case['data'])
            if not success:
                print(f"❌ {test_case['name']}: 插入失败")
                failed += 1
                continue
            
            # 读取记录验证
            records = db.get_history_records(phone=test_case['data']['phone'], limit=1)
            if not records:
                print(f"❌ {test_case['name']}: 读取失败")
                failed += 1
                continue
            
            record = records[0]
            
            # 验证每个浮点数字段
            all_correct = True
            for field, expected_value in test_case['expected'].items():
                actual_value = record.get(field)
                if actual_value != expected_value:
                    print(f"❌ {test_case['name']}: {field} 不匹配")
                    print(f"   期望: {expected_value}, 实际: {actual_value}")
                    all_correct = False
            
            if all_correct:
                print(f"✅ {test_case['name']}: 通过")
                passed += 1
            else:
                failed += 1
                
        except Exception as e:
            print(f"❌ {test_case['name']}: 异常 - {e}")
            failed += 1
    
    # 恢复原数据库路径
    db.db_path = original_path
    
    # 清理临时目录
    shutil.rmtree(temp_dir)
    
    print(f"\n测试结果: {passed} 通过, {failed} 失败")
    return failed == 0


def main():
    """运行所有测试"""
    print("="*60)
    print("数据库浮点数精度单元测试")
    print("="*60)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_passed = True
    
    # 测试1: 插入精度
    if not test_insert_float_precision():
        all_passed = False
    
    # 测试2: 更新精度
    if not test_update_float_precision():
        all_passed = False
    
    # 测试3: 累计计算精度
    if not test_accumulation_precision():
        all_passed = False
    
    # 测试4: 边界值
    if not test_boundary_values():
        all_passed = False
    
    # 总结
    print("\n" + "="*60)
    if all_passed:
        print("✅ 所有测试通过！")
    else:
        print("❌ 部分测试失败，请检查日志")
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
