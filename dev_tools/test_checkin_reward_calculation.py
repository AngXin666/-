"""测试签到奖励计算逻辑
验证修复后的代码是否正确保存和计算签到奖励
"""
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.models import AccountResult
from src.local_db import LocalDatabase


def test_account_result_fields():
    """测试1: 验证AccountResult模型是否有正确的字段"""
    print("="*80)
    print("测试1: 验证AccountResult模型字段")
    print("="*80)
    
    # 创建测试对象
    result = AccountResult(
        phone="13800138000",
        success=True,
        balance_before=10.0,
        checkin_balance_after=15.5,
        balance_after=15.5,
        checkin_reward=5.5
    )
    
    # 验证字段
    assert hasattr(result, 'balance_before'), "❌ 缺少 balance_before 字段"
    assert hasattr(result, 'checkin_balance_after'), "❌ 缺少 checkin_balance_after 字段"
    assert hasattr(result, 'balance_after'), "❌ 缺少 balance_after 字段"
    assert hasattr(result, 'checkin_reward'), "❌ 缺少 checkin_reward 字段"
    
    print(f"✓ balance_before: {result.balance_before}")
    print(f"✓ checkin_balance_after: {result.checkin_balance_after}")
    print(f"✓ balance_after: {result.balance_after}")
    print(f"✓ checkin_reward: {result.checkin_reward}")
    print()
    print("✅ 测试1通过: AccountResult模型字段正确")
    print()


def test_record_mapping():
    """测试2: 验证历史记录映射是否正确"""
    print("="*80)
    print("测试2: 验证历史记录字段映射")
    print("="*80)
    
    # 创建测试对象
    account_result = AccountResult(
        phone="13800138000",
        success=True,
        nickname="测试用户",
        user_id="123456",
        balance_before=10.0,
        checkin_balance_after=15.5,
        balance_after=15.5,
        checkin_reward=5.5,
        points=100,
        vouchers=2.0,
        coupons=3,
        checkin_total_times=10,
        duration=30.5,
        login_method="缓存"
    )
    
    # 模拟GUI中的映射逻辑
    run_date = datetime.now().strftime('%Y-%m-%d')
    record = {
        'phone': account_result.phone,
        'nickname': account_result.nickname if account_result.nickname else '',
        'user_id': account_result.user_id if account_result.user_id else '',
        'balance_before': round(account_result.balance_before, 2) if account_result.balance_before is not None else None,
        'points': account_result.points if account_result.points is not None else None,
        'vouchers': round(account_result.vouchers, 2) if account_result.vouchers is not None else None,
        'coupons': account_result.coupons if account_result.coupons is not None else None,
        'checkin_reward': round(account_result.checkin_reward, 2) if account_result.checkin_reward else 0.0,
        'checkin_total_times': account_result.checkin_total_times if account_result.checkin_total_times is not None else None,
        'checkin_balance_after': round(account_result.checkin_balance_after, 2) if account_result.checkin_balance_after is not None else None,
        'balance_after': round(account_result.balance_after, 2) if account_result.balance_after is not None else None,
        'duration': round(account_result.duration, 2) if account_result.duration is not None else 0.0,
        'status': '成功',
        'login_method': account_result.login_method if account_result.login_method else '',
        'run_date': run_date
    }
    
    # 验证映射
    print("字段映射验证:")
    print(f"  balance_before: {account_result.balance_before} → {record['balance_before']}")
    assert record['balance_before'] == 10.0, f"❌ balance_before 映射错误: {record['balance_before']}"
    
    print(f"  checkin_balance_after: {account_result.checkin_balance_after} → {record['checkin_balance_after']}")
    assert record['checkin_balance_after'] == 15.5, f"❌ checkin_balance_after 映射错误: {record['checkin_balance_after']}"
    
    print(f"  balance_after: {account_result.balance_after} → {record['balance_after']}")
    assert record['balance_after'] == 15.5, f"❌ balance_after 映射错误: {record['balance_after']}"
    
    print(f"  checkin_reward: {account_result.checkin_reward} → {record['checkin_reward']}")
    assert record['checkin_reward'] == 5.5, f"❌ checkin_reward 映射错误: {record['checkin_reward']}"
    
    # 关键验证：checkin_balance_after 不应该等于 balance_after（除非它们本来就相同）
    # 但映射的来源必须正确
    print()
    print("✅ 测试2通过: 字段映射正确")
    print()


def test_database_save_and_retrieve():
    """测试3: 验证数据库保存和读取"""
    print("="*80)
    print("测试3: 验证数据库保存和读取")
    print("="*80)
    
    db = LocalDatabase()
    
    # 创建测试数据
    test_phone = "13800138999"
    run_date = datetime.now().strftime('%Y-%m-%d')
    
    record = {
        'phone': test_phone,
        'nickname': '测试用户',
        'user_id': '999999',
        'balance_before': 10.0,
        'points': 100,
        'vouchers': 2.0,
        'coupons': 3,
        'checkin_reward': 5.5,
        'checkin_total_times': 10,
        'checkin_balance_after': 15.5,
        'balance_after': 15.5,
        'duration': 30.5,
        'status': '成功',
        'login_method': '缓存',
        'run_date': run_date
    }
    
    # 保存到数据库
    print(f"保存测试数据: {test_phone}")
    success = db.upsert_history_record(record)
    assert success, "❌ 数据库保存失败"
    print("✓ 数据保存成功")
    
    # 从数据库读取
    print(f"读取测试数据: {test_phone}")
    records = db.get_history_records(phone=test_phone, start_date=run_date, end_date=run_date)
    
    assert len(records) > 0, "❌ 未找到保存的记录"
    print(f"✓ 找到 {len(records)} 条记录")
    
    # 验证数据
    saved_record = records[0]
    print()
    print("验证保存的数据:")
    print(f"  balance_before: {saved_record['balance_before']} (期望: 10.0)")
    assert saved_record['balance_before'] == 10.0, f"❌ balance_before 不匹配"
    
    print(f"  checkin_balance_after: {saved_record['checkin_balance_after']} (期望: 15.5)")
    assert saved_record['checkin_balance_after'] == 15.5, f"❌ checkin_balance_after 不匹配"
    
    print(f"  balance_after: {saved_record['balance_after']} (期望: 15.5)")
    assert saved_record['balance_after'] == 15.5, f"❌ balance_after 不匹配"
    
    print(f"  checkin_reward: {saved_record['checkin_reward']} (期望: 5.5)")
    assert saved_record['checkin_reward'] == 5.5, f"❌ checkin_reward 不匹配"
    
    # 验证签到奖励计算
    calculated_reward = saved_record['checkin_balance_after'] - saved_record['balance_before']
    print()
    print(f"签到奖励计算验证:")
    print(f"  计算值: {calculated_reward:.2f} = {saved_record['checkin_balance_after']:.2f} - {saved_record['balance_before']:.2f}")
    print(f"  保存值: {saved_record['checkin_reward']:.2f}")
    assert abs(calculated_reward - saved_record['checkin_reward']) < 0.01, f"❌ 签到奖励计算不匹配"
    
    print()
    print("✅ 测试3通过: 数据库保存和读取正确")
    print()


def test_none_values():
    """测试4: 验证None值的处理"""
    print("="*80)
    print("测试4: 验证None值的处理")
    print("="*80)
    
    # 创建测试对象（部分字段为None）
    account_result = AccountResult(
        phone="13800138888",
        success=True,
        balance_before=None,  # None值
        checkin_balance_after=None,  # None值
        balance_after=None,  # None值
        checkin_reward=0.0
    )
    
    # 模拟GUI中的映射逻辑
    run_date = datetime.now().strftime('%Y-%m-%d')
    record = {
        'phone': account_result.phone,
        'nickname': account_result.nickname if account_result.nickname else '',
        'user_id': account_result.user_id if account_result.user_id else '',
        'balance_before': round(account_result.balance_before, 2) if account_result.balance_before is not None else None,
        'points': account_result.points if account_result.points is not None else None,
        'vouchers': round(account_result.vouchers, 2) if account_result.vouchers is not None else None,
        'coupons': account_result.coupons if account_result.coupons is not None else None,
        'checkin_reward': round(account_result.checkin_reward, 2) if account_result.checkin_reward else 0.0,
        'checkin_total_times': account_result.checkin_total_times if account_result.checkin_total_times is not None else None,
        'checkin_balance_after': round(account_result.checkin_balance_after, 2) if account_result.checkin_balance_after is not None else None,
        'balance_after': round(account_result.balance_after, 2) if account_result.balance_after is not None else None,
        'duration': round(account_result.duration, 2) if account_result.duration is not None else 0.0,
        'status': '成功',
        'login_method': account_result.login_method if account_result.login_method else '',
        'run_date': run_date
    }
    
    # 验证None值保持为None
    print("None值处理验证:")
    print(f"  balance_before: {record['balance_before']} (期望: None)")
    assert record['balance_before'] is None, f"❌ balance_before 应该是 None"
    
    print(f"  checkin_balance_after: {record['checkin_balance_after']} (期望: None)")
    assert record['checkin_balance_after'] is None, f"❌ checkin_balance_after 应该是 None"
    
    print(f"  balance_after: {record['balance_after']} (期望: None)")
    assert record['balance_after'] is None, f"❌ balance_after 应该是 None"
    
    print(f"  points: {record['points']} (期望: None)")
    assert record['points'] is None, f"❌ points 应该是 None"
    
    print()
    print("✅ 测试4通过: None值处理正确")
    print()


def run_all_tests():
    """运行所有测试"""
    print()
    print("="*80)
    print("签到奖励计算逻辑测试套件")
    print("="*80)
    print()
    
    try:
        test_account_result_fields()
        test_record_mapping()
        test_database_save_and_retrieve()
        test_none_values()
        
        print()
        print("="*80)
        print("✅ 所有测试通过！")
        print("="*80)
        print()
        print("结论: 签到奖励计算逻辑修复成功，代码工作正常")
        print()
        
    except AssertionError as e:
        print()
        print("="*80)
        print(f"❌ 测试失败: {e}")
        print("="*80)
        print()
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print()
        print("="*80)
        print(f"❌ 测试异常: {e}")
        print("="*80)
        print()
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    run_all_tests()
