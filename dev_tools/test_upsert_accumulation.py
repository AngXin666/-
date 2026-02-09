"""测试upsert累计逻辑
验证签到奖励和转账金额的累计是否正确
"""
import sys
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.local_db import LocalDatabase


def test_checkin_reward_accumulation():
    """测试签到奖励累计"""
    print("="*80)
    print("测试: 签到奖励累计逻辑")
    print("="*80)
    
    db = LocalDatabase()
    test_phone = "13900139000"
    run_date = datetime.now().strftime('%Y-%m-%d')
    
    # 第一次保存：签到完成
    print("\n第一次保存: 签到完成")
    record1 = {
        'phone': test_phone,
        'balance_before': 10.0,
        'checkin_reward': 5.5,
        'checkin_balance_after': 15.5,
        'balance_after': 15.5,
        'status': '成功',
        'run_date': run_date
    }
    
    success = db.upsert_history_record(record1)
    assert success, "❌ 第一次保存失败"
    print("✓ 第一次保存成功")
    
    # 读取验证
    records = db.get_history_records(phone=test_phone, start_date=run_date, end_date=run_date)
    assert len(records) > 0, "❌ 未找到记录"
    
    saved = records[0]
    print(f"  签到奖励: {saved['checkin_reward']:.2f} (期望: 5.5)")
    assert saved['checkin_reward'] == 5.5, f"❌ 签到奖励不匹配: {saved['checkin_reward']}"
    
    # 第二次保存：转账完成（不应该改变签到奖励）
    print("\n第二次保存: 转账完成（签到奖励为0）")
    record2 = {
        'phone': test_phone,
        'checkin_reward': 0.0,  # 转账时签到奖励为0
        'transfer_amount': 10.0,
        'transfer_recipient': '收款人A',
        'balance_after': 5.5,  # 转账后余额减少
        'status': '成功',
        'run_date': run_date
    }
    
    success = db.upsert_history_record(record2)
    assert success, "❌ 第二次保存失败"
    print("✓ 第二次保存成功")
    
    # 读取验证
    records = db.get_history_records(phone=test_phone, start_date=run_date, end_date=run_date)
    saved = records[0]
    
    print(f"  签到奖励: {saved['checkin_reward']:.2f} (期望: 5.5，不应该变化)")
    print(f"  转账金额: {saved['transfer_amount']:.2f} (期望: 10.0)")
    print(f"  余额后: {saved['balance_after']:.2f} (期望: 5.5)")
    
    # 签到奖励不应该变化（因为第二次的checkin_reward=0，不会累计）
    assert saved['checkin_reward'] == 5.5, f"❌ 签到奖励被错误修改: {saved['checkin_reward']}"
    assert saved['transfer_amount'] == 10.0, f"❌ 转账金额不匹配: {saved['transfer_amount']}"
    assert saved['balance_after'] == 5.5, f"❌ 余额后不匹配: {saved['balance_after']}"
    
    print("\n✅ 测试通过: 签到奖励累计逻辑正确")
    print()


def test_multiple_checkin_accumulation():
    """测试多次签到累计（理论场景）"""
    print("="*80)
    print("测试: 多次签到累计（理论场景）")
    print("="*80)
    
    db = LocalDatabase()
    test_phone = "13900139001"
    run_date = datetime.now().strftime('%Y-%m-%d')
    
    # 第一次签到
    print("\n第一次签到: 奖励 3.0 元")
    record1 = {
        'phone': test_phone,
        'balance_before': 10.0,
        'checkin_reward': 3.0,
        'checkin_balance_after': 13.0,
        'balance_after': 13.0,
        'status': '成功',
        'run_date': run_date
    }
    
    db.upsert_history_record(record1)
    records = db.get_history_records(phone=test_phone, start_date=run_date, end_date=run_date)
    saved = records[0]
    print(f"  累计签到奖励: {saved['checkin_reward']:.2f}")
    assert saved['checkin_reward'] == 3.0, f"❌ 第一次签到奖励不匹配"
    
    # 第二次签到（假设可以多次签到）
    print("\n第二次签到: 奖励 2.5 元")
    record2 = {
        'phone': test_phone,
        'checkin_reward': 2.5,
        'checkin_balance_after': 15.5,
        'balance_after': 15.5,
        'status': '成功',
        'run_date': run_date
    }
    
    db.upsert_history_record(record2)
    records = db.get_history_records(phone=test_phone, start_date=run_date, end_date=run_date)
    saved = records[0]
    print(f"  累计签到奖励: {saved['checkin_reward']:.2f} (期望: 5.5 = 3.0 + 2.5)")
    assert saved['checkin_reward'] == 5.5, f"❌ 累计签到奖励不匹配: {saved['checkin_reward']}"
    
    print("\n✅ 测试通过: 多次签到累计正确")
    print()


def test_transfer_accumulation():
    """测试转账金额累计"""
    print("="*80)
    print("测试: 转账金额累计")
    print("="*80)
    
    db = LocalDatabase()
    test_phone = "13900139002"
    run_date = datetime.now().strftime('%Y-%m-%d')
    
    # 初始记录
    print("\n初始记录: 签到完成")
    record1 = {
        'phone': test_phone,
        'balance_before': 20.0,
        'checkin_reward': 5.0,
        'balance_after': 25.0,
        'status': '成功',
        'run_date': run_date
    }
    
    db.upsert_history_record(record1)
    
    # 第一次转账
    print("\n第一次转账: 10.0 元")
    record2 = {
        'phone': test_phone,
        'transfer_amount': 10.0,
        'transfer_recipient': '收款人A',
        'balance_after': 15.0,
        'status': '成功',
        'run_date': run_date
    }
    
    db.upsert_history_record(record2)
    records = db.get_history_records(phone=test_phone, start_date=run_date, end_date=run_date)
    saved = records[0]
    print(f"  累计转账: {saved['transfer_amount']:.2f}")
    assert saved['transfer_amount'] == 10.0, f"❌ 第一次转账金额不匹配"
    
    # 第二次转账
    print("\n第二次转账: 5.0 元")
    record3 = {
        'phone': test_phone,
        'transfer_amount': 5.0,
        'transfer_recipient': '收款人B',
        'balance_after': 10.0,
        'status': '成功',
        'run_date': run_date
    }
    
    db.upsert_history_record(record3)
    records = db.get_history_records(phone=test_phone, start_date=run_date, end_date=run_date)
    saved = records[0]
    print(f"  累计转账: {saved['transfer_amount']:.2f} (期望: 15.0 = 10.0 + 5.0)")
    assert saved['transfer_amount'] == 15.0, f"❌ 累计转账金额不匹配: {saved['transfer_amount']}"
    
    # 验证签到奖励没有被改变
    print(f"  签到奖励: {saved['checkin_reward']:.2f} (期望: 5.0，不应该变化)")
    assert saved['checkin_reward'] == 5.0, f"❌ 签到奖励被错误修改"
    
    print("\n✅ 测试通过: 转账金额累计正确")
    print()


def run_all_tests():
    """运行所有测试"""
    print()
    print("="*80)
    print("UPSERT累计逻辑测试套件")
    print("="*80)
    print()
    
    try:
        test_checkin_reward_accumulation()
        test_multiple_checkin_accumulation()
        test_transfer_accumulation()
        
        print()
        print("="*80)
        print("✅ 所有测试通过！")
        print("="*80)
        print()
        print("结论: UPSERT累计逻辑工作正常")
        print("  - 签到奖励只在值>0时累计")
        print("  - 转账金额只在值>0时累计")
        print("  - 其他字段正确更新")
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
