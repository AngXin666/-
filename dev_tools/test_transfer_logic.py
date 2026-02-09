"""
转账逻辑单元测试
测试转账成功/失败的判断逻辑
"""


def test_transfer_logic():
    """测试转账判断逻辑的严谨性"""
    
    print("=" * 60)
    print("转账逻辑单元测试")
    print("=" * 60)
    
    # 测试用例
    test_cases = [
        # (initial_balance, final_balance, expected_success, description)
        (100.0, 50.0, True, "正常转账：100元 -> 50元"),
        (100.0, 0.0, True, "全部转账：100元 -> 0元"),
        (50.5, 20.3, True, "小数转账：50.5元 -> 20.3元"),
        (100.0, 100.0, False, "余额无变化：100元 -> 100元"),
        (100.0, 150.0, False, "余额增加：100元 -> 150元（异常）"),
        (0.0, 0.0, False, "零余额转账：0元 -> 0元"),
        (None, 50.0, False, "缺少转账前余额"),
        (100.0, None, False, "无法获取转账后余额"),
        (None, None, False, "缺少所有余额信息"),
    ]
    
    passed = 0
    failed = 0
    
    for i, (initial, final, expected, desc) in enumerate(test_cases, 1):
        print(f"\n测试 {i}: {desc}")
        print(f"  转账前余额: {initial}")
        print(f"  转账后余额: {final}")
        print(f"  预期结果: {'成功' if expected else '失败'}")
        
        # 模拟转账判断逻辑
        actual_success = judge_transfer_success(initial, final)
        
        print(f"  实际结果: {'成功' if actual_success else '失败'}")
        
        if actual_success == expected:
            print(f"  ✓ 测试通过")
            passed += 1
        else:
            print(f"  ✗ 测试失败")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    return failed == 0


def judge_transfer_success(initial_balance, final_balance):
    """
    转账成功判断逻辑（从 balance_transfer.py 提取）
    
    Args:
        initial_balance: 转账前余额
        final_balance: 转账后余额
        
    Returns:
        bool: 是否成功
    """
    # 1. 必须有转账前余额
    if initial_balance is None:
        return False
    
    # 2. 必须获取到转账后余额
    if final_balance is None:
        return False
    
    # 3. 计算余额变化
    balance_change = final_balance - initial_balance
    
    # 4. 余额减少 = 成功
    if balance_change < 0:
        return True
    else:
        return False


if __name__ == "__main__":
    success = test_transfer_logic()
    exit(0 if success else 1)
