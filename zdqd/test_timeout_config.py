"""
测试超时配置

验证所有主要超时是否已设置为 15 秒
"""

from src.timeouts_config import TimeoutsConfig


def test_main_timeouts():
    """测试主要超时配置"""
    print("="*60)
    print("测试主要超时配置（应该都是 15 秒）")
    print("="*60)
    
    main_timeouts = {
        'NAVIGATION_TIMEOUT': 15.0,
        'PAGE_LOAD_TIMEOUT': 15.0,
        'CHECKIN_TIMEOUT': 15.0,
        'OCR_TIMEOUT_LONG': 15.0,
        'HTTP_REQUEST_TIMEOUT': 15.0,
        'SMART_WAIT_TIMEOUT': 15.0,
    }
    
    all_passed = True
    
    for name, expected in main_timeouts.items():
        actual = getattr(TimeoutsConfig, name)
        status = "✓" if actual == expected else "✗"
        
        if actual != expected:
            all_passed = False
            print(f"{status} {name:30s} = {actual:6.2f} 秒 (期望: {expected:.2f} 秒) ❌")
        else:
            print(f"{status} {name:30s} = {actual:6.2f} 秒 ✅")
    
    print()
    if all_passed:
        print("✅ 所有主要超时配置正确！")
    else:
        print("❌ 部分超时配置不正确！")
    
    return all_passed


def test_special_timeouts():
    """测试特殊超时配置（应该保持原值）"""
    print("\n" + "="*60)
    print("测试特殊超时配置（保持原值）")
    print("="*60)
    
    special_timeouts = {
        'TRANSFER_TIMEOUT': 20.0,  # 转账需要更长时间
        'PAGE_TRANSITION_TIMEOUT': 5.0,  # 快速切换
        'OCR_TIMEOUT': 5.0,  # 常规 OCR
        'OCR_TIMEOUT_SHORT': 2.0,  # 快速 OCR
        'PAGE_DETECT_TIMEOUT': 5.0,  # 页面检测
        'ELEMENT_DETECT_TIMEOUT': 3.0,  # 元素检测
        'HTTP_REQUEST_SHORT': 5.0,  # 短 HTTP 请求
    }
    
    all_passed = True
    
    for name, expected in special_timeouts.items():
        actual = getattr(TimeoutsConfig, name)
        status = "✓" if actual == expected else "✗"
        
        if actual != expected:
            all_passed = False
            print(f"{status} {name:30s} = {actual:6.2f} 秒 (期望: {expected:.2f} 秒) ❌")
        else:
            print(f"{status} {name:30s} = {actual:6.2f} 秒 ✅")
    
    print()
    if all_passed:
        print("✅ 所有特殊超时配置正确！")
    else:
        print("❌ 部分特殊超时配置不正确！")
    
    return all_passed


def test_wait_times():
    """测试等待时间配置"""
    print("\n" + "="*60)
    print("测试等待时间配置")
    print("="*60)
    
    wait_times = {
        'WAIT_SHORT': 0.5,
        'WAIT_MEDIUM': 1.0,
        'WAIT_LONG': 2.0,
        'WAIT_EXTRA_LONG': 3.0,
    }
    
    all_passed = True
    
    for name, expected in wait_times.items():
        actual = getattr(TimeoutsConfig, name)
        status = "✓" if actual == expected else "✗"
        
        if actual != expected:
            all_passed = False
            print(f"{status} {name:30s} = {actual:6.2f} 秒 (期望: {expected:.2f} 秒) ❌")
        else:
            print(f"{status} {name:30s} = {actual:6.2f} 秒 ✅")
    
    print()
    if all_passed:
        print("✅ 所有等待时间配置正确！")
    else:
        print("❌ 部分等待时间配置不正确！")
    
    return all_passed


def print_all_config():
    """打印所有配置"""
    print("\n" + "="*60)
    print("所有超时配置")
    print("="*60)
    
    TimeoutsConfig.print_config()


if __name__ == "__main__":
    # 运行测试
    result1 = test_main_timeouts()
    result2 = test_special_timeouts()
    result3 = test_wait_times()
    
    # 打印所有配置
    print_all_config()
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    if result1 and result2 and result3:
        print("✅ 所有测试通过！超时配置正确。")
    else:
        print("❌ 部分测试失败！请检查超时配置。")
    
    print("="*60)
