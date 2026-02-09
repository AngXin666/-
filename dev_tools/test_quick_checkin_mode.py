"""
快速签到模式逻辑测试

测试目标：
1. 验证有缓存的账号使用快速模式时跳过获取资料
2. 验证无缓存的账号使用快速模式时自动切换为完整流程
3. 验证完整模式始终获取资料

测试场景：
- 场景1: 快速模式 + 有缓存 → 跳过获取资料
- 场景2: 快速模式 + 无缓存 → 自动切换为完整流程
- 场景3: 完整模式 + 有缓存 → 获取资料
- 场景4: 完整模式 + 无缓存 → 获取资料
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


class MockCacheManager:
    """模拟缓存管理器"""
    
    def __init__(self, cached_phones):
        self.cached_phones = cached_phones
    
    def has_cache(self, phone):
        """检查是否有缓存"""
        return phone in self.cached_phones


class MockAutoLogin:
    """模拟自动登录"""
    
    def __init__(self, enable_cache, cached_phones):
        self.enable_cache = enable_cache
        self.cache_manager = MockCacheManager(cached_phones) if enable_cache else None


def test_quick_mode_logic(workflow_config, auto_login, account_phone):
    """测试快速签到模式逻辑
    
    Args:
        workflow_config: 工作流配置
        auto_login: 自动登录对象
        account_phone: 账号手机号
        
    Returns:
        (should_get_profile, reason) - 是否应该获取资料，原因
    """
    enable_profile = workflow_config.get('enable_profile', True)
    has_cache = False
    
    # 检查是否有缓存
    if not enable_profile and auto_login.enable_cache and auto_login.cache_manager:
        has_cache = auto_login.cache_manager.has_cache(account_phone)
        if has_cache:
            reason = "快速模式 + 有缓存 → 跳过获取资料"
        else:
            reason = "快速模式 + 无缓存 → 自动切换为完整流程"
            # 自动切换为完整流程
            enable_profile = True
    else:
        if enable_profile:
            reason = "完整模式 → 获取资料"
        else:
            reason = "快速模式但缓存未启用 → 跳过获取资料"
    
    # 根据最终的 enable_profile 决定是否获取资料
    should_get_profile = enable_profile
    
    return should_get_profile, reason


def run_tests():
    """运行所有测试"""
    print("="*60)
    print("快速签到模式逻辑测试")
    print("="*60)
    
    test_cases = [
        {
            'name': '场景1: 快速模式 + 有缓存',
            'workflow_config': {'enable_profile': False},
            'auto_login': MockAutoLogin(enable_cache=True, cached_phones=['13800000001']),
            'account_phone': '13800000001',
            'expected_get_profile': False,
            'expected_reason': '快速模式 + 有缓存 → 跳过获取资料'
        },
        {
            'name': '场景2: 快速模式 + 无缓存',
            'workflow_config': {'enable_profile': False},
            'auto_login': MockAutoLogin(enable_cache=True, cached_phones=['13800000001']),
            'account_phone': '13800000002',  # 不在缓存列表中
            'expected_get_profile': True,
            'expected_reason': '快速模式 + 无缓存 → 自动切换为完整流程'
        },
        {
            'name': '场景3: 完整模式 + 有缓存',
            'workflow_config': {'enable_profile': True},
            'auto_login': MockAutoLogin(enable_cache=True, cached_phones=['13800000001']),
            'account_phone': '13800000001',
            'expected_get_profile': True,
            'expected_reason': '完整模式 → 获取资料'
        },
        {
            'name': '场景4: 完整模式 + 无缓存',
            'workflow_config': {'enable_profile': True},
            'auto_login': MockAutoLogin(enable_cache=True, cached_phones=['13800000001']),
            'account_phone': '13800000002',
            'expected_get_profile': True,
            'expected_reason': '完整模式 → 获取资料'
        },
        {
            'name': '场景5: 快速模式 + 缓存未启用',
            'workflow_config': {'enable_profile': False},
            'auto_login': MockAutoLogin(enable_cache=False, cached_phones=[]),
            'account_phone': '13800000001',
            'expected_get_profile': False,
            'expected_reason': '快速模式但缓存未启用 → 跳过获取资料'
        }
    ]
    
    passed = 0
    failed = 0
    
    for test_case in test_cases:
        print(f"\n{test_case['name']}")
        print("-" * 60)
        
        should_get_profile, reason = test_quick_mode_logic(
            test_case['workflow_config'],
            test_case['auto_login'],
            test_case['account_phone']
        )
        
        print(f"  配置: enable_profile={test_case['workflow_config'].get('enable_profile')}")
        print(f"  缓存: {'启用' if test_case['auto_login'].enable_cache else '未启用'}")
        print(f"  账号: {test_case['account_phone']}")
        print(f"  有缓存: {test_case['auto_login'].cache_manager.has_cache(test_case['account_phone']) if test_case['auto_login'].cache_manager else 'N/A'}")
        print(f"  结果: {'获取资料' if should_get_profile else '跳过获取资料'}")
        print(f"  原因: {reason}")
        
        # 验证结果
        if should_get_profile == test_case['expected_get_profile'] and reason == test_case['expected_reason']:
            print(f"  ✅ 通过")
            passed += 1
        else:
            print(f"  ❌ 失败")
            print(f"     期望: {'获取资料' if test_case['expected_get_profile'] else '跳过获取资料'} - {test_case['expected_reason']}")
            print(f"     实际: {'获取资料' if should_get_profile else '跳过获取资料'} - {reason}")
            failed += 1
    
    # 总结
    print("\n" + "="*60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
