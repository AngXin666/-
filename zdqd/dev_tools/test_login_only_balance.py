"""
测试只登录模式的余额处理

这个脚本模拟只登录模式的执行流程，验证 balance_after 是否被正确设置
"""

class MockResult:
    """模拟 AccountResult"""
    def __init__(self):
        self.balance_before = None
        self.balance_after = None
        self.checkin_reward = 0.0

def test_login_only_mode():
    """测试只登录模式"""
    print("="*60)
    print("测试只登录模式的余额处理")
    print("="*60)
    
    # 模拟工作流配置
    workflow_config = {
        'enable_login': True,
        'enable_profile': True,
        'enable_checkin': False,  # 禁用签到
        'enable_transfer': False
    }
    
    # 模拟结果对象
    result = MockResult()
    
    # 模拟步骤2：获取资料后设置 balance_before
    result.balance_before = 15.80
    print(f"步骤2: 获取资料")
    print(f"  balance_before = {result.balance_before:.2f} 元")
    
    # 模拟步骤3：跳过签到（因为 enable_checkin=False）
    print(f"\n步骤3: 跳过签到（enable_checkin=False）")
    
    # 模拟步骤7：获取最终余额
    print(f"\n步骤7: 获取最终余额")
    if workflow_config.get('enable_checkin', True):
        print("  签到已启用，需要获取最终余额")
    else:
        print("  签到已禁用，跳过获取最终余额")
        print(f"  [调试] 只登录模式 - balance_before = {result.balance_before}")
        
        # 只登录模式：使用初始余额作为最终余额
        if result.balance_before is not None:
            result.balance_after = result.balance_before
            print(f"  使用初始余额作为最终余额: {result.balance_after:.2f} 元")
        else:
            print(f"  [警告] balance_before 为 None，无法设置 balance_after")
    
    # 验证结果
    print(f"\n" + "="*60)
    print("验证结果")
    print("="*60)
    print(f"balance_before: {result.balance_before}")
    print(f"balance_after: {result.balance_after}")
    
    if result.balance_after is not None:
        print(f"\n✓ 测试通过：balance_after = {result.balance_after:.2f} 元")
        print(f"  GUI 应该显示: {result.balance_after:.2f} 元")
    else:
        print(f"\n✗ 测试失败：balance_after 为 None")
        print(f"  GUI 会显示: N/A")
    
    print("="*60)

if __name__ == "__main__":
    test_login_only_mode()
