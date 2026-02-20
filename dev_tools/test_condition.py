"""
测试判断条件
"""

balance_before = 5.76
balance_after = 5.76
old_checkin_reward = -21.0

print(f"balance_before: {balance_before}")
print(f"balance_after: {balance_after}")
print(f"old_checkin_reward: {old_checkin_reward}")
print()

# 测试判断条件
if balance_before is not None and abs(balance_after - balance_before) < 0.001:
    print("✓ 签到失败判断通过")
    new_checkin_reward = 0.0
    print(f"new_checkin_reward: {new_checkin_reward}")
    print()
    
    # 检查是否需要更新
    if abs(new_checkin_reward - old_checkin_reward) > 0.001:
        print("✓ 需要更新")
    else:
        print("✗ 不需要更新")
else:
    print("✗ 签到失败判断未通过")
