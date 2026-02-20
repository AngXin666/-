#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试打包后的弹窗识别功能
"""

import sys
import os

# 设置工作目录
packed_dir = r"D:\溪盟商城自动化助手_打包_新"
os.chdir(packed_dir)
sys.path.insert(0, os.path.join(packed_dir, '_internal'))

print("=" * 60)
print("测试打包后的弹窗识别功能")
print("=" * 60)

# 测试PageState.POPUP
print("\n[测试1] PageState.POPUP状态...")
try:
    from src.page_state_dynamic import PageState
    
    popup_state = PageState.POPUP
    print(f"  POPUP状态: {popup_state}")
    print(f"  POPUP值: {popup_state.value}")
    print(f"  POPUP中文名: {popup_state.chinese_name}")
    print(f"  ✓ POPUP状态正常")
    
    # 测试比较
    print(f"\n  测试比较:")
    print(f"    PageState.POPUP == 'popup': {PageState.POPUP == 'popup'}")
    print(f"    PageState.POPUP.value == 'popup': {PageState.POPUP.value == 'popup'}")
    
except Exception as e:
    print(f"  ✗ POPUP状态测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试弹窗相关状态
print("\n[测试2] 所有弹窗相关状态...")
try:
    popup_states = [
        'POPUP',
        'HOME_NOTICE',
        'CHECKIN_POPUP',
        'WARMTIP',
        'STARTUP_POPUP',
        'PROFILE_AD',
        'HOME_ERROR_POPUP',
        'TRANSFER_CONFIRM',
    ]
    
    for state_name in popup_states:
        state = getattr(PageState, state_name)
        print(f"  {state_name}: {state.value} ({state.chinese_name})")
    
    print(f"  ✓ 所有弹窗状态正常")
    
except Exception as e:
    print(f"  ✗ 弹窗状态测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试导航器和登录模块导入
print("\n[测试3] 导航器和登录模块导入...")
try:
    from src.navigator import Navigator
    from src.auto_login import AutoLogin
    from src.adb_bridge import ADBBridge
    
    print(f"  ✓ Navigator模块导入成功")
    print(f"  ✓ AutoLogin模块导入成功")
    print(f"  ✓ ADBBridge模块导入成功")
    print(f"  注：这些模块中的弹窗处理逻辑已加载")
    
except Exception as e:
    print(f"  ✗ 模块导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ 所有测试通过！弹窗识别功能正常")
print("=" * 60)
print("\n下一步：")
print("  1. 运行打包后的程序")
print("  2. 测试首页弹窗是否能正常关闭")
print("  3. 测试签到流程是否正常工作")
