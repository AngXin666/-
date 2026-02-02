"""
手动测试GUI错误显示和勾选状态功能

这个脚本会启动GUI，让用户手动测试：
1. 错误显示功能
2. 勾选状态的保存和加载

测试步骤：
1. 启动GUI
2. 检查表格中是否显示了详细的错误信息（而不是简单的"成功"/"失败"）
3. 勾选/取消勾选一些账号
4. 关闭GUI
5. 重新启动GUI
6. 检查勾选状态是否被正确恢复
"""

import sys
import os
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("=" * 60)
print("GUI功能手动测试")
print("=" * 60)
print()
print("测试内容：")
print("1. 错误显示功能 - 表格的'状态'列应该显示详细的错误信息")
print("2. 勾选状态保存 - 勾选状态应该在重启后保持")
print()
print("测试步骤：")
print("1. 观察表格中的'状态'列，应该显示详细错误（如'登录失败:密码错误'）")
print("2. 勾选或取消勾选一些账号")
print("3. 关闭GUI窗口")
print("4. 重新运行此脚本")
print("5. 检查勾选状态是否被正确恢复")
print()
print("=" * 60)
print()

# 检查配置文件
config_file = Path(".kiro/settings/account_selection.json")
if config_file.exists():
    import json
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    print(f"当前配置文件状态：")
    print(f"  版本: {config.get('version')}")
    print(f"  最后更新: {config.get('last_updated')}")
    print(f"  已保存的勾选状态: {len(config.get('selections', {}))} 个账号")
    if config.get('selections'):
        print(f"  勾选详情:")
        for phone, checked in config['selections'].items():
            status = "✓ 已勾选" if checked else "✗ 未勾选"
            print(f"    {phone}: {status}")
    print()
else:
    print("配置文件不存在，这是第一次运行")
    print()

print("=" * 60)
print("正在启动GUI...")
print("=" * 60)
print()

# 启动GUI
try:
    from gui import AutomationGUI
    import tkinter as tk
    
    root = tk.Tk()
    app = AutomationGUI(root)
    
    print("GUI已启动！")
    print()
    print("请在GUI中测试以下功能：")
    print("1. 查看表格的'状态'列，确认显示详细错误信息")
    print("2. 勾选/取消勾选一些账号")
    print("3. 关闭窗口后重新运行此脚本，验证勾选状态是否保持")
    print()
    
    root.mainloop()
    
    print()
    print("=" * 60)
    print("GUI已关闭")
    print("=" * 60)
    print()
    
    # 显示最新的配置文件状态
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"最新配置文件状态：")
        print(f"  最后更新: {config.get('last_updated')}")
        print(f"  勾选状态: {len(config.get('selections', {}))} 个账号")
        if config.get('selections'):
            print(f"  勾选详情:")
            for phone, checked in config['selections'].items():
                status = "✓ 已勾选" if checked else "✗ 未勾选"
                print(f"    {phone}: {status}")
    
except Exception as e:
    print(f"❌ 启动GUI失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
