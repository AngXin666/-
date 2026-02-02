"""
测试点击勾选框小圆圈
通过状态切换验证坐标有效性
"""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.adb_bridge import ADBBridge
import win32api
import win32con
import time

async def test():
    adb = ADBBridge()
    await adb.connect('127.0.0.1:5555')
    
    print("="*60)
    print("测试点击勾选框小圆圈")
    print("="*60)
    print("\n使用说明：")
    print("- 点击后观察屏幕状态")
    print("- 如果已勾选：按鼠标左键")
    print("- 如果未勾选：按鼠标右键")
    print("- 按 ESC 键跳过当前测试")
    print("\n原理：")
    print("- 如果坐标有效，点击会切换勾选状态")
    print("- 如果坐标无效，状态不会改变")
    print("\n请确保当前在登录页面，且协议框未勾选\n")
    
    input("按回车开始测试...")
    
    # 根据OCR识别的结果，勾选框位置
    checkbox_coords = [
        (64, 590),   # OCR计算的位置
        (65, 627),   # 手动记录的位置
        (60, 590),   # OCR位置左侧
        (68, 590),   # OCR位置右侧
        (64, 585),   # OCR位置上方
        (64, 595),   # OCR位置下方
    ]
    
    successful_coords = []
    expected_state = False  # 期望的状态：False=未勾选，True=已勾选
    
    for i, (x, y) in enumerate(checkbox_coords):
        print(f"\n{'='*60}")
        print(f"【测试 {i+1}/{len(checkbox_coords)}】点击坐标 ({x}, {y})")
        print(f"当前期望状态: {'未勾选' if not expected_state else '已勾选'}")
        print(f"点击后期望状态: {'已勾选' if not expected_state else '未勾选'}")
        print(f"{'='*60}")
        
        await adb.tap('127.0.0.1:5555', x, y)
        print(f"✓ 已点击")
        print(f"\n请观察屏幕，当前应该是: {'已勾选' if not expected_state else '未勾选'}")
        print("  - 已勾选？按鼠标左键")
        print("  - 未勾选？按鼠标右键")
        print("  - 跳过？按 ESC 键")
        
        # 等待用户按键
        actual_state = None
        while True:
            # 检查左键
            if win32api.GetAsyncKeyState(win32con.VK_LBUTTON) & 0x8000:
                actual_state = True  # 已勾选
                time.sleep(0.5)  # 防抖
                break
            
            # 检查右键
            if win32api.GetAsyncKeyState(win32con.VK_RBUTTON) & 0x8000:
                actual_state = False  # 未勾选
                time.sleep(0.5)  # 防抖
                break
            
            # 检查ESC键
            if win32api.GetAsyncKeyState(win32con.VK_ESCAPE) & 0x8000:
                actual_state = None  # 跳过
                time.sleep(0.5)  # 防抖
                break
            
            await asyncio.sleep(0.05)
        
        # 判断结果
        if actual_state is None:
            print("⏭️  跳过")
        elif actual_state == (not expected_state):
            # 状态正确切换了
            print(f"✅ 成功！状态从 {'未勾选' if expected_state else '已勾选'} 切换到 {'已勾选' if actual_state else '未勾选'}")
            successful_coords.append((x, y))
            expected_state = actual_state  # 更新期望状态
        else:
            # 状态没有切换
            print(f"❌ 失败！状态没有改变，仍然是 {'已勾选' if actual_state else '未勾选'}")
            print(f"   说明坐标 ({x}, {y}) 无效")
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)
    
    if successful_coords:
        print(f"\n✅ 有效的坐标（共 {len(successful_coords)} 个）：")
        for x, y in successful_coords:
            print(f"  - ({x}, {y})")
        
        if len(successful_coords) > 1:
            avg_x = sum(x for x, y in successful_coords) // len(successful_coords)
            avg_y = sum(y for x, y in successful_coords) // len(successful_coords)
            print(f"\n建议使用平均坐标: ({avg_x}, {avg_y})")
        elif len(successful_coords) == 1:
            print(f"\n建议使用坐标: {successful_coords[0]}")
    else:
        print("\n❌ 没有有效的坐标")
        print("建议使用 record_coords_simple.py 重新记录")

asyncio.run(test())
