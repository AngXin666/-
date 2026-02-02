"""
记录鼠标点击位置工具
用于手动点击"我的"按钮并记录坐标作为备用
"""

import time
import sys


def main():
    print("=" * 60)
    print("记录鼠标点击位置工具")
    print("=" * 60)
    
    try:
        import pyautogui
        import win32gui
    except ImportError as e:
        print(f"\n❌ 缺少必要的库: {e}")
        print("\n请安装所需库:")
        print("  pip install pyautogui pywin32")
        return
    
    # 查找 MuMu 窗口
    print("\n1. 查找 MuMu 模拟器窗口...")
    
    def find_mumu_window():
        def callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if "MuMu" in title or "mumu" in title.lower():
                    windows.append((hwnd, title))
            return True
        
        windows = []
        win32gui.EnumWindows(callback, windows)
        return windows
    
    windows = find_mumu_window()
    
    if not windows:
        print("❌ 未找到 MuMu 模拟器窗口")
        print("   请确保 MuMu 模拟器正在运行")
        return
    
    print(f"✓ 找到 {len(windows)} 个 MuMu 窗口:")
    for i, (hwnd, title) in enumerate(windows, 1):
        print(f"   {i}. {title}")
    
    # 使用第一个窗口
    hwnd, title = windows[0]
    print(f"\n使用窗口: {title}")
    
    # 获取窗口位置
    rect = win32gui.GetWindowRect(hwnd)
    window_x = rect[0]
    window_y = rect[1]
    window_width = rect[2] - rect[0]
    window_height = rect[3] - rect[1]
    
    print(f"窗口位置: ({window_x}, {window_y})")
    print(f"窗口尺寸: {window_width}x{window_height}")
    
    print("\n" + "=" * 60)
    print("实时显示鼠标位置")
    print("=" * 60)
    print("\n操作说明:")
    print("1. 确保 MuMu 模拟器窗口可见")
    print("2. 确保溪盟商城应用已打开并在首页")
    print("3. 将鼠标移动到'我的'按钮上")
    print("4. 看到正确的坐标后，按 Ctrl+C 记录")
    print("\n开始实时显示鼠标位置...")
    print("(按 Ctrl+C 记录当前位置)\n")
    
    try:
        while True:
            # 获取鼠标位置
            screen_x, screen_y = pyautogui.position()
            
            # 计算相对于窗口的坐标
            rel_x = screen_x - window_x
            rel_y = screen_y - window_y
            
            # 考虑窗口标题栏和边框
            # MuMu 模拟器标题栏高度约 30 像素，边框约 8 像素
            title_bar_height = 30
            border_width = 8
            
            # 调整坐标
            content_x = rel_x - border_width
            content_y = rel_y - title_bar_height
            
            # 计算内容区域尺寸
            content_width = window_width - 2 * border_width
            content_height = window_height - title_bar_height - border_width
            
            # 转换到 540x960 坐标系
            if content_width > 0 and content_height > 0:
                emulator_x = int((content_x / content_width) * 540)
                emulator_y = int((content_y / content_height) * 960)
                
                # 实时显示（覆盖上一行）
                print(f"\r屏幕: ({screen_x:4d}, {screen_y:4d}) -> 模拟器 (540x960): ({emulator_x:3d}, {emulator_y:3d})", end='', flush=True)
            
            time.sleep(0.05)  # 50ms 更新一次
            
    except KeyboardInterrupt:
        # 记录当前鼠标位置
        screen_x, screen_y = pyautogui.position()
        
        # 计算模拟器坐标
        rel_x = screen_x - window_x
        rel_y = screen_y - window_y
        content_x = rel_x - border_width
        content_y = rel_y - title_bar_height
        content_width = window_width - 2 * border_width
        content_height = window_height - title_bar_height - border_width
        
        emulator_x = int((content_x / content_width) * 540)
        emulator_y = int((content_y / content_height) * 960)
        
        print("\n\n" + "=" * 60)
        print("记录完成！")
        print("=" * 60)
        print(f"\n屏幕坐标: ({screen_x}, {screen_y})")
        print(f"模拟器坐标 (540x960): ({emulator_x}, {emulator_y})")
        
        print(f"\n建议在 navigator.py 中添加备用坐标:")
        print(f"  TAB_MY_BACKUP = ({emulator_x}, {emulator_y})  # 手动记录的备用坐标")
        print(f"\n然后在 OCR 识别失败时使用此坐标")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
