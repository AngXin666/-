"""
记录签到弹窗关闭按钮坐标
通过鼠标点击记录模拟器窗口内的坐标
- 左键点击：记录坐标
- 右键点击：退出程序
"""
import sys


def main():
    print("=" * 60)
    print("签到弹窗关闭按钮坐标记录工具")
    print("=" * 60)
    print()
    
    # 检查依赖
    try:
        from pynput import mouse
    except ImportError:
        print("❌ 需要安装 pynput 库")
        print("请运行: pip install pynput")
        return
    
    try:
        import win32gui
    except ImportError:
        print("❌ 需要安装 pywin32 库")
        print("请运行: pip install pywin32")
        return
    
    # 查找MuMu模拟器窗口
    def find_mumu_window():
        def callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if "MuMu" in title or "mumu" in title.lower():
                    windows.append((hwnd, title))
            return True
        
        windows = []
        win32gui.EnumWindows(callback, windows)
        
        if windows:
            hwnd, title = windows[0]
            print(f"✓ 找到模拟器窗口: {title}")
            return hwnd
        return None
    
    # 获取窗口位置
    print("1. 查找MuMu模拟器窗口...")
    emulator_window = find_mumu_window()
    if not emulator_window:
        print("❌ 未找到MuMu模拟器窗口")
        print("请确保MuMu模拟器正在运行")
        return
    
    rect = win32gui.GetWindowRect(emulator_window)
    left, top, right, bottom = rect
    print(f"✓ 窗口位置: ({left}, {top}) - ({right}, {bottom})")
    print(f"  窗口大小: {right - left} x {bottom - top}")
    print()
    
    # 坐标转换函数
    def screen_to_emulator(screen_x, screen_y):
        """将屏幕坐标转换为模拟器内坐标（540x960）"""
        # Windows 窗口边框和标题栏偏移
        border = 8
        titlebar = 31
        
        # 转换为窗口内坐标
        window_x = screen_x - left - border
        window_y = screen_y - top - titlebar
        
        # 计算窗口客户区大小
        client_width = right - left - 2 * border
        client_height = bottom - top - titlebar - border
        
        # 计算缩放比例（模拟器分辨率 540x960）
        scale_x = 540 / client_width
        scale_y = 960 / client_height
        
        # 转换为模拟器内坐标
        emulator_x = int(window_x * scale_x)
        emulator_y = int(window_y * scale_y)
        
        return (emulator_x, emulator_y)
    
    # 记录的坐标
    recorded_coords = []
    
    # 鼠标点击事件处理
    def on_click(x, y, button, pressed):
        if not pressed:
            return True
        
        if button == mouse.Button.left:
            # 左键 - 记录坐标
            coords = screen_to_emulator(x, y)
            emulator_x, emulator_y = coords
            
            # 检查是否在模拟器范围内
            if 0 <= emulator_x <= 540 and 0 <= emulator_y <= 960:
                recorded_coords.append(coords)
                print(f"✓ 记录坐标 #{len(recorded_coords)}: ({emulator_x}, {emulator_y})")
                print(f"  屏幕位置: ({x}, {y})")
            else:
                print(f"⚠️ 坐标超出范围: ({emulator_x}, {emulator_y})")
                print(f"  请点击模拟器窗口内")
            
            return True
        
        elif button == mouse.Button.right:
            # 右键 - 退出
            print("\n右键点击，退出记录...")
            return False
    
    # 提示用户
    print("2. 准备记录坐标")
    print()
    print("操作说明：")
    print("  - 左键点击：记录签到弹窗关闭按钮位置")
    print("  - 右键点击：完成记录并退出")
    print()
    print("请在模拟器窗口中点击签到弹窗的关闭按钮...")
    print("（建议记录3个位置：中心、左偏、右偏）")
    print()
    
    # 启动监听
    with mouse.Listener(on_click=on_click) as listener:
        listener.join()
    
    # 输出结果
    print("\n" + "=" * 60)
    print("记录完成")
    print("=" * 60)
    print()
    
    if not recorded_coords:
        print("未记录任何坐标")
        return
    
    print(f"共记录 {len(recorded_coords)} 个坐标：")
    print()
    for i, (x, y) in enumerate(recorded_coords, 1):
        print(f"  {i}. ({x}, {y})")
    
    print()
    print("=" * 60)
    print("配置代码")
    print("=" * 60)
    print()
    print("请将以下坐标添加到 src/page_detector_hybrid.py 中：")
    print()
    print("# 签到弹窗关闭按钮坐标（MuMu模拟器 540x960）")
    
    if len(recorded_coords) == 1:
        x, y = recorded_coords[0]
        print(f"CHECKIN_POPUP_CLOSE = [")
        print(f"    ({x}, {y}),")
        print(f"    ({x - 5}, {y}),")
        print(f"    ({x + 5}, {y})")
        print(f"]")
    elif len(recorded_coords) >= 3:
        print(f"CHECKIN_POPUP_CLOSE = [")
        for x, y in recorded_coords[:3]:
            print(f"    ({x}, {y}),")
        print(f"]")
    else:
        print(f"CHECKIN_POPUP_CLOSE = [")
        for x, y in recorded_coords:
            print(f"    ({x}, {y}),")
        # 补充到3个
        x, y = recorded_coords[0]
        for i in range(3 - len(recorded_coords)):
            offset = (i + 1) * 5
            print(f"    ({x + offset}, {y}),")
        print(f"]")
    
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback
        traceback.print_exc()
