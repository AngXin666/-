"""
简单的坐标记录工具
点击鼠标左键记录坐标，按 ESC 退出
"""
import win32gui
import win32api
import win32con
import time

print("=" * 60)
print("坐标记录工具")
print("=" * 60)
print("说明：")
print("1. 将鼠标移动到 MuMu 模拟器窗口内")
print("2. 点击鼠标左键记录坐标")
print("3. 按 ESC 键退出")
print("=" * 60)
print()

# 查找 MuMu 窗口
def find_mumu_window():
    """查找 MuMu 模拟器窗口"""
    window_titles = [
        "MuMu安卓设备",
        "MuMu模拟器12",
        "MuMu Player 12",
        "MuMuPlayer",
    ]
    
    for title in window_titles:
        hwnd = win32gui.FindWindow(None, title)
        if hwnd:
            return hwnd
    return None

# 获取窗口内坐标
def get_window_coords(hwnd):
    """获取窗口内的相对坐标"""
    # 获取鼠标屏幕坐标
    cursor_pos = win32api.GetCursorPos()
    
    # 获取窗口位置
    rect = win32gui.GetWindowRect(hwnd)
    window_x = rect[0]
    window_y = rect[1]
    
    # 获取客户区位置（去除标题栏和边框）
    client_rect = win32gui.GetClientRect(hwnd)
    client_point = win32gui.ClientToScreen(hwnd, (0, 0))
    
    # 计算窗口内坐标
    rel_x = cursor_pos[0] - client_point[0]
    rel_y = cursor_pos[1] - client_point[1]
    
    return rel_x, rel_y

# 查找窗口
hwnd = find_mumu_window()
if not hwnd:
    print("❌ 未找到 MuMu 模拟器窗口")
    print("请确保 MuMu 模拟器正在运行")
    input("按回车键退出...")
    exit(1)

print(f"✓ 找到 MuMu 模拟器窗口")
print()

coords = []
last_click_time = 0

try:
    while True:
        # 检查 ESC 键
        if win32api.GetAsyncKeyState(win32con.VK_ESCAPE) & 0x8000:
            break
        
        # 检查鼠标左键
        if win32api.GetAsyncKeyState(win32con.VK_LBUTTON) & 0x8000:
            current_time = time.time()
            # 防抖：0.3秒内只记录一次
            if current_time - last_click_time > 0.3:
                x, y = get_window_coords(hwnd)
                coords.append((x, y))
                print(f"✓ 记录坐标 #{len(coords)}: ({x}, {y})")
                last_click_time = current_time
        
        time.sleep(0.05)

except KeyboardInterrupt:
    pass

print()
print("=" * 60)
print("记录完成！")
print("=" * 60)

if coords:
    print(f"\n共记录 {len(coords)} 个坐标：")
    for i, (x, y) in enumerate(coords, 1):
        print(f"  {i}. ({x}, {y})")
    
    # 计算平均值
    if len(coords) > 1:
        avg_x = sum(x for x, y in coords) // len(coords)
        avg_y = sum(y for x, y in coords) // len(coords)
        print(f"\n平均坐标: ({avg_x}, {avg_y})")
        print(f"\n建议使用: COORD = ({avg_x}, {avg_y})")
else:
    print("\n未记录任何坐标")

print("\n程序将在3秒后自动退出...")
time.sleep(3)
