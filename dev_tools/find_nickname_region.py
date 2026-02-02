"""
查找昵称和用户ID的像素区域工具

使用方法：
1. 打开模拟器，进入个人页面
2. 运行此脚本
3. 在弹出的窗口中，用鼠标框选昵称区域
4. 记录显示的坐标，更新到 profile_reader.py 的 REGIONS 中
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from PIL import Image, ImageDraw, ImageFont
    import tkinter as tk
    from tkinter import messagebox
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

from src.adb_bridge import ADBBridge
from src.emulator_controller import EmulatorController
from io import BytesIO


class RegionSelector:
    """区域选择器"""
    
    def __init__(self, image: Image.Image):
        self.image = image
        self.root = tk.Tk()
        self.root.title("选择昵称和用户ID区域")
        
        # 缩放图片以适应屏幕
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        img_width, img_height = image.size
        scale = min(screen_width * 0.8 / img_width, screen_height * 0.8 / img_height)
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        self.scale = scale
        self.display_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 创建画布
        self.canvas = tk.Canvas(self.root, width=new_width, height=new_height)
        self.canvas.pack()
        
        # 显示图片
        from PIL import ImageTk
        self.photo = ImageTk.PhotoImage(self.display_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        
        # 绑定鼠标事件
        self.start_x = None
        self.start_y = None
        self.rect = None
        
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
        # 说明标签
        self.label = tk.Label(self.root, text="用鼠标框选昵称区域，然后松开鼠标", font=("Arial", 12))
        self.label.pack(pady=10)
        
        # 当前定义的区域
        self.regions = {
            'nickname': (50, 80, 400, 150),      # 昵称区域（顶部，ID上方）
            'user_id': (50, 150, 400, 200),      # 用户ID区域（昵称下方）
            'balance': (50, 200, 150, 300),      # 余额数字区域
            'points': (180, 200, 260, 300),      # 积分数字区域
            'vouchers': (315, 200, 390, 300),    # 抵扣券数字区域
            'coupons': (410, 200, 490, 300),     # 优惠券数字区域
        }
        
        # 绘制当前定义的区域
        self.draw_regions()
    
    def draw_regions(self):
        """绘制当前定义的区域"""
        colors = {
            'nickname': 'red',
            'user_id': 'blue',
            'balance': 'green',
            'points': 'orange',
            'vouchers': 'purple',
            'coupons': 'brown'
        }
        
        for name, (x1, y1, x2, y2) in self.regions.items():
            # 转换为显示坐标
            dx1 = int(x1 * self.scale)
            dy1 = int(y1 * self.scale)
            dx2 = int(x2 * self.scale)
            dy2 = int(y2 * self.scale)
            
            color = colors.get(name, 'gray')
            self.canvas.create_rectangle(dx1, dy1, dx2, dy2, outline=color, width=2)
            self.canvas.create_text(dx1 + 5, dy1 + 5, text=name, anchor=tk.NW, fill=color, font=("Arial", 10, "bold"))
    
    def on_press(self, event):
        """鼠标按下"""
        self.start_x = event.x
        self.start_y = event.y
        
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline='yellow', width=3
        )
    
    def on_drag(self, event):
        """鼠标拖动"""
        if self.rect:
            self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)
    
    def on_release(self, event):
        """鼠标松开"""
        end_x = event.x
        end_y = event.y
        
        # 转换为原始图片坐标
        orig_x1 = int(min(self.start_x, end_x) / self.scale)
        orig_y1 = int(min(self.start_y, end_y) / self.scale)
        orig_x2 = int(max(self.start_x, end_x) / self.scale)
        orig_y2 = int(max(self.start_y, end_y) / self.scale)
        
        # 显示坐标
        region_str = f"({orig_x1}, {orig_y1}, {orig_x2}, {orig_y2})"
        messagebox.showinfo(
            "区域坐标",
            f"选中的区域坐标（原始图片）:\n{region_str}\n\n"
            f"请将此坐标更新到 src/profile_reader.py 的 REGIONS 字典中"
        )
        
        print(f"\n选中的区域坐标: {region_str}")
        print(f"更新到 REGIONS 字典:")
        print(f"'nickname': {region_str},  # 昵称区域")
    
    def run(self):
        """运行"""
        self.root.mainloop()


async def main():
    """主函数"""
    if not HAS_DEPS:
        print("❌ 缺少依赖库，请安装: pip install pillow")
        return
    
    print("=" * 60)
    print("昵称和用户ID区域查找工具")
    print("=" * 60)
    
    # 初始化模拟器控制器
    emulator_path = input("请输入模拟器路径（直接回车使用默认）: ").strip()
    if not emulator_path:
        emulator_path = "D:\\Program Files\\MuMuPlayer-12.0"
    
    print(f"\n使用模拟器路径: {emulator_path}")
    
    controller = EmulatorController(emulator_path)
    
    # 获取设备列表
    print("\n正在获取设备列表...")
    devices = await controller.list_devices()
    
    if not devices:
        print("❌ 未找到设备")
        return
    
    print(f"✅ 找到 {len(devices)} 个设备:")
    for i, device in enumerate(devices):
        print(f"  {i+1}. {device}")
    
    # 选择设备
    if len(devices) == 1:
        device_id = devices[0]
        print(f"\n自动选择设备: {device_id}")
    else:
        choice = input(f"\n请选择设备 (1-{len(devices)}): ").strip()
        try:
            device_id = devices[int(choice) - 1]
        except (ValueError, IndexError):
            print("❌ 无效的选择")
            return
    
    # 初始化ADB
    adb = ADBBridge(controller)
    
    # 截图
    print(f"\n正在截图...")
    print("提示：请确保模拟器已打开个人页面")
    
    screenshot_data = await adb.screencap(device_id)
    if not screenshot_data:
        print("❌ 截图失败")
        return
    
    image = Image.open(BytesIO(screenshot_data))
    print(f"✅ 截图成功，分辨率: {image.size}")
    
    # 启动区域选择器
    print("\n启动区域选择器...")
    print("说明：")
    print("1. 红色框：当前定义的昵称区域")
    print("2. 蓝色框：当前定义的用户ID区域")
    print("3. 用鼠标框选正确的昵称区域")
    print("4. 记录显示的坐标，更新到 profile_reader.py")
    
    selector = RegionSelector(image)
    selector.run()


if __name__ == "__main__":
    asyncio.run(main())
