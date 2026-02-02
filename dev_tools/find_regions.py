"""
查找并标注区域坐标的工具脚本
"""
import asyncio
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adb_bridge import ADBBridge
from io import BytesIO

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("❌ 需要安装 Pillow: pip install Pillow")
    sys.exit(1)

async def capture_and_annotate():
    """截图并标注坐标网格"""
    print("=" * 60)
    print("区域坐标查找工具")
    print("=" * 60)
    
    # 初始化ADB
    adb_path = r"D:\Program Files\Netease\MuMu\nx_device\12.0\shell\adb.exe"
    adb = ADBBridge(adb_path=adb_path)
    
    # 使用MuMu模拟器实例0
    device_id = "127.0.0.1:16384"
    print(f"\n使用设备: {device_id}")
    
    try:
        # 截图
        print("\n正在截图...")
        screenshot_data = await adb.screencap(device_id)
        if not screenshot_data:
            print("❌ 截图失败")
            return
        
        # 打开图像
        image = Image.open(BytesIO(screenshot_data))
        width, height = image.size
        print(f"✓ 截图成功: {width}x{height}")
        
        # 创建绘图对象
        draw = ImageDraw.Draw(image)
        
        # 尝试加载字体（如果失败则使用默认字体）
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        # 绘制网格线（每50像素一条）
        grid_step = 50
        
        # 垂直线
        for x in range(0, width, grid_step):
            draw.line([(x, 0), (x, height)], fill=(255, 0, 0, 128), width=1)
            # 标注X坐标
            if x % 100 == 0:
                draw.text((x + 2, 2), str(x), fill=(255, 0, 0), font=font)
        
        # 水平线
        for y in range(0, height, grid_step):
            draw.line([(0, y), (width, y)], fill=(255, 0, 0, 128), width=1)
            # 标注Y坐标
            if y % 100 == 0:
                draw.text((2, y + 2), str(y), fill=(255, 0, 0), font=font)
        
        # 标注当前定义的区域（只识别数字，不包含标签）
        regions = {
            'balance': (50, 200, 150, 300),      # 余额数字区域
            'points': (180, 200, 260, 300),      # 积分数字区域
            'vouchers': (315, 200, 390, 300),    # 抵扣券数字区域
            'coupons': (410, 200, 490, 300),     # 优惠券数字区域
        }
        
        colors = {
            'balance': (0, 255, 0),      # 绿色
            'points': (0, 0, 255),       # 蓝色
            'vouchers': (255, 165, 0),   # 橙色
            'coupons': (255, 0, 255),    # 紫色
        }
        
        labels = {
            'balance': '余额',
            'points': '积分',
            'vouchers': '抵扣券',
            'coupons': '优惠券',
        }
        
        print("\n当前定义的区域:")
        for name, (x1, y1, x2, y2) in regions.items():
            color = colors[name]
            label = labels[name]
            
            # 绘制矩形框
            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
            
            # 绘制标签
            draw.text((x1 + 5, y1 + 5), f"{label}\n({x1},{y1})\n({x2},{y2})", 
                     fill=color, font=font)
            
            print(f"  {label}: ({x1}, {y1}, {x2}, {y2})")
        
        # 保存标注后的图像
        output_dir = Path(__file__).parent / "test_screenshots"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "region_coordinates.png"
        
        image.save(output_path)
        print(f"\n✓ 标注图像已保存: {output_path}")
        print("\n说明:")
        print("  - 红色网格线: 每50像素一条")
        print("  - 红色数字: 坐标值（每100像素标注一次）")
        print("  - 彩色矩形框: 当前定义的识别区域")
        print("\n请查看图像，然后告诉我正确的区域坐标！")
        print("格式: 区域名 (x1, y1, x2, y2)")
        print("例如: 积分 (180, 220, 280, 270)")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(capture_and_annotate())
