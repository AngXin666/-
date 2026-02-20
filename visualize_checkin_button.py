"""可视化签到按钮点击位置"""
import asyncio
import sys
sys.path.insert(0, 'src')

from adb_bridge import ADBBridge
from PIL import Image, ImageDraw
from io import BytesIO

async def visualize_click_position():
    """截图并标注点击位置"""
    # 使用MuMu模拟器的ADB路径
    adb_path = r"D:\Program Files\Netease\MuMu\nx_device\12.0\shell\adb.exe"
    adb = ADBBridge(adb_path=adb_path)
    
    # 使用固定的设备ID
    device_id = "127.0.0.1:16384"
    print(f"使用设备: {device_id}")
    
    # 连接设备
    print("连接设备...")
    connected = await adb.connect(device_id)
    if not connected:
        print("❌ 设备连接失败")
        return
    print("✓ 设备已连接")
    
    # 截图
    print("截图中...")
    screenshot_data = await adb.screencap(device_id)
    if not screenshot_data:
        print("❌ 截图失败")
        return
    
    # 打开图片
    image = Image.open(BytesIO(screenshot_data))
    print(f"✓ 截图成功: {image.width}x{image.height}")
    
    # 签到按钮坐标
    CHECKIN_BUTTON = (475, 550)
    
    # 在图片上标注点击位置
    draw = ImageDraw.Draw(image)
    
    # 画一个红色十字标记
    x, y = CHECKIN_BUTTON
    size = 20
    draw.line([(x - size, y), (x + size, y)], fill='red', width=3)
    draw.line([(x, y - size), (x, y + size)], fill='red', width=3)
    
    # 画一个红色圆圈
    draw.ellipse([(x - 10, y - 10), (x + 10, y + 10)], outline='red', width=3)
    
    # 添加坐标文字
    draw.text((x + 15, y - 15), f"({x}, {y})", fill='red')
    
    # 保存图片
    output_path = "checkin_button_position.png"
    image.save(output_path)
    print(f"✓ 已保存标注图片: {output_path}")
    print(f"点击位置: {CHECKIN_BUTTON}")
    
    # 打开图片
    import os
    os.startfile(output_path)

if __name__ == "__main__":
    asyncio.run(visualize_click_position())
