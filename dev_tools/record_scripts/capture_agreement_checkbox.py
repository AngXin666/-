"""
截取协议勾选框区域，用于创建模板
"""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.adb_bridge import ADBBridge
from src.screen_capture import ScreenCapture
import cv2

async def capture():
    adb = ADBBridge()
    await adb.connect('127.0.0.1:5555')
    screen = ScreenCapture(adb)
    
    print("="*60)
    print("协议勾选框模板截取工具")
    print("="*60)
    print("\n请确保当前在登录页面")
    print("程序将截取整个屏幕，然后你需要手动裁剪协议勾选框区域")
    print("\n" + "="*60)
    
    # 截取屏幕
    img = await screen.capture('127.0.0.1:5555')
    if img is None:
        print("❌ 截图失败")
        return
    
    # 保存完整截图
    cv2.imwrite('login_full_screen.png', img)
    print("✓ 完整截图已保存: login_full_screen.png")
    
    print("\n请使用图片编辑工具（如画图）打开 login_full_screen.png")
    print("裁剪出协议勾选框区域（包含小方框和文字）")
    print("保存为: templates/协议勾选框.png")
    print("\n建议裁剪区域大小: 约 100x30 像素")

asyncio.run(capture())
