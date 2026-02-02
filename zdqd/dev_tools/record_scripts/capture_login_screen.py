"""截取当前登录页面"""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.adb_bridge import ADBBridge
from src.screen_capture import ScreenCapture

async def capture():
    adb = ADBBridge()
    await adb.connect('127.0.0.1:5555')
    screen = ScreenCapture(adb)
    img = await screen.capture('127.0.0.1:5555')
    if img is not None:
        import cv2
        cv2.imwrite('login_current_state.png', img)
        print("✓ 截图已保存: login_current_state.png")
    else:
        print("❌ 截图失败")

asyncio.run(capture())
