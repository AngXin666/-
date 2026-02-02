import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.adb_bridge import ADBBridge
from src.page_detector_hybrid import PageDetectorHybrid

async def check():
    adb = ADBBridge()
    await adb.connect('127.0.0.1:5555')
    detector = PageDetectorHybrid(adb)
    result = await detector.detect_page('127.0.0.1:5555', use_ocr=True)
    print(f'当前页面: {result.state.value}')
    print(f'详情: {result.details}')

asyncio.run(check())
