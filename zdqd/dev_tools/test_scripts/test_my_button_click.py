"""
测试"我的"按钮点击是否有效
"""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.adb_bridge import ADBBridge
from src.page_detector_hybrid import PageDetectorHybrid

async def test_my_button():
    adb = ADBBridge()
    await adb.connect('127.0.0.1:5555')
    detector = PageDetectorHybrid(adb)
    
    # 检测当前页面
    print("【步骤1】检测当前页面")
    result = await detector.detect_page('127.0.0.1:5555', use_ocr=True)
    print(f"当前页面: {result.state.value}")
    print(f"详情: {result.details}")
    
    # 点击"我的"按钮（MuMu坐标）
    print("\n【步骤2】点击'我的'按钮")
    print("使用坐标: (450, 920)")
    await adb.tap('127.0.0.1:5555', 450, 920)
    
    # 等待3秒
    print("等待3秒...")
    await asyncio.sleep(3)
    
    # 再次检测页面
    print("\n【步骤3】检测点击后的页面")
    result = await detector.detect_page('127.0.0.1:5555', use_ocr=True)
    print(f"当前页面: {result.state.value}")
    print(f"详情: {result.details}")
    
    # 如果还在首页，说明坐标不对
    if result.state.value == 'home':
        print("\n❌ 点击后仍在首页，'我的'按钮坐标可能不正确")
        print("建议使用 record_coords_simple.py 重新记录坐标")
    elif result.state.value in ['profile', 'profile_logged']:
        print("\n✅ 成功跳转到个人页面")
    else:
        print(f"\n⚠️  跳转到了其他页面: {result.state.value}")

asyncio.run(test_my_button())
