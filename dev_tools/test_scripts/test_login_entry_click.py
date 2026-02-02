"""
测试登录入口点击是否有效
"""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.adb_bridge import ADBBridge
from src.page_detector_hybrid import PageDetectorHybrid

async def test_login_entry():
    adb = ADBBridge()
    await adb.connect('127.0.0.1:5555')
    detector = PageDetectorHybrid(adb)
    
    # 先导航到个人页面
    print("【步骤1】导航到个人页面")
    await adb.tap('127.0.0.1:5555', 450, 920)  # 点击"我的"
    await asyncio.sleep(3)
    
    result = await detector.detect_page('127.0.0.1:5555', use_ocr=True)
    print(f"当前页面: {result.state.value}")
    print(f"详情: {result.details}")
    
    if result.state.value != 'profile':
        print("❌ 未能到达个人页面")
        return
    
    # 点击登录入口
    print("\n【步骤2】点击登录入口")
    print("使用坐标: (270, 200)")
    await adb.tap('127.0.0.1:5555', 270, 200)
    
    # 等待3秒
    print("等待3秒...")
    await asyncio.sleep(3)
    
    # 检测页面
    print("\n【步骤3】检测点击后的页面")
    result = await detector.detect_page('127.0.0.1:5555', use_ocr=True)
    print(f"当前页面: {result.state.value}")
    print(f"详情: {result.details}")
    
    if result.state.value == 'login':
        print("\n✅ 成功进入登录页面")
    elif result.state.value == 'profile':
        print("\n❌ 点击后仍在个人页面，登录入口坐标可能不正确")
        print("建议使用 record_coords_simple.py 重新记录坐标")
    else:
        print(f"\n⚠️  跳转到了其他页面: {result.state.value}")

asyncio.run(test_login_entry())
