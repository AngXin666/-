"""
测试登录按钮和协议勾选框坐标
"""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.adb_bridge import ADBBridge

async def test_coords():
    adb = ADBBridge()
    await adb.connect('127.0.0.1:5555')
    
    print("="*60)
    print("测试登录按钮和协议勾选框坐标")
    print("="*60)
    print("\n请确保当前在登录页面，并且已经输入了账号密码")
    print("\n测试步骤：")
    print("1. 点击协议勾选框 (65, 641)")
    print("2. 等待1秒观察是否勾选")
    print("3. 点击登录按钮 (257, 745)")
    print("4. 等待观察是否触发登录")
    print("\n" + "="*60)
    
    input("\n按回车开始测试...")
    
    # 测试协议勾选框
    print("\n【步骤1】点击协议勾选框")
    print("坐标: (65, 641)")
    await adb.tap('127.0.0.1:5555', 65, 641)
    print("✓ 已点击，请观察屏幕上协议是否被勾选")
    await asyncio.sleep(2)
    
    # 测试登录按钮
    print("\n【步骤2】点击登录按钮")
    print("坐标: (257, 745)")
    await adb.tap('127.0.0.1:5555', 257, 745)
    print("✓ 已点击，请观察是否触发登录")
    await asyncio.sleep(3)
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)
    print("\n如果协议没有勾选或登录按钮没有反应，")
    print("请使用 record_coords_simple.py 重新记录坐标")

asyncio.run(test_coords())
