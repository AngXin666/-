"""
测试协议勾选框和登录按钮
"""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.adb_bridge import ADBBridge

async def test():
    adb = ADBBridge()
    await adb.connect('127.0.0.1:5555')
    
    print("="*60)
    print("测试协议勾选框和登录按钮")
    print("="*60)
    print("\n请确保：")
    print("1. 当前在登录页面")
    print("2. 已经输入了账号密码")
    print("\n" + "="*60)
    
    # 测试1：点击协议勾选框
    print("\n【测试1】点击协议勾选框 (65, 627)")
    print("请观察屏幕，看协议是否被勾选")
    await adb.tap('127.0.0.1:5555', 65, 627)
    await asyncio.sleep(2)
    
    response = input("\n协议是否被勾选了？(y/n): ")
    if response.lower() != 'y':
        print("\n协议没有勾选上，请使用 record_coords_simple.py 重新记录协议勾选框坐标")
        print("记录时，请点击协议勾选框的小方框中心位置")
        return
    
    print("\n✓ 协议勾选成功")
    
    # 测试2：点击登录按钮
    print("\n【测试2】点击登录按钮 (257, 745)")
    print("请观察屏幕，看是否触发登录")
    await adb.tap('127.0.0.1:5555', 257, 745)
    await asyncio.sleep(3)
    
    response = input("\n是否触发了登录？(y/n): ")
    if response.lower() != 'y':
        print("\n登录按钮没有反应，请使用 record_coords_simple.py 重新记录登录按钮坐标")
        print("记录时，请点击青色的'登录'按钮中心位置")
        return
    
    print("\n✓ 登录按钮有效")
    print("\n" + "="*60)
    print("测试完成！坐标都正确")
    print("="*60)

asyncio.run(test())
