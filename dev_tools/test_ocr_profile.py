"""
测试OCR识别个人资料页面
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from src.adb_bridge import ADBBridge
from src.profile_reader import ProfileReader

async def main():
    # 初始化ADB
    adb = ADBBridge()
    
    # 获取设备列表
    result = await adb.shell("127.0.0.1:16384", "echo test")
    if "error" in result.lower() or not result.strip():
        print("❌ 未找到设备")
        return
    
    device_id = "127.0.0.1:16384"
    print(f"✓ 使用设备: {device_id}")
    
    # 初始化ProfileReader
    profile_reader = ProfileReader(adb)
    
    # 获取个人资料
    print("\n开始获取个人资料...")
    print("=" * 80)
    
    result = await profile_reader.get_full_profile(device_id)
    
    print("\n" + "=" * 80)
    print("获取结果：")
    print("=" * 80)
    print(f"昵称: {result.get('nickname')}")
    print(f"用户ID: {result.get('user_id')}")
    print(f"手机号: {result.get('phone')}")
    print(f"余额: {result.get('balance')}")
    print(f"积分: {result.get('points')}")
    print(f"抵扣券: {result.get('vouchers')}")
    print("=" * 80)

if __name__ == '__main__':
    asyncio.run(main())
