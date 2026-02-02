"""
测试区域OCR识别
"""
import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adb_bridge import ADBBridge
from src.profile_reader import ProfileReader


async def test_region_ocr():
    """测试区域OCR识别"""
    print("=" * 60)
    print("测试区域OCR识别")
    print("=" * 60)
    
    # 初始化ADB
    adb_path = r"D:\Program Files\Netease\MuMu\nx_device\12.0\shell\adb.exe"
    adb = ADBBridge(adb_path=adb_path)
    
    # 初始化ProfileReader
    reader = ProfileReader(adb)
    
    # 使用MuMu模拟器实例0
    device_id = "127.0.0.1:16384"
    print(f"\n使用设备: {device_id}")
    
    try:
        print("\n开始获取个人资料...")
        profile = await reader.get_full_profile(device_id)
        
        print("\n" + "=" * 60)
        print("识别结果:")
        print("=" * 60)
        print(f"昵称: {profile.get('nickname')}")
        print(f"用户ID: {profile.get('user_id')}")
        print(f"手机号: {profile.get('phone')}")
        print(f"余额: {profile.get('balance')}")
        print(f"积分: {profile.get('points')}")
        print(f"抵扣券: {profile.get('vouchers')}")
        print(f"优惠券: {profile.get('coupons')}")
        print("=" * 60)
        
        # 检查识别结果
        success_count = sum(1 for v in profile.values() if v is not None)
        total_count = len(profile)
        
        print(f"\n识别成功率: {success_count}/{total_count} ({success_count*100//total_count}%)")
        
        if profile.get('points') is not None:
            print(f"✓ 积分识别成功: {profile.get('points')}")
        else:
            print("✗ 积分识别失败")
        
        if profile.get('vouchers') is not None:
            print(f"✓ 抵扣券识别成功: {profile.get('vouchers')}")
        else:
            print("✗ 抵扣券识别失败")
        
        if profile.get('coupons') is not None:
            print(f"✓ 优惠券识别成功: {profile.get('coupons')}")
        else:
            print("✗ 优惠券识别失败")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_region_ocr())
