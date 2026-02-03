"""
测试个人资料YOLO检测修复
Test Profile YOLO Detection Fix
"""

import asyncio
import sys
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.adb_bridge import ADBBridge
from src.page_detector_integrated import PageDetectorIntegrated
from src.profile_reader import ProfileReader


async def test_profile_yolo():
    """测试个人资料YOLO检测"""
    print("=" * 60)
    print("测试个人资料YOLO检测修复")
    print("=" * 60)
    
    # 初始化ADB
    adb = ADBBridge()
    
    # 获取设备列表
    devices = await adb.list_devices()
    if not devices:
        print("❌ 未找到设备")
        return
    
    device_id = devices[0]
    print(f"\n✓ 使用设备: {device_id}")
    
    # 初始化整合检测器
    print("\n初始化整合检测器...")
    detector = PageDetectorIntegrated(adb)
    
    # 初始化ProfileReader（传入整合检测器）
    print("初始化ProfileReader...")
    profile_reader = ProfileReader(adb, yolo_detector=detector)
    
    # 检查检测器初始化状态
    print(f"\n检测器状态:")
    print(f"  - _integrated_detector: {profile_reader._integrated_detector is not None}")
    print(f"  - _yolo_detector: {profile_reader._yolo_detector is not None}")
    
    # 先测试整合检测器本身
    print("\n" + "=" * 60)
    print("测试整合检测器")
    print("=" * 60)
    
    try:
        result = await detector.detect_page(device_id, use_cache=False, detect_elements=True)
        print(f"\n页面检测结果:")
        print(f"  - 页面类型: {result.state.chinese_name}")
        print(f"  - 置信度: {result.confidence:.2%}")
        print(f"  - 检测到的元素数量: {len(result.elements)}")
        print(f"  - 使用的YOLO模型: {result.yolo_model_used}")
        
        if result.elements:
            print(f"\n元素详情:")
            for elem in result.elements:
                print(f"  - {elem.class_name}: 置信度={elem.confidence:.2f}, bbox={elem.bbox}")
        else:
            print(f"\n⚠️ 未检测到任何元素")
            
    except Exception as e:
        print(f"\n❌ 整合检测器测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试获取个人资料
    print("\n" + "=" * 60)
    print("测试获取完整个人资料")
    print("=" * 60)
    
    try:
        profile = await profile_reader.get_full_profile(device_id)
        
        print("\n获取结果:")
        print(f"  - 昵称: {profile.get('nickname')}")
        print(f"  - 用户ID: {profile.get('user_id')}")
        print(f"  - 手机号: {profile.get('phone')}")
        print(f"  - 余额: {profile.get('balance')}")
        print(f"  - 积分: {profile.get('points')}")
        print(f"  - 抵扣券: {profile.get('vouchers')}")
        print(f"  - 优惠券: {profile.get('coupons')}")
        
        # 检查是否成功获取数据
        success_count = sum(1 for v in profile.values() if v is not None)
        total_count = len(profile)
        
        print(f"\n成功率: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
        
        if success_count >= 4:
            print("\n✓ 测试通过：成功获取大部分数据")
        else:
            print("\n⚠️ 测试警告：获取数据不完整")
            
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    asyncio.run(test_profile_yolo())
