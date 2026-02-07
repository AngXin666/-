"""
诊断获取个人资料慢的问题
"""
import asyncio
import time
import sys
sys.path.insert(0, 'src')

from adb_bridge import ADBBridge
from page_detector_integrated import PageDetectorIntegrated
from profile_reader import ProfileReader

async def diagnose():
    """诊断获取个人资料的性能"""
    print("=" * 80)
    print("诊断获取个人资料性能")
    print("=" * 80)
    
    # 初始化
    adb = ADBBridge()
    detector = PageDetectorIntegrated(adb)
    profile_reader = ProfileReader(adb, yolo_detector=detector)
    
    # 获取设备列表
    devices = await adb.list_devices()
    if not devices:
        print("❌ 未找到设备")
        return
    
    device_id = devices[0]
    print(f"✓ 使用设备: {device_id}\n")
    
    # 测试1：检测页面状态
    print("[测试1] 检测页面状态（不检测元素）")
    print("-" * 80)
    start = time.time()
    result1 = await detector.detect_page(device_id, use_cache=False, detect_elements=False)
    time1 = time.time() - start
    print(f"  页面状态: {result1.state.value}")
    print(f"  置信度: {result1.confidence:.2%}")
    print(f"  耗时: {time1:.3f}秒\n")
    
    # 测试2：检测页面状态（检测元素）
    print("[测试2] 检测页面状态（检测元素）")
    print("-" * 80)
    start = time.time()
    result2 = await detector.detect_page(device_id, use_cache=False, detect_elements=True)
    time2 = time.time() - start
    print(f"  页面状态: {result2.state.value}")
    print(f"  置信度: {result2.confidence:.2%}")
    print(f"  检测到元素数: {len(result2.elements)}")
    if result2.elements:
        for elem in result2.elements:
            print(f"    - {elem.class_name}: {elem.bbox}")
    print(f"  耗时: {time2:.3f}秒\n")
    
    # 测试3：获取完整个人资料
    print("[测试3] 获取完整个人资料")
    print("-" * 80)
    start = time.time()
    profile = await profile_reader.get_full_profile(device_id)
    time3 = time.time() - start
    print(f"  昵称: {profile.get('nickname')}")
    print(f"  用户ID: {profile.get('user_id')}")
    print(f"  余额: {profile.get('balance')}")
    print(f"  积分: {profile.get('points')}")
    print(f"  抵扣券: {profile.get('vouchers')}")
    print(f"  优惠券: {profile.get('coupons')}")
    print(f"  耗时: {time3:.3f}秒\n")
    
    # 总结
    print("=" * 80)
    print("性能总结")
    print("=" * 80)
    print(f"  检测页面（无元素）: {time1:.3f}秒")
    print(f"  检测页面（有元素）: {time2:.3f}秒")
    print(f"  获取完整资料: {time3:.3f}秒")
    print(f"  元素检测增加耗时: {time2 - time1:.3f}秒")
    print(f"  OCR识别增加耗时: {time3 - time2:.3f}秒")
    
    if time3 > 3.0:
        print(f"\n⚠️ 警告：获取资料耗时过长（{time3:.3f}秒 > 3秒）")
        if len(result2.elements) == 0:
            print("  可能原因：YOLO未检测到任何元素")
        elif time3 - time2 > 2.0:
            print("  可能原因：OCR识别耗时过长")
        elif time2 - time1 > 1.0:
            print("  可能原因：YOLO元素检测耗时过长")
    else:
        print(f"\n✓ 性能正常（{time3:.3f}秒 < 3秒）")

if __name__ == '__main__':
    asyncio.run(diagnose())
