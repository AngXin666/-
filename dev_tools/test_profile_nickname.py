"""测试个人页昵称获取"""
import sys
import os
import asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.adb_bridge import ADBBridge
from src.profile_reader import ProfileReader
from src.model_manager import ModelManager
from src.emulator_controller import EmulatorController

async def main():
    # 初始化模拟器控制器
    emulator_controller = EmulatorController()
    
    # 初始化ADB（使用正确的ADB路径）
    adb_path = emulator_controller.get_adb_path()
    adb = ADBBridge(adb_path=adb_path)
    
    # 获取已连接的设备列表
    import subprocess
    result = subprocess.run(
        [adb_path, "devices"],
        capture_output=True,
        text=True
    )
    
    # 解析设备列表
    devices = []
    for line in result.stdout.split('\n'):
        if '\tdevice' in line:
            device_id = line.split('\t')[0]
            devices.append(device_id)
    
    if not devices:
        print("❌ 未找到已连接的设备")
        print("请确保模拟器正在运行")
        return
    
    device_id = devices[0]
    print(f"✓ 找到设备: {device_id}\n")
    
    # 测试截图
    print("测试截图...")
    try:
        screenshot = await adb.screencap(device_id)
        if screenshot:
            print(f"✓ 截图成功，大小: {len(screenshot)} 字节\n")
        else:
            print("❌ 截图失败，返回空数据\n")
            return
    except Exception as e:
        print(f"❌ 截图失败: {e}\n")
        return
    
    # 初始化ModelManager和检测器
    print("初始化模型...")
    model_manager = ModelManager.get_instance()
    model_manager.initialize_all_models(adb)  # 不需要await
    detector = model_manager.get_page_detector_integrated()
    print("✓ 模型初始化完成\n")
    
    # 初始化ProfileReader
    profile_reader = ProfileReader(adb, yolo_detector=detector)
    
    print("="*60)
    print("测试: 使用 get_full_profile_with_retry (正常流程)")
    print("="*60)
    
    try:
        # 使用正常流程的方法
        profile_data = await profile_reader.get_full_profile_with_retry(
            device_id, 
            account="test_account",
            max_retries=1  # 只尝试1次，快速测试
        )
        
        print("\n获取结果:")
        print(f"  昵称: {profile_data.get('nickname')}")
        print(f"  用户ID: {profile_data.get('user_id')}")
        print(f"  余额: {profile_data.get('balance')}")
        print(f"  积分: {profile_data.get('points')}")
        print(f"  抵扣券: {profile_data.get('vouchers')}")
        print(f"  优惠券: {profile_data.get('coupons')}")
        
        # 额外测试：直接截图并OCR，看看优惠券区域有什么文本
        print("\n" + "="*60)
        print("额外测试：检查优惠券区域的OCR结果")
        print("="*60)
        
        from PIL import Image
        from io import BytesIO
        from src.ocr_image_processor import enhance_for_ocr
        
        screenshot_data = await adb.screencap(device_id)
        if screenshot_data:
            image = Image.open(BytesIO(screenshot_data))
            
            # 优惠券区域（根据REGIONS定义）
            coupon_region = (410, 230, 490, 330)
            cropped = image.crop(coupon_region)
            enhanced = enhance_for_ocr(cropped)
            
            # OCR识别
            ocr_pool = model_manager.get_ocr_thread_pool()
            
            ocr_result = await ocr_pool.recognize(enhanced, timeout=5.0)
            
            if ocr_result and ocr_result.texts:
                print(f"\n优惠券区域OCR结果:")
                for i, text in enumerate(ocr_result.texts):
                    print(f"  [{i}] '{text}'")
            else:
                print(f"\n⚠️ 优惠券区域OCR失败")
        
        # 检查昵称是否是错误识别
        nickname = profile_data.get('nickname')
        if nickname:
            if nickname in ['西', '1 0', '10', '1', '0']:
                print(f"\n⚠️⚠️⚠️ 昵称识别错误: '{nickname}'")
            else:
                print(f"\n✓ 昵称识别正常: '{nickname}'")
        else:
            print(f"\n⚠️ 未获取到昵称")
            
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())
