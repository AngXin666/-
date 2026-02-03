"""
使用原始标注图测试ProfileReader
Test ProfileReader with Original Annotated Images
"""

import sys
from pathlib import Path
import os
import asyncio

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("❌ PIL未安装")
    sys.exit(1)

from src.page_detector_integrated import PageDetectorIntegrated
from src.profile_reader import ProfileReader


class MockADB:
    """模拟ADB，用于测试"""
    
    def __init__(self, test_image_path):
        self.test_image_path = test_image_path
    
    async def screencap(self, device_id: str) -> bytes:
        """返回测试图片的字节数据"""
        with open(self.test_image_path, 'rb') as f:
            return f.read()


async def test_profile_reader():
    """测试ProfileReader"""
    print("=" * 70)
    print("使用原始标注图测试ProfileReader")
    print("=" * 70)
    
    # 查找测试图片
    print("\n[1] 查找测试图片...")
    test_image_path = None
    test_dirs = [
        '原始标注图/个人页_已登录_余额积分/images',
        '原始标注图/个人页_已登录_头像首页/images',
    ]
    
    for img_dir in test_dirs:
        if os.path.exists(img_dir):
            images = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                test_image_path = os.path.join(img_dir, images[0])
                break
    
    if not test_image_path:
        print("❌ 未找到测试图片")
        return
    
    print(f"✓ 测试图片: {test_image_path}")
    
    # 创建模拟ADB
    adb = MockADB(test_image_path)
    
    # 使用ModelManager初始化所有模型
    print("\n[2] 初始化ModelManager...")
    from src.model_manager import ModelManager
    
    try:
        manager = ModelManager.get_instance()
        
        # 初始化所有模型（包括整合检测器和OCR线程池）
        print("  正在加载模型...")
        stats = manager.initialize_all_models(
            adb_bridge=adb,
            log_callback=lambda msg: None  # 静默加载，不输出详细日志
        )
        
        print(f"  ✓ ModelManager初始化完成")
        print(f"    - 加载模型数: {stats['models_loaded']}")
        print(f"    - 总耗时: {stats['total_time']:.2f}秒")
        
        # 从ModelManager获取整合检测器
        detector = manager.get_page_detector_integrated()
        print(f"  ✓ 整合检测器已获取")
        
    except Exception as e:
        print(f"  ❌ ModelManager初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 创建ProfileReader
    print("\n[3] 初始化ProfileReader...")
    try:
        profile_reader = ProfileReader(adb, yolo_detector=detector)
        print(f"  ✓ ProfileReader初始化成功")
        print(f"    - 整合检测器: {'✓' if profile_reader._integrated_detector else '✗'}")
        print(f"    - YOLO检测器: {'✓' if profile_reader._yolo_detector else '✗'}")
        print(f"    - OCR线程池: {'✓' if profile_reader._ocr_pool else '✗'}")
    except Exception as e:
        print(f"  ❌ ProfileReader初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"  - 整合检测器: {'✓' if profile_reader._integrated_detector else '✗'}")
    print(f"  - YOLO检测器: {'✓' if profile_reader._yolo_detector else '✗'}")
    print(f"  - OCR线程池: {'✓' if profile_reader._ocr_pool else '✗'}")
    
    # 测试获取完整个人资料
    print("\n[4] 测试获取完整个人资料...")
    print("=" * 70)
    
    device_id = "test_device"
    
    try:
        profile = await profile_reader.get_full_profile(device_id)
        
        print("\n获取结果:")
        print("-" * 70)
        print(f"  - 昵称: {profile.get('nickname')}")
        print(f"  - 用户ID: {profile.get('user_id')}")
        print(f"  - 手机号: {profile.get('phone')}")
        print(f"  - 余额: {profile.get('balance')}")
        print(f"  - 积分: {profile.get('points')}")
        print(f"  - 抵扣券: {profile.get('vouchers')}")
        print(f"  - 优惠券: {profile.get('coupons')}")
        
        # 统计成功获取的字段
        success_fields = []
        failed_fields = []
        
        field_names = {
            'nickname': '昵称',
            'user_id': '用户ID',
            'phone': '手机号',
            'balance': '余额',
            'points': '积分',
            'vouchers': '抵扣券',
            'coupons': '优惠券'
        }
        
        for key, name in field_names.items():
            if profile.get(key) is not None:
                success_fields.append(name)
            else:
                failed_fields.append(name)
        
        print("\n" + "=" * 70)
        print(f"测试总结:")
        print(f"  - 成功获取: {len(success_fields)}/{len(field_names)} 个字段")
        
        if success_fields:
            print(f"  - ✓ 成功: {', '.join(success_fields)}")
        
        if failed_fields:
            print(f"  - ✗ 失败: {', '.join(failed_fields)}")
        
        print("=" * 70)
        
        # 评估结果
        if len(success_fields) >= 5:  # 至少获取5个字段（手机号除外）
            print("\n✅ 测试通过：ProfileReader能正确获取个人资料数据")
        elif len(success_fields) >= 3:
            print(f"\n⚠️ 测试部分通过：获取了 {len(success_fields)} 个字段")
        else:
            print(f"\n❌ 测试失败：只获取了 {len(success_fields)} 个字段")
    
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    asyncio.run(test_profile_reader())
