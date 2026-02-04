"""
测试余额OCR修复 - 不使用固定坐标

测试新的全屏OCR + 关键字定位方法
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.profile_reader import ProfileReader
from src.adb_bridge import ADBBridge
from src.model_manager import ModelManager


async def test_balance_ocr():
    """测试余额OCR（不使用固定坐标）"""
    
    print("=" * 60)
    print("测试余额OCR修复 - 全屏OCR + 关键字定位")
    print("=" * 60)
    
    try:
        # 初始化
        print("\n[步骤1] 初始化组件...")
        adb = ADBBridge()
        model_manager = ModelManager()
        model_manager.initialize_all_models(adb)
        
        # 获取设备列表
        devices = await adb.list_devices()
        if not devices:
            print("  ✗ 没有连接的设备")
            return False
        
        device_id = devices[0]
        print(f"  ✓ 使用设备: {device_id}")
        
        # 初始化 ProfileReader
        profile_reader = ProfileReader(adb, model_manager)
        
        print("\n[步骤2] 测试余额获取...")
        print("  说明: 新方法不使用固定坐标，而是：")
        print("    1. 全屏OCR识别所有文本")
        print("    2. 查找'余额'关键字")
        print("    3. 在关键字附近提取数字")
        
        # 获取动态数据
        result = await profile_reader._get_dynamic_data_only(device_id)
        
        print("\n[步骤3] 结果:")
        print(f"  余额: {result.get('balance')}")
        print(f"  积分: {result.get('points')}")
        print(f"  抵扣券: {result.get('vouchers')}")
        print(f"  优惠券: {result.get('coupons')}")
        
        # 验证
        success = True
        if result.get('balance') is None:
            print("\n  ⚠️ 警告: 未能获取余额")
            print("  可能原因:")
            print("    1. OCR未识别到'余额'关键字")
            print("    2. '余额'关键字附近没有数字")
            print("    3. 当前不在个人页")
            success = False
        else:
            print(f"\n  ✓ 成功获取余额: {result['balance']:.2f} 元")
        
        if result.get('vouchers') is not None:
            print(f"  ✓ 成功获取抵扣券: {result['vouchers']} 张")
        
        print("\n" + "=" * 60)
        if success:
            print("✓ 测试通过！新的OCR方法工作正常")
        else:
            print("⚠️ 测试部分通过，请检查日志")
        print("=" * 60)
        
        return success
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_balance_ocr())
    sys.exit(0 if success else 1)
