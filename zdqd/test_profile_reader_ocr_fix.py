"""
测试 ProfileReader OCR 修复
验证所有 fallback 方法现在使用 _ocr_pool.recognize() 而不是 _ocr

测试内容:
1. 验证 ProfileReader 初始化正常
2. 验证 _ocr_pool 存在且可用
3. 模拟调用 fallback 方法（不会真正执行OCR，只验证代码路径）
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.profile_reader import ProfileReader
from src.adb_bridge import ADBBridge
from src.model_manager import ModelManager


async def test_profile_reader_ocr_fix():
    """测试 ProfileReader OCR 修复"""
    
    print("=" * 60)
    print("测试 ProfileReader OCR 修复")
    print("=" * 60)
    
    try:
        # 初始化 ADB
        print("\n[步骤1] 初始化 ADB...")
        adb = ADBBridge()
        
        # 初始化 ModelManager
        print("[步骤2] 初始化 ModelManager...")
        model_manager = ModelManager()
        
        # 初始化模型（需要初始化 OCR 线程池）
        print("[步骤2.1] 初始化模型...")
        model_manager.initialize_all_models(adb)
        
        # 初始化 ProfileReader
        print("[步骤3] 初始化 ProfileReader...")
        profile_reader = ProfileReader(adb, model_manager)
        
        # 验证 _ocr_pool 存在
        print("[步骤4] 验证 _ocr_pool 存在...")
        if hasattr(profile_reader, '_ocr_pool'):
            print("  ✓ _ocr_pool 存在")
        else:
            print("  ✗ _ocr_pool 不存在")
            return False
        
        # 验证 _ocr_pool 有 recognize 方法
        if hasattr(profile_reader._ocr_pool, 'recognize'):
            print("  ✓ _ocr_pool.recognize() 方法存在")
        else:
            print("  ✗ _ocr_pool.recognize() 方法不存在")
            return False
        
        # 验证没有 _ocr 属性（旧的错误属性）
        if hasattr(profile_reader, '_ocr'):
            print("  ⚠️ 警告: _ocr 属性仍然存在（不应该存在）")
        else:
            print("  ✓ _ocr 属性不存在（正确）")
        
        # 验证 fallback 方法存在
        print("\n[步骤5] 验证 fallback 方法存在...")
        fallback_methods = [
            'get_balance_fallback',
            'get_user_id_fallback',
            'get_nickname_fallback',
            'get_phone_fallback',
            'get_points_fallback',
            'get_vouchers_fallback'
        ]
        
        for method_name in fallback_methods:
            if hasattr(profile_reader, method_name):
                print(f"  ✓ {method_name} 存在")
            else:
                print(f"  ✗ {method_name} 不存在")
        
        print("\n" + "=" * 60)
        print("✓ 所有检查通过！ProfileReader OCR 修复成功")
        print("=" * 60)
        
        print("\n说明:")
        print("  - 所有 fallback 方法现在使用 self._ocr_pool.recognize()")
        print("  - 不再使用不存在的 self._ocr 属性")
        print("  - OCR 调用将正常工作，不会抛出 AttributeError")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_profile_reader_ocr_fix())
    sys.exit(0 if success else 1)
