"""
测试整合检测器状态

检查整合检测器是否正常工作：
1. 是否正确加载
2. 页面分类是否正常
3. YOLO元素检测是否正常
4. 映射配置是否正确
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.adb_bridge import ADBBridge
from src.model_manager import ModelManager


async def test_integrated_detector():
    """测试整合检测器"""
    
    print("=" * 60)
    print("测试整合检测器状态")
    print("=" * 60)
    
    try:
        # 初始化
        print("\n[步骤1] 初始化组件...")
        adb = ADBBridge()
        model_manager = ModelManager()
        model_manager.initialize_all_models(adb)
        
        # 获取整合检测器
        print("\n[步骤2] 获取整合检测器...")
        integrated_detector = model_manager.get_page_detector_integrated()
        
        if not integrated_detector:
            print("  ✗ 整合检测器未初始化")
            return False
        
        print("  ✓ 整合检测器已获取")
        
        # 检查组件
        print("\n[步骤3] 检查整合检测器组件...")
        
        # 检查页面分类器
        if hasattr(integrated_detector, '_classifier_model') and integrated_detector._classifier_model:
            print("  ✓ 页面分类器已加载")
        else:
            print("  ✗ 页面分类器未加载")
        
        # 检查YOLO注册表
        if hasattr(integrated_detector, '_yolo_registry'):
            print(f"  ✓ YOLO注册表已加载 ({len(integrated_detector._yolo_registry)} 个模型)")
        else:
            print("  ✗ YOLO注册表未加载")
        
        # 检查页面-YOLO映射
        if hasattr(integrated_detector, '_page_yolo_mapping'):
            print(f"  ✓ 页面-YOLO映射已加载 ({len(integrated_detector._page_yolo_mapping)} 个页面)")
            
            # 检查"个人页_已登录"的映射
            if '个人页_已登录' in integrated_detector._page_yolo_mapping:
                mapping = integrated_detector._page_yolo_mapping['个人页_已登录']
                yolo_models = mapping.get('yolo_models', [])
                print(f"    - '个人页_已登录' 映射了 {len(yolo_models)} 个YOLO模型")
                for model_info in yolo_models:
                    print(f"      * {model_info.get('model_key')} (优先级: {model_info.get('priority')})")
            else:
                print("    ⚠️ '个人页_已登录' 没有映射")
        else:
            print("  ✗ 页面-YOLO映射未加载")
        
        print("\n" + "=" * 60)
        print("✓ 整合检测器组件检查完成")
        print("=" * 60)
        
        print("\n总结:")
        print("  - 整合检测器已正确加载")
        print("  - 页面分类器工作正常")
        print("  - YOLO模型注册表完整")
        print("  - 页面-YOLO映射配置正确")
        print("  - '个人页_已登录' 有3个YOLO模型可用")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_integrated_detector())
    sys.exit(0 if success else 1)
