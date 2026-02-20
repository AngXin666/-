"""
测试YOLO按钮检测功能
"""
import asyncio
import sys
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.adb_bridge import ADBBridge
from src.page_detector_integrated import PageDetectorIntegrated
from src.model_manager import ModelManager

async def test_button_detection():
    """测试按钮检测"""
    print("="*60)
    print("测试YOLO按钮检测功能")
    print("="*60)
    
    # 1. 初始化ADB
    print("\n[1] 初始化ADB...")
    adb = ADBBridge()
    
    # 使用固定设备ID（第一个MuMu实例）
    device_id = "127.0.0.1:16384"
    print(f"  ✓ 使用设备: {device_id}")
    
    # 2. 初始化模型管理器和检测器
    print("\n[2] 初始化模型...")
    model_manager = ModelManager.get_instance()
    detector = model_manager.get_page_detector()
    
    if not detector:
        print("  ✗ 无法获取页面检测器")
        return False
    
    print(f"  ✓ 检测器类型: {type(detector).__name__}")
    
    # 3. 检测当前页面
    print("\n[3] 检测当前页面...")
    page_result = await detector.detect_page(device_id, use_cache=False, detect_elements=False)
    
    if not page_result:
        print("  ✗ 页面检测失败")
        return False
    
    print(f"  ✓ 当前页面: {page_result.state.value}")
    print(f"  ✓ 置信度: {page_result.confidence:.2%}")
    
    # 4. 如果是首页，尝试检测签到按钮
    if page_result.state.value == "首页":
        print("\n[4] 检测签到按钮...")
        
        # 使用元素检测
        page_result_with_elements = await detector.detect_page(
            device_id, 
            use_cache=False, 
            detect_elements=True
        )
        
        if page_result_with_elements and page_result_with_elements.elements:
            print(f"  ✓ 检测到 {len(page_result_with_elements.elements)} 个元素:")
            for element in page_result_with_elements.elements:
                print(f"    - {element.class_name}: {element.center} (置信度: {element.confidence:.2%})")
                
                if "签到" in element.class_name:
                    print(f"\n  ✓✓✓ 找到签到按钮: {element.center}")
                    return True
        else:
            print("  ✗ 未检测到任何元素")
            
            # 尝试直接使用YOLO检测
            print("\n[5] 尝试直接YOLO检测...")
            try:
                # 检查YOLO检测器是否可用
                if hasattr(detector, '_yolo_detector'):
                    print("  ✓ YOLO检测器可用")
                else:
                    print("  ✗ YOLO检测器不可用")
            except Exception as e:
                print(f"  ✗ 检查YOLO检测器失败: {e}")
    else:
        print(f"\n  ⚠️ 当前不在首页，无法测试签到按钮检测")
        print(f"  请先导航到首页")
    
    return False

if __name__ == "__main__":
    try:
        result = asyncio.run(test_button_detection())
        if result:
            print("\n" + "="*60)
            print("✓ 测试通过：YOLO按钮检测正常工作")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("✗ 测试失败：YOLO按钮检测有问题")
            print("="*60)
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n✗ 测试出错: {e}")
        import traceback
        traceback.print_exc()
