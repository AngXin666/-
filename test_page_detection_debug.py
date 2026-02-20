"""
调试页面检测问题
"""
import asyncio
import sys
from pathlib import Path
from io import BytesIO

sys.path.insert(0, str(Path(__file__).parent))

from src.adb_bridge import ADBBridge
from src.model_manager import ModelManager
from PIL import Image

async def main():
    print("="*60)
    print("调试页面检测")
    print("="*60)
    
    device_id = "127.0.0.1:16384"
    adb_path = r"D:\Program Files\Netease\MuMu\nx_device\12.0\shell\adb.exe"
    adb = ADBBridge(adb_path=adb_path)
    
    # 1. 测试截图
    print("\n[1] 测试ADB截图...")
    try:
        screenshot_data = await adb.screencap(device_id)
        if screenshot_data:
            print(f"  ✓ 截图成功，大小: {len(screenshot_data)} 字节")
            
            # 保存截图
            image = Image.open(BytesIO(screenshot_data))
            image.save("debug_screenshot.png")
            print(f"  ✓ 截图已保存: debug_screenshot.png")
            print(f"  ✓ 图片尺寸: {image.size}")
        else:
            print("  ✗ 截图失败，返回None")
            return
    except Exception as e:
        print(f"  ✗ 截图出错: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 2. 初始化模型
    print("\n[2] 初始化模型...")
    model_manager = ModelManager.get_instance()
    model_manager.initialize_all_models(adb)
    detector = model_manager.get_page_detector_integrated()
    print(f"  ✓ 检测器: {type(detector).__name__}")
    
    # 3. 检测页面（不使用缓存）
    print("\n[3] 检测页面...")
    try:
        page_result = await detector.detect_page(device_id, use_cache=False, detect_elements=False)
        
        if page_result:
            print(f"  ✓ 页面检测成功")
            print(f"    - 状态: {page_result.state.value}")
            print(f"    - 置信度: {page_result.confidence:.2%}")
            print(f"    - 检测方法: {page_result.detection_method if hasattr(page_result, 'detection_method') else 'N/A'}")
        else:
            print("  ✗ 页面检测返回None")
    except Exception as e:
        print(f"  ✗ 页面检测出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. 尝试元素检测
    print("\n[4] 尝试元素检测...")
    try:
        page_result = await detector.detect_page(device_id, use_cache=False, detect_elements=True)
        
        if page_result:
            print(f"  ✓ 页面: {page_result.state.value} (置信度: {page_result.confidence:.2%})")
            
            if page_result.elements:
                print(f"  ✓ 检测到 {len(page_result.elements)} 个元素:")
                for element in page_result.elements:
                    print(f"    - {element.class_name}: {element.center} (置信度: {element.confidence:.2%})")
            else:
                print("  ⚠️ 未检测到任何元素")
        else:
            print("  ✗ 页面检测返回None")
    except Exception as e:
        print(f"  ✗ 元素检测出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n✗ 出错: {e}")
        import traceback
        traceback.print_exc()
