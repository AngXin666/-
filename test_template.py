"""测试脚本模板 - 包含自动设备检测、模型初始化、截图等功能
使用方法：复制此文件，修改 main() 函数中的测试逻辑即可
"""
import asyncio
import sys
import subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.adb_bridge import ADBBridge
from src.model_manager import ModelManager

async def main():
    """主测试函数 - 在这里编写你的测试逻辑"""
    
    # ==================== 初始化设备 ====================
    adb_path = r"D:\Program Files\Netease\MuMu\nx_device\12.0\shell\adb.exe"
    adb = ADBBridge(adb_path)
    
    # 自动获取设备列表
    print("正在获取设备列表...")
    try:
        result = subprocess.run(
            [adb_path, "devices"], 
            capture_output=True, 
            text=True, 
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        lines = result.stdout.strip().split('\n')[1:]
        devices = [line.split('\t')[0] for line in lines if line.strip() and 'device' in line]
        
        if not devices:
            print("❌ 没有找到正在运行的设备")
            return
        
        device_id = devices[0]
        print(f"✓ 找到设备: {device_id}")
    except Exception as e:
        print(f"❌ 获取设备列表失败: {e}")
        return
    
    # ==================== 初始化模型 ====================
    print("\n正在初始化模型...")
    model_manager = ModelManager.get_instance()
    model_manager.initialize_all_models(adb)
    
    # 获取常用模型（根据需要选择）
    integrated_detector = model_manager.get_page_detector_integrated()  # 整合检测器
    ocr_pool = model_manager.get_ocr_thread_pool()  # OCR线程池
    
    print("✓ 模型初始化完成\n")
    
    # ==================== 截图示例 ====================
    print("=" * 60)
    print("截图测试")
    print("=" * 60)
    
    screenshot_data = await adb.screencap(device_id)
    if screenshot_data:
        print(f"✓ 截图成功，大小: {len(screenshot_data)} 字节")
        
        # 可选：保存截图到文件
        # from PIL import Image
        # from io import BytesIO
        # image = Image.open(BytesIO(screenshot_data))
        # image.save("test_screenshot.png")
        # print("✓ 截图已保存到 test_screenshot.png")
    else:
        print("❌ 截图失败")
    
    # ==================== 页面检测示例 ====================
    print("\n" + "=" * 60)
    print("页面检测测试")
    print("=" * 60)
    
    page_result = await integrated_detector.detect_page(
        device_id, 
        use_cache=False, 
        detect_elements=True
    )
    
    print(f"页面类型: {page_result.state.chinese_name}")
    print(f"置信度: {page_result.confidence:.2%}")
    print(f"检测到 {len(page_result.elements)} 个元素")
    
    for i, element in enumerate(page_result.elements[:5], 1):  # 只显示前5个
        print(f"  {i}. {element.class_name} (置信度: {element.confidence:.2%})")
    
    # ==================== OCR识别示例 ====================
    print("\n" + "=" * 60)
    print("OCR识别测试")
    print("=" * 60)
    
    from PIL import Image
    from io import BytesIO
    from src.ocr_image_processor import enhance_for_ocr
    
    screenshot_data = await adb.screencap(device_id)
    if screenshot_data:
        image = Image.open(BytesIO(screenshot_data))
        enhanced_image = enhance_for_ocr(image)
        ocr_result = await ocr_pool.recognize(enhanced_image)
        
        if ocr_result and ocr_result.texts:
            print(f"✓ 识别到 {len(ocr_result.texts)} 个文本")
            print(f"前10个文本: {ocr_result.texts[:10]}")
        else:
            print("❌ OCR识别失败")
    
    # ==================== 自定义测试逻辑 ====================
    print("\n" + "=" * 60)
    print("自定义测试")
    print("=" * 60)
    
    # 在这里添加你的测试代码
    # 例如：
    # - 测试特定功能
    # - 性能测试
    # - 多次重复测试
    # - 等等
    
    print("✓ 测试完成")

if __name__ == '__main__':
    asyncio.run(main())
