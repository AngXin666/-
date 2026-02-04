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
    print("YOLO检测可视化")
    print("=" * 60)
    
    # 创建可视化图像
    from PIL import Image, ImageDraw, ImageFont
    from io import BytesIO
    import re
    
    screenshot_data = await adb.screencap(device_id)
    if not screenshot_data:
        print("❌ 截图失败")
        return
    
    image = Image.open(BytesIO(screenshot_data))
    vis_image = image.copy()
    draw = ImageDraw.Draw(vis_image)
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("msyh.ttc", 20)
    except:
        font = ImageFont.load_default()
    
    # 颜色映射
    colors = {
        '余额': '#FF0000',      # 红色
        '积分': '#00FF00',      # 绿色
        '抵扣': '#0000FF',      # 蓝色
        '优惠': '#FFFF00',      # 黄色
        '昵称': '#FF00FF',      # 品红
        'ID': '#00FFFF',        # 青色
        '首页': '#FFA500',      # 橙色
        '我的': '#800080'       # 紫色
    }
    
    # 对每个元素进行OCR识别并绘制
    for i, element in enumerate(page_result.elements):
        x1, y1, x2, y2 = element.bbox
        class_name = element.class_name
        confidence = element.confidence
        
        print(f"\n元素 {i+1}: {class_name}")
        print(f"  位置: ({x1:.0f}, {y1:.0f}) - ({x2:.0f}, {y2:.0f})")
        print(f"  置信度: {confidence:.2%}")
        
        # 选择颜色
        color = '#FFFFFF'  # 默认白色
        for key, c in colors.items():
            if key in class_name:
                color = c
                break
        
        # 绘制边框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # 绘制标签
        label = f"{class_name} {confidence:.0%}"
        draw.text((x1, y1 - 25), label, fill=color, font=font)
        
        # 如果是数字区域，进行OCR识别
        if '数字' in class_name or '文字' in class_name:
            # 裁剪区域
            region = image.crop((x1, y1, x2, y2))
            
            # OCR识别
            ocr_result = await ocr_pool.recognize(region, timeout=3.0)
            
            if ocr_result and ocr_result.texts:
                ocr_text = ' '.join(ocr_result.texts)
                print(f"  OCR识别: {ocr_text}")
                
                # 提取数字
                numbers = re.findall(r'(\d+\.?\d*)', ocr_text)
                if numbers:
                    print(f"  提取数字: {numbers}")
            else:
                print(f"  OCR识别: 失败")
    
    # 保存可视化结果
    from pathlib import Path
    from datetime import datetime
    
    output_dir = Path("debug_profile_detection")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"detection_{timestamp}.png"
    
    vis_image.save(output_path)
    print(f"\n✓ 可视化结果已保存: {output_path}")
    
    # 打开图片
    import os
    os.startfile(output_path)
    print("✓ 已打开可视化图片")

    
    print("✓ 测试完成")

if __name__ == '__main__':
    asyncio.run(main())
