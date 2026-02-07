"""测试页面分类器对签到弹窗和温馨提示的识别能力"""
import sys
import asyncio
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.adb_bridge import ADBBridge
from src.page_detector_integrated import PageDetectorIntegrated
from src.page_detector import PageState
from src.model_manager import ModelManager

async def test_popup_recognition():
    """测试弹窗识别"""
    print("="*80)
    print("测试页面分类器对签到弹窗和温馨提示的识别")
    print("="*80)
    
    # 初始化
    print("\n[初始化] 初始化ADB和检测器...")
    adb = ADBBridge()
    
    # 直接使用设备ID
    device_id = "127.0.0.1:16416"
    print(f"✓ 使用设备: {device_id}")
    
    # 初始化ModelManager（单例）
    model_manager = ModelManager.get_instance()
    
    # 获取整合检测器
    detector = model_manager.get_integrated_detector()
    print(f"✓ 检测器初始化完成")
    
    # 测试当前页面
    print("\n" + "="*80)
    print("测试当前页面识别")
    print("="*80)
    
    print("\n[步骤1] 截图并检测当前页面...")
    result = await detector.detect_page(device_id, use_cache=False, detect_elements=False)
    
    if result:
        print(f"\n✓ 页面检测结果:")
        print(f"  - 页面类型: {result.state.value}")
        print(f"  - 置信度: {result.confidence:.2%}")
        
        if hasattr(result, 'details') and result.details:
            print(f"  - 详细信息:")
            for key, value in result.details.items():
                print(f"    - {key}: {value}")
        
        # 判断是否是弹窗
        if result.state == PageState.CHECKIN_POPUP:
            print(f"\n⚠️ 检测到签到弹窗，但可能是温馨提示")
            print(f"建议：使用OCR验证是否包含'温馨提示'文字")
        elif result.state == PageState.WARMTIP:
            print(f"\n✓ 正确识别为温馨提示弹窗")
        elif result.state == PageState.CHECKIN:
            print(f"\n✓ 识别为签到页")
        else:
            print(f"\n⚠️ 当前页面: {result.state.value}")
    else:
        print(f"\n❌ 页面检测失败")
    
    # OCR验证
    print("\n" + "="*80)
    print("OCR验证弹窗类型")
    print("="*80)
    
    try:
        from PIL import Image
        from io import BytesIO
        
        print("\n[步骤2] 使用OCR识别页面文字...")
        screenshot_data = await adb.screencap(device_id)
        if screenshot_data:
            image = Image.open(BytesIO(screenshot_data))
            
            # 获取OCR线程池
            ocr_pool = model_manager.get_ocr_thread_pool()
            if ocr_pool:
                ocr_result = await ocr_pool.recognize(image, timeout=3.0)
                if ocr_result and ocr_result.texts:
                    text_str = ''.join(ocr_result.texts)
                    
                    print(f"\n✓ OCR识别结果:")
                    print(f"  - 识别到 {len(ocr_result.texts)} 个文本块")
                    
                    # 检查关键词
                    has_warmtip = "温馨提示" in text_str
                    has_congrats = "恭喜" in text_str
                    has_success = "成功" in text_str
                    has_know = "知道了" in text_str or "知道" in text_str
                    
                    print(f"\n关键词检测:")
                    print(f"  - 温馨提示: {'✓ 是' if has_warmtip else '✗ 否'}")
                    print(f"  - 恭喜: {'✓ 是' if has_congrats else '✗ 否'}")
                    print(f"  - 成功: {'✓ 是' if has_success else '✗ 否'}")
                    print(f"  - 知道了: {'✓ 是' if has_know else '✗ 否'}")
                    
                    # 判断弹窗类型
                    print(f"\n弹窗类型判断:")
                    if has_warmtip:
                        print(f"  ✓ 温馨提示弹窗（次数用完）")
                    elif has_congrats and has_success:
                        print(f"  ✓ 签到奖励弹窗")
                    else:
                        print(f"  ⚠️ 无法判断弹窗类型")
                    
                    # 显示部分文本（调试用）
                    print(f"\nOCR文本预览（前200字符）:")
                    print(f"  {text_str[:200]}")
                else:
                    print(f"\n❌ OCR识别失败")
            else:
                print(f"\n❌ OCR线程池未初始化")
        else:
            print(f"\n❌ 截图失败")
    except Exception as e:
        print(f"\n❌ OCR验证出错: {e}")
    
    # 总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    
    if result:
        if result.state == PageState.CHECKIN_POPUP:
            print("\n⚠️ 页面分类器识别为签到弹窗")
            print("建议：")
            print("  1. 如果实际是温馨提示，说明页面分类器误判")
            print("  2. 需要在代码中增加OCR验证来纠正")
            print("  3. 或者训练页面分类器识别温馨提示页面")
        elif result.state == PageState.WARMTIP:
            print("\n✓ 页面分类器正确识别为温馨提示")
        else:
            print(f"\n当前页面: {result.state.value}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    asyncio.run(test_popup_recognition())
