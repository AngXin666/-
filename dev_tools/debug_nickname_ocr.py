"""
调试昵称OCR识别 - 截图并标注所有识别到的文本
"""
import asyncio
import sys
from pathlib import Path
from io import BytesIO

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adb_bridge import ADBBridge
from src.ocr_thread_pool import get_ocr_pool
from src.ocr_image_processor import enhance_for_ocr

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("❌ 需要安装 Pillow: pip install Pillow")
    sys.exit(1)


async def debug_nickname_ocr():
    """调试昵称OCR识别"""
    print("=" * 60)
    print("调试昵称OCR识别")
    print("=" * 60)
    
    # 初始化模拟器控制器
    from src.emulator_controller import EmulatorController
    emulator = EmulatorController()
    
    # 获取运行的实例
    print("\n正在检测运行的模拟器实例...")
    running_instances = await emulator.get_running_instances()
    
    if not running_instances:
        print("❌ 没有找到运行的模拟器实例")
        print("提示: 请先启动MuMu模拟器")
        return
    
    print(f"✓ 找到 {len(running_instances)} 个运行的实例: {running_instances}")
    
    # 使用第一个实例
    instance_id = running_instances[0]
    port = 16384 + instance_id * 32
    device_id = f"127.0.0.1:{port}"
    
    print(f"\n使用实例 {instance_id}: {device_id}")
    
    # 初始化ADB
    adb = ADBBridge(adb_path=emulator._adb_path)
    
    # 获取OCR线程池
    ocr_pool = get_ocr_pool()
    
    try:
        # 截图
        print("\n正在截图...")
        screenshot_data = await adb.screencap(device_id)
        if not screenshot_data:
            print("❌ 截图失败")
            return
        
        # 保存原始截图
        image = Image.open(BytesIO(screenshot_data))
        output_dir = Path(__file__).parent / "test_screenshots"
        output_dir.mkdir(exist_ok=True)
        
        original_path = output_dir / "nickname_debug_original.png"
        image.save(original_path)
        print(f"✓ 原始截图已保存: {original_path}")
        
        # 图像增强
        print("\n正在进行OCR识别...")
        enhanced_image = enhance_for_ocr(image)
        
        # OCR识别
        ocr_result = await ocr_pool.recognize(enhanced_image, timeout=10.0)
        
        if not ocr_result or not ocr_result.texts:
            print("❌ OCR未识别到文本")
            return
        
        texts = ocr_result.texts
        boxes = ocr_result.boxes if hasattr(ocr_result, 'boxes') else None
        
        print(f"\n✓ OCR识别到 {len(texts)} 个文本块")
        print("\n" + "=" * 60)
        print("识别到的所有文本（按顺序）:")
        print("=" * 60)
        for i, text in enumerate(texts):
            print(f"[{i}] {text}")
        
        # 创建标注图像
        print("\n正在创建标注图像...")
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        
        # 尝试加载字体
        try:
            font = ImageFont.truetype("msyh.ttc", 16)  # 微软雅黑
            small_font = ImageFont.truetype("msyh.ttc", 12)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", 16)
                small_font = ImageFont.truetype("arial.ttf", 12)
            except:
                font = ImageFont.load_default()
                small_font = ImageFont.load_default()
        
        # 标注每个文本块
        if boxes is not None and len(boxes) > 0:
            for i, (text, box) in enumerate(zip(texts, boxes)):
                # 绘制边框
                points = [(int(p[0]), int(p[1])) for p in box]
                draw.polygon(points, outline=(255, 0, 0), width=2)
                
                # 绘制索引号
                x, y = int(box[0][0]), int(box[0][1])
                draw.text((x, y - 20), f"[{i}]", fill=(255, 0, 0), font=small_font)
                
                # 绘制文本内容
                draw.text((x, y - 40), text[:10], fill=(0, 0, 255), font=small_font)
        else:
            print("  ! 没有位置信息，无法标注")
        
        # 保存标注图像
        annotated_path = output_dir / "nickname_debug_annotated.png"
        annotated_image.save(annotated_path)
        print(f"✓ 标注图像已保存: {annotated_path}")
        
        # 分析昵称识别
        print("\n" + "=" * 60)
        print("昵称识别分析:")
        print("=" * 60)
        
        # 查找可能的昵称
        exclude_keywords = [
            "ID", "id", "普通会员", "VIP", "会员", "手机", "余额", "积分", 
            "抵扣券", "优惠券", "抵扣券", "我的", "设置", "首页", "分类",
            "商城", "订单", "查看", "待付款", "待发货", "待收货", "待评价"
        ]
        
        print("\n可能的昵称候选（前10个文本块，排除关键字）:")
        for i, text in enumerate(texts[:10]):
            # 跳过纯数字、时间格式、单字符
            if text.isdigit() or len(text) <= 1:
                continue
            
            # 跳过时间格式
            if ':' in text or '：' in text:
                continue
            
            # 跳过包含关键字的文本
            if any(kw in text for kw in exclude_keywords):
                print(f"  [{i}] {text} - ✗ 包含关键字，跳过")
                continue
            
            # 昵称通常是2-20个字符
            if 2 <= len(text) <= 20:
                print(f"  [{i}] {text} - ✓ 可能是昵称")
            else:
                print(f"  [{i}] {text} - ? 长度不符（{len(text)}字符）")
        
        print("\n" + "=" * 60)
        print("请查看标注图像，确认昵称的位置和索引号")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(debug_nickname_ocr())
