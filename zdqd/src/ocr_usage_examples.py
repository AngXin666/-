"""
OCR增强功能使用示例
OCR Enhancement Usage Examples

这个文件展示如何使用OCR增强功能来提高识别准确率
"""

import asyncio
from PIL import Image
from io import BytesIO

from .adb_bridge import ADBBridge
from .ocr_enhancer import get_ocr_enhancer
from .image_processor import ImageProcessor


class OCRUsageExamples:
    """OCR使用示例"""
    
    def __init__(self, adb: ADBBridge):
        self.adb = adb
        self.enhancer = get_ocr_enhancer()
    
    async def example_1_recognize_amount(self, device_id: str):
        """示例1: 识别签到奖励金额（高准确率）"""
        print("\n=== 示例1: 识别签到奖励金额 ===")
        
        # 1. 获取截图
        screenshot_data = await self.adb.screencap(device_id)
        image = Image.open(BytesIO(screenshot_data))
        
        # 2. 裁剪金额区域（弹窗中央）
        amount_region = (120, 300, 300, 200)  # (x, y, width, height)
        
        # 3. 使用增强器识别金额
        amount = await self.enhancer.recognize_amount(
            image,
            min_value=0.01,
            max_value=100.0
        )
        
        if amount:
            print(f"✓ 识别到金额: {amount:.2f} 元")
        else:
            print("✗ 金额识别失败")
        
        return amount
    
    async def example_2_find_button_with_retry(self, device_id: str, button_text: str):
        """示例2: 查找按钮文字（多模式重试）"""
        print(f"\n=== 示例2: 查找按钮 '{button_text}' ===")
        
        # 1. 获取截图
        screenshot_data = await self.adb.screencap(device_id)
        image = Image.open(BytesIO(screenshot_data))
        
        # 2. 使用多模式重试识别
        result = await self.enhancer.recognize_with_retry(
            image,
            modes=['text', 'balanced', 'high_contrast'],
            max_attempts=3
        )
        
        if result.success:
            print(f"✓ 识别成功")
            print(f"  - 文本: {result.text}")
            print(f"  - 置信度: {result.confidence:.2f}")
            print(f"  - 使用方法: {result.method}")
            
            # 检查是否包含目标文字
            if button_text in result.text:
                print(f"  - ✓ 找到目标文字")
                return True
        else:
            print("✗ 识别失败")
        
        return False
    
    async def example_3_recognize_remaining_times(self, device_id: str):
        """示例3: 识别剩余次数（数字识别）"""
        print("\n=== 示例3: 识别剩余次数 ===")
        
        # 1. 获取截图
        screenshot_data = await self.adb.screencap(device_id)
        image = Image.open(BytesIO(screenshot_data))
        
        # 2. 使用数字识别
        result = await self.enhancer.recognize_number(
            image,
            pattern=r'剩余[：:\s]*(\d+)'  # 匹配 "剩余3次" 或 "剩余：3"
        )
        
        if result.success:
            print(f"✓ 识别成功")
            print(f"  - 识别结果: {result.text}")
            print(f"  - 所有匹配: {result.all_results}")
            
            try:
                times = int(result.text)
                print(f"  - 剩余次数: {times}")
                return times
            except ValueError:
                pass
        else:
            print("✗ 识别失败")
        
        return None
    
    async def example_4_fuzzy_match(self, device_id: str, target_text: str):
        """示例4: 模糊匹配文字（容错识别）"""
        print(f"\n=== 示例4: 模糊匹配 '{target_text}' ===")
        
        # 1. 获取截图
        screenshot_data = await self.adb.screencap(device_id)
        image = Image.open(BytesIO(screenshot_data))
        
        # 2. 模糊匹配
        found = await self.enhancer.find_text_with_fuzzy(
            image,
            target_text,
            similarity_threshold=0.7  # 70%相似度即可
        )
        
        if found:
            print(f"✓ 找到匹配文字")
        else:
            print(f"✗ 未找到匹配文字")
        
        return found
    
    async def example_5_multi_scale(self, device_id: str):
        """示例5: 多尺度识别（处理不同大小的文字）"""
        print("\n=== 示例5: 多尺度识别 ===")
        
        # 1. 获取截图
        screenshot_data = await self.adb.screencap(device_id)
        image = Image.open(BytesIO(screenshot_data))
        
        # 2. 多尺度识别
        result = await self.enhancer.recognize_multi_scale(
            image,
            scales=[0.8, 1.0, 1.2, 1.5]  # 尝试不同缩放比例
        )
        
        if result.success:
            print(f"✓ 识别成功")
            print(f"  - 文本: {result.text}")
            print(f"  - 最佳缩放: {result.method}")
            print(f"  - 置信度: {result.confidence:.2f}")
        else:
            print("✗ 识别失败")
        
        return result
    
    async def example_6_region_recognition(self, device_id: str):
        """示例6: 区域识别（只识别特定区域）"""
        print("\n=== 示例6: 区域识别 ===")
        
        # 1. 获取截图
        screenshot_data = await self.adb.screencap(device_id)
        image = Image.open(BytesIO(screenshot_data))
        
        # 2. 定义多个感兴趣区域
        regions = {
            '顶部': (0, 0, 540, 200),
            '中部': (0, 300, 540, 400),
            '底部': (0, 700, 540, 260)
        }
        
        results = {}
        for name, region in regions.items():
            result = await self.enhancer.recognize_with_region(
                image,
                region,
                mode='text'
            )
            
            if result.success:
                print(f"✓ {name}区域: {result.text[:50]}...")
                results[name] = result.text
            else:
                print(f"✗ {name}区域: 识别失败")
        
        return results


# 实际使用示例
async def practical_example_checkin_amount():
    """实际应用: 签到金额识别（完整流程）"""
    print("\n" + "="*60)
    print("实际应用: 签到金额识别")
    print("="*60)
    
    # 假设已有ADB实例
    # adb = ADBBridge()
    # device_id = "emulator-5554"
    # enhancer = get_ocr_enhancer()
    
    # 完整流程:
    print("""
    1. 获取签到弹窗截图
    2. 裁剪金额显示区域
    3. 使用'number'模式预处理
    4. OCR识别
    5. 正则提取金额
    6. 范围验证
    
    代码示例:
    
    # 获取截图
    screenshot_data = await adb.screencap(device_id)
    image = Image.open(BytesIO(screenshot_data))
    
    # 裁剪金额区域
    amount_region = (120, 300, 300, 200)
    cropped = ImageProcessor.crop_region(image, *amount_region)
    
    # 使用增强器识别
    amount = await enhancer.recognize_amount(
        cropped,
        min_value=0.01,
        max_value=100.0
    )
    
    if amount:
        print(f"识别到金额: {amount:.2f} 元")
    """)


async def practical_example_button_click():
    """实际应用: 按钮点击（容错识别）"""
    print("\n" + "="*60)
    print("实际应用: 按钮点击")
    print("="*60)
    
    print("""
    场景: 需要点击"签到"按钮，但OCR可能识别成"签到"、"签 到"、"筌到"等
    
    解决方案: 使用模糊匹配
    
    代码示例:
    
    # 获取截图
    screenshot_data = await adb.screencap(device_id)
    image = Image.open(BytesIO(screenshot_data))
    
    # 模糊匹配"签到"
    found = await enhancer.find_text_with_fuzzy(
        image,
        "签到",
        similarity_threshold=0.7  # 70%相似即可
    )
    
    if found:
        # 点击按钮
        await adb.tap(device_id, x, y)
    """)


# 注意：这是示例代码，不要在打包时自动执行
# if __name__ == "__main__":
#     # 运行示例
#     asyncio.run(practical_example_checkin_amount())
#     asyncio.run(practical_example_button_click())
