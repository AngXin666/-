"""
OCR图像预处理模块使用示例
演示如何使用 ocr_image_processor 模块简化代码
"""

from io import BytesIO
from PIL import Image
import cv2
import numpy as np

# 导入OCR图像预处理模块
from .ocr_image_processor import enhance_for_ocr


# ============================================
# 示例1: 在PIL图像处理中使用
# ============================================

async def example_pil_usage(screenshot_bytes):
    """示例：使用PIL处理截图"""
    
    # 原来的代码（需要手动写灰度图转换和对比度增强）:
    # img = Image.open(BytesIO(screenshot_bytes))
    # gray_img = img.convert('L')
    # enhanced_img = ImageEnhance.Contrast(gray_img).enhance(2.0)
    
    # 新的代码（一行搞定）:
    img = Image.open(BytesIO(screenshot_bytes))
    enhanced_img = enhance_for_ocr(img)
    
    # 然后直接用于OCR识别
    # ocr_result = ocr_engine(enhanced_img)
    return enhanced_img


# ============================================
# 示例2: 在OpenCV图像处理中使用
# ============================================

async def example_cv2_usage(screenshot_array):
    """示例：使用OpenCV处理截图"""
    
    # 原来的代码（需要手动写灰度图转换、PIL转换、对比度增强、再转回numpy）:
    # gray = cv2.cvtColor(screenshot_array, cv2.COLOR_BGR2GRAY)
    # pil_image = Image.fromarray(gray)
    # enhanced = ImageEnhance.Contrast(pil_image).enhance(2.0)
    # enhanced_array = np.array(enhanced)
    
    # 新的代码（一行搞定）:
    enhanced_array = enhance_for_ocr(screenshot_array)
    
    # 然后直接用于OCR识别
    # text = pytesseract.image_to_string(enhanced_array, lang='chi_sim+eng')
    return enhanced_array


# ============================================
# 示例3: 在实际OCR识别方法中使用
# ============================================

class ExampleOCRReader:
    """示例：OCR读取器类"""
    
    def __init__(self, adb):
        self.adb = adb
        from rapidocr import RapidOCR
        self._ocr = RapidOCR()
    
    async def read_page_info(self, device_id: str):
        """读取页面信息"""
        
        # 获取截图
        screenshot = await self.adb.screencap(device_id)
        if not screenshot:
            return None
        
        # 原来的代码（4行）:
        # img = Image.open(BytesIO(screenshot))
        # gray_img = img.convert('L')
        # enhanced_img = ImageEnhance.Contrast(gray_img).enhance(2.0)
        # ocr_result = self._ocr(enhanced_img)
        
        # 新的代码（2行）:
        img = Image.open(BytesIO(screenshot))
        enhanced_img = enhance_for_ocr(img)
        ocr_result = self._ocr(enhanced_img)
        
        # 处理OCR结果
        if ocr_result and ocr_result.txts:
            return list(ocr_result.txts)
        return []


# ============================================
# 示例4: 自定义对比度增强倍数
# ============================================

async def example_custom_contrast(screenshot_bytes):
    """示例：使用自定义对比度增强倍数"""
    
    img = Image.open(BytesIO(screenshot_bytes))
    
    # 使用默认的2.0倍对比度
    enhanced_2x = enhance_for_ocr(img)
    
    # 使用1.5倍对比度
    enhanced_1_5x = enhance_for_ocr(img, contrast_factor=1.5)
    
    # 使用2.5倍对比度
    enhanced_2_5x = enhance_for_ocr(img, contrast_factor=2.5)
    
    return enhanced_2x


# ============================================
# 代码对比总结
# ============================================

"""
使用新模块的优势：

1. 代码更简洁
   原来: 3-4行代码
   现在: 1行代码

2. 统一标准
   所有OCR识别都使用相同的预处理方式
   避免不同地方使用不同的处理方法

3. 易于维护
   如果需要调整预处理方式，只需修改一个地方
   不需要到处修改代码

4. 自动适配
   自动检测PIL或OpenCV图像类型
   使用合适的处理方法

5. 灵活性
   可以自定义对比度增强倍数
   默认使用测试验证的最佳值（2.0倍）
"""
