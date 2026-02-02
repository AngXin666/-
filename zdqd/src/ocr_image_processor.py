"""
OCR图像预处理模块
OCR Image Preprocessing Module

提供统一的图像预处理方法，用于提高OCR识别准确率
"""

from typing import Union
import numpy as np

try:
    from PIL import Image, ImageEnhance
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class OCRImageProcessor:
    """OCR图像预处理器
    
    提供统一的图像预处理方法：
    1. 灰度图转换
    2. 对比度增强2倍
    
    这个方案在测试中表现最好，特别适合识别灰色背景上的文字
    """
    
    @staticmethod
    def enhance_for_ocr_pil(image: 'Image.Image', contrast_factor: float = 2.0) -> 'Image.Image':
        """使用PIL增强图像以提高OCR识别准确率
        
        处理流程：
        1. 转换为灰度图
        2. 增强对比度（默认2倍）
        
        Args:
            image: PIL Image对象
            contrast_factor: 对比度增强倍数，默认2.0
            
        Returns:
            增强后的PIL Image对象（灰度图）
        """
        if not HAS_PIL:
            raise ImportError("PIL/Pillow未安装")
        
        # 1. 转换为灰度图
        gray_image = image.convert('L')
        
        # 2. 增强对比度
        enhanced_image = ImageEnhance.Contrast(gray_image).enhance(contrast_factor)
        
        return enhanced_image
    
    @staticmethod
    def enhance_for_ocr_cv2(image: np.ndarray, contrast_factor: float = 2.0) -> np.ndarray:
        """使用OpenCV增强图像以提高OCR识别准确率
        
        处理流程：
        1. 转换为灰度图（如果是彩色图）
        2. 转换为PIL格式
        3. 增强对比度（默认2倍）
        4. 转换回numpy数组
        
        Args:
            image: OpenCV图像（numpy数组）
            contrast_factor: 对比度增强倍数，默认2.0
            
        Returns:
            增强后的numpy数组（灰度图）
        """
        if not HAS_CV2 or not HAS_PIL:
            raise ImportError("OpenCV或PIL/Pillow未安装")
        
        # 1. 转换为灰度图（如果是彩色图）
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 2. 转换为PIL格式
        pil_image = Image.fromarray(gray)
        
        # 3. 增强对比度
        enhanced = ImageEnhance.Contrast(pil_image).enhance(contrast_factor)
        
        # 4. 转换回numpy数组
        enhanced_array = np.array(enhanced)
        
        return enhanced_array
    
    @staticmethod
    def enhance_for_ocr(image: Union['Image.Image', np.ndarray], 
                       contrast_factor: float = 2.0) -> Union['Image.Image', np.ndarray]:
        """自动检测图像类型并增强
        
        根据输入图像类型自动选择合适的处理方法
        
        Args:
            image: PIL Image对象或OpenCV图像（numpy数组）
            contrast_factor: 对比度增强倍数，默认2.0
            
        Returns:
            增强后的图像（类型与输入相同）
        """
        if HAS_PIL and isinstance(image, Image.Image):
            return OCRImageProcessor.enhance_for_ocr_pil(image, contrast_factor)
        elif isinstance(image, np.ndarray):
            return OCRImageProcessor.enhance_for_ocr_cv2(image, contrast_factor)
        else:
            raise TypeError(f"不支持的图像类型: {type(image)}")


# 提供便捷的函数接口
def enhance_for_ocr(image: Union['Image.Image', np.ndarray], 
                   contrast_factor: float = 2.0) -> Union['Image.Image', np.ndarray]:
    """增强图像以提高OCR识别准确率（便捷函数）
    
    处理流程：
    1. 转换为灰度图
    2. 增强对比度（默认2倍）
    
    Args:
        image: PIL Image对象或OpenCV图像（numpy数组）
        contrast_factor: 对比度增强倍数，默认2.0
        
    Returns:
        增强后的图像（类型与输入相同）
        
    Example:
        # PIL图像
        from PIL import Image
        img = Image.open('screenshot.png')
        enhanced = enhance_for_ocr(img)
        
        # OpenCV图像
        import cv2
        img = cv2.imread('screenshot.png')
        enhanced = enhance_for_ocr(img)
    """
    return OCRImageProcessor.enhance_for_ocr(image, contrast_factor)
