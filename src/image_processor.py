"""
图像预处理模块 - 提升 OCR 识别准确率
Image Processor Module
"""

try:
    from PIL import Image, ImageEnhance, ImageFilter
    import numpy as np
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class ImageProcessor:
    """图像预处理器 - 简单实用的图像增强"""
    
    @staticmethod
    def preprocess_for_ocr(image: 'Image.Image', mode: str = 'balanced') -> 'Image.Image':
        """为 OCR 预处理图像
        
        Args:
            image: PIL Image 对象
            mode: 预处理模式
                - 'balanced': 平衡模式（默认，适合大多数情况）
                - 'high_contrast': 高对比度模式（适合低对比度文字）
                - 'denoise': 去噪模式（适合有噪点的图像）
                - 'text': 文本模式（专门优化文字识别）
                - 'number': 数字模式（专门优化数字识别）
        
        Returns:
            处理后的图像
        """
        if not HAS_PIL:
            return image
        
        try:
            # 1. 转换为灰度图
            if image.mode != 'L':
                image = image.convert('L')
            
            if mode == 'high_contrast':
                # 高对比度模式
                # 增强对比度
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(2.0)
                
                # 锐化
                image = image.filter(ImageFilter.SHARPEN)
                
                # 二值化（阈值法）
                image = ImageProcessor._binarize(image, threshold=128)
            
            elif mode == 'denoise':
                # 去噪模式
                # 中值滤波去噪
                image = image.filter(ImageFilter.MedianFilter(size=3))
                
                # 轻微增强对比度
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.5)
            
            elif mode == 'text':
                # 文本模式 - 优化中文文字识别
                # 1. 去噪
                image = image.filter(ImageFilter.MedianFilter(size=3))
                
                # 2. 增强对比度
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.8)
                
                # 3. 增强锐度
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(2.0)
                
                # 4. 自适应二值化
                image = ImageProcessor._adaptive_binarize(image)
            
            elif mode == 'number':
                # 数字模式 - 专门优化数字和金额识别
                # 1. 强对比度
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(2.5)
                
                # 2. 双重锐化
                image = image.filter(ImageFilter.SHARPEN)
                image = image.filter(ImageFilter.SHARPEN)
                
                # 3. 高阈值二值化（数字通常是深色）
                image = ImageProcessor._binarize(image, threshold=140)
                
                # 4. 形态学处理（去除小噪点）
                image = ImageProcessor._morphology_clean(image)
            
            else:  # balanced
                # 平衡模式（默认）
                # 轻微增强对比度
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.3)
                
                # 轻微锐化
                image = image.filter(ImageFilter.SHARPEN)
            
            return image
        except Exception:
            # 预处理失败，返回原图
            return image
    
    @staticmethod
    def _binarize(image: 'Image.Image', threshold: int = 128) -> 'Image.Image':
        """二值化图像
        
        Args:
            image: 灰度图像
            threshold: 阈值（0-255）
        
        Returns:
            二值化后的图像
        """
        try:
            # 转换为 numpy 数组
            img_array = np.array(image)
            
            # 二值化
            binary_array = np.where(img_array > threshold, 255, 0).astype(np.uint8)
            
            # 转回 PIL Image
            return Image.fromarray(binary_array, mode='L')
        except Exception:
            return image
    
    @staticmethod
    def crop_region(image: 'Image.Image', x: int, y: int, width: int, height: int) -> 'Image.Image':
        """裁剪图像区域
        
        Args:
            image: PIL Image 对象
            x, y: 左上角坐标
            width, height: 宽度和高度
        
        Returns:
            裁剪后的图像
        """
        try:
            return image.crop((x, y, x + width, y + height))
        except Exception:
            return image
    
    @staticmethod
    def enhance_text_region(image: 'Image.Image') -> 'Image.Image':
        """增强文本区域（专门用于识别金额、数字等）
        
        Args:
            image: PIL Image 对象
        
        Returns:
            增强后的图像
        """
        if not HAS_PIL:
            return image
        
        try:
            # 转换为灰度图
            if image.mode != 'L':
                image = image.convert('L')
            
            # 强化对比度
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.5)
            
            # 锐化
            image = image.filter(ImageFilter.SHARPEN)
            image = image.filter(ImageFilter.SHARPEN)  # 二次锐化
            
            # 二值化
            image = ImageProcessor._binarize(image, threshold=140)
            
            return image
        except Exception:
            return image
    
    @staticmethod
    def _adaptive_binarize(image: 'Image.Image', block_size: int = 15) -> 'Image.Image':
        """自适应二值化（Otsu方法的简化版）
        
        Args:
            image: 灰度图像
            block_size: 块大小
        
        Returns:
            二值化后的图像
        """
        try:
            img_array = np.array(image)
            
            # 计算全局阈值（使用平均值）
            threshold = int(np.mean(img_array))
            
            # 二值化
            binary_array = np.where(img_array > threshold, 255, 0).astype(np.uint8)
            
            return Image.fromarray(binary_array, mode='L')
        except Exception:
            return image
    
    @staticmethod
    def _morphology_clean(image: 'Image.Image', iterations: int = 1) -> 'Image.Image':
        """形态学处理 - 去除小噪点
        
        Args:
            image: 二值图像
            iterations: 迭代次数
        
        Returns:
            处理后的图像
        """
        try:
            # 使用开运算（先腐蚀后膨胀）去除小噪点
            for _ in range(iterations):
                image = image.filter(ImageFilter.MinFilter(3))  # 腐蚀
                image = image.filter(ImageFilter.MaxFilter(3))  # 膨胀
            return image
        except Exception:
            return image
    
    @staticmethod
    def resize_for_ocr(image: 'Image.Image', target_height: int = 64) -> 'Image.Image':
        """调整图像大小以优化OCR识别
        
        OCR引擎对图像大小敏感,太小识别不准,太大速度慢
        建议文字高度在32-64像素之间
        
        Args:
            image: PIL Image 对象
            target_height: 目标高度（像素）
        
        Returns:
            调整后的图像
        """
        try:
            width, height = image.size
            
            # 如果图像太小,放大
            if height < target_height:
                scale = target_height / height
                new_width = int(width * scale)
                new_height = target_height
                image = image.resize((new_width, new_height), Image.LANCZOS)
            
            # 如果图像太大,缩小
            elif height > target_height * 3:
                scale = (target_height * 2) / height
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = image.resize((new_width, new_height), Image.LANCZOS)
            
            return image
        except Exception:
            return image
    
    @staticmethod
    def multi_scale_process(image: 'Image.Image', scales: list = None) -> list:
        """多尺度处理 - 生成不同尺度的图像用于OCR
        
        有时候OCR在不同尺度下识别效果不同,可以尝试多个尺度
        
        Args:
            image: PIL Image 对象
            scales: 缩放比例列表,默认 [0.8, 1.0, 1.2]
        
        Returns:
            list: 不同尺度的图像列表
        """
        if scales is None:
            scales = [0.8, 1.0, 1.2]
        
        images = []
        try:
            width, height = image.size
            for scale in scales:
                new_width = int(width * scale)
                new_height = int(height * scale)
                scaled_image = image.resize((new_width, new_height), Image.LANCZOS)
                images.append(scaled_image)
        except Exception:
            images = [image]
        
        return images
