"""
屏幕捕获与分析模块
Screen Capture and Analysis Module
"""

import os
import re
from typing import Optional, Tuple
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image

from .adb_bridge import ADBBridge
from .ocr_image_processor import enhance_for_ocr


class ScreenCapture:
    """屏幕捕获与分析器"""
    
    def __init__(self, adb_bridge: ADBBridge, screenshot_dir: str = "./screenshots"):
        """初始化屏幕捕获器
        
        Args:
            adb_bridge: ADB 桥接器实例
            screenshot_dir: 截图保存目录
        """
        self.adb_bridge = adb_bridge
        self.screenshot_dir = screenshot_dir
        Path(screenshot_dir).mkdir(parents=True, exist_ok=True)
    
    async def capture(self, device_id: str) -> Optional[np.ndarray]:
        """捕获屏幕并返回图像数组
        
        Args:
            device_id: 设备 ID
            
        Returns:
            OpenCV 图像数组 (BGR 格式)，失败返回 None
        """
        try:
            png_data = await self.adb_bridge.screencap(device_id)
            if not png_data:
                return None
            
            # 将 PNG 数据转换为 numpy 数组
            nparr = np.frombuffer(png_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        except Exception:
            return None
    
    async def save_screenshot(self, device_id: str, filename: str = None) -> Optional[str]:
        """保存截图到文件
        
        Args:
            device_id: 设备 ID
            filename: 文件名（可选，默认使用时间戳）
            
        Returns:
            保存的文件路径，失败返回 None
        """
        try:
            img = await self.capture(device_id)
            if img is None:
                return None
            
            if filename is None:
                import time
                filename = f"screenshot_{int(time.time())}.png"
            
            filepath = os.path.join(self.screenshot_dir, filename)
            cv2.imwrite(filepath, img)
            return filepath
        except Exception:
            return None

    async def find_image(self, device_id: str, template: np.ndarray, 
                         threshold: float = 0.8) -> Optional[Tuple[int, int]]:
        """在屏幕中查找模板图像
        
        Args:
            device_id: 设备 ID
            template: 模板图像（OpenCV 格式）
            threshold: 匹配阈值 (0-1)
            
        Returns:
            匹配位置的中心坐标 (x, y)，未找到返回 None
        """
        try:
            screen = await self.capture(device_id)
            if screen is None:
                return None
            
            # 模板匹配
            result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val >= threshold:
                # 计算中心坐标
                h, w = template.shape[:2]
                center_x = max_loc[0] + w // 2
                center_y = max_loc[1] + h // 2
                return (center_x, center_y)
            
            return None
        except Exception:
            return None
    
    async def find_image_from_file(self, device_id: str, template_path: str,
                                   threshold: float = 0.8) -> Optional[Tuple[int, int]]:
        """从文件加载模板并在屏幕中查找
        
        Args:
            device_id: 设备 ID
            template_path: 模板图像文件路径
            threshold: 匹配阈值
            
        Returns:
            匹配位置的中心坐标，未找到返回 None
        """
        try:
            template = cv2.imread(template_path)
            if template is None:
                return None
            return await self.find_image(device_id, template, threshold)
        except Exception:
            return None
    
    def compare_images(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """比较两张图片的相似度
        
        Args:
            img1: 第一张图片
            img2: 第二张图片
            
        Returns:
            相似度分数 (0-1)
        """
        try:
            # 确保图片大小相同
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
            # 转换为灰度图
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            # 计算结构相似度
            result = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)
            return float(result[0][0])
        except Exception:
            return 0.0

    async def ocr_region(self, device_id: str, 
                         region: Tuple[int, int, int, int]) -> str:
        """对指定区域进行 OCR 识别
        
        Args:
            device_id: 设备 ID
            region: 区域坐标 (x, y, width, height)
            
        Returns:
            识别的文字内容
        """
        try:
            screen = await self.capture(device_id)
            if screen is None:
                return ""
            
            x, y, w, h = region
            roi = screen[y:y+h, x:x+w]
            
            # 预处理图像以提高 OCR 准确率
            # 1. 使用OCR图像预处理模块增强图像
            enhanced_array = enhance_for_ocr(roi)
            
            # 2. 二值化
            _, binary = cv2.threshold(enhanced_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # OCR 识别（支持中英文）
            text = pytesseract.image_to_string(binary, lang='chi_sim+eng')
            return text.strip()
        except Exception:
            return ""
    
    async def ocr_full_screen(self, device_id: str) -> str:
        """对整个屏幕进行 OCR 识别
        
        Args:
            device_id: 设备 ID
            
        Returns:
            识别的文字内容
        """
        try:
            screen = await self.capture(device_id)
            if screen is None:
                return ""
            
            # 使用OCR图像预处理模块增强图像
            enhanced_array = enhance_for_ocr(screen)
            
            # OCR识别
            text = pytesseract.image_to_string(enhanced_array, lang='chi_sim+eng')
            return text.strip()
        except Exception:
            return ""
    
    async def extract_balance(self, device_id: str, 
                              region: Optional[Tuple[int, int, int, int]] = None) -> Optional[float]:
        """提取余额数字
        
        Args:
            device_id: 设备 ID
            region: 余额显示区域（可选）
            
        Returns:
            余额数值，提取失败返回 None
        """
        try:
            if region:
                text = await self.ocr_region(device_id, region)
            else:
                text = await self.ocr_full_screen(device_id)
            
            # 提取数字（支持小数）
            numbers = re.findall(r'\d+\.?\d*', text)
            if numbers:
                # 返回最大的数字（通常是余额）
                return max(float(n) for n in numbers)
            return None
        except Exception:
            return None
    
    async def find_text_location(self, device_id: str, target_text: str) -> Optional[Tuple[int, int]]:
        """查找文字在屏幕中的位置
        
        Args:
            device_id: 设备 ID
            target_text: 要查找的文字
            
        Returns:
            文字中心坐标，未找到返回 None
        """
        try:
            from rapidocr_onnxruntime import RapidOCR
            from PIL import Image
            from io import BytesIO
            
            # 获取截图
            screenshot_data = await self.adb_bridge.screencap(device_id)
            if not screenshot_data:
                return None
            
            # 转换为PIL Image
            img = Image.open(BytesIO(screenshot_data))
            
            # 转换为灰度图以提高OCR识别准确率
            # 灰度图可以减少颜色干扰，提高文字识别率
            gray_img = img.convert('L')
            
            # 使用 RapidOCR 识别文字
            ocr = RapidOCR()
            result, _ = ocr(gray_img)
            
            if not result:
                return None
            
            # 查找目标文字
            for item in result:
                text = item[1]  # 识别到的文字
                box = item[0]   # 文字位置 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                
                if target_text in text:
                    # 计算中心点
                    x_coords = [point[0] for point in box]
                    y_coords = [point[1] for point in box]
                    center_x = int(sum(x_coords) / len(x_coords))
                    center_y = int(sum(y_coords) / len(y_coords))
                    return (center_x, center_y)
            
            return None
        except Exception as e:
            print(f"[OCR错误] {e}")
            return None
    
    async def extract_all_text(self, device_id: str) -> str:
        """提取屏幕上的所有文字
        
        Args:
            device_id: 设备 ID
            
        Returns:
            所有识别到的文字
        """
        try:
            screen = await self.capture(device_id)
            if screen is None:
                return ""
            
            # 使用OCR图像预处理模块增强图像以提高OCR识别准确率
            enhanced_array = enhance_for_ocr(screen)
            
            # 使用 pytesseract 识别所有文字
            text = pytesseract.image_to_string(enhanced_array, lang='chi_sim+eng')
            return text.strip()
        except Exception:
            return ""
