"""
OCR识别增强模块 - 提供多种策略提高识别准确率
OCR Enhancer Module - Multiple strategies to improve recognition accuracy
"""

import asyncio
import re
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from .image_processor import ImageProcessor
from .ocr_thread_pool import get_ocr_pool, OCRResult


@dataclass
class EnhancedOCRResult:
    """增强的OCR识别结果"""
    text: str  # 最终识别的文本
    confidence: float  # 置信度 (0-1)
    method: str  # 使用的识别方法
    all_results: List[str]  # 所有尝试的结果
    success: bool  # 是否成功


class OCREnhancer:
    """OCR识别增强器 - 提供多种策略提高准确率"""
    
    def __init__(self):
        """初始化OCR增强器"""
        self._ocr_pool = get_ocr_pool() if HAS_PIL else None
    
    async def recognize_with_retry(self, 
                                   image: 'Image.Image',
                                   modes: List[str] = None,
                                   max_attempts: int = 3) -> EnhancedOCRResult:
        """使用多种预处理模式重试识别
        
        Args:
            image: PIL图像对象
            modes: 预处理模式列表，默认 ['balanced', 'text', 'high_contrast']
            max_attempts: 最大尝试次数
            
        Returns:
            EnhancedOCRResult: 增强的识别结果
        """
        if modes is None:
            modes = ['balanced', 'text', 'high_contrast']
        
        all_results = []
        best_result = None
        best_confidence = 0.0
        
        for attempt, mode in enumerate(modes[:max_attempts]):
            try:
                # 预处理图像
                enhanced = ImageProcessor.preprocess_for_ocr(image, mode=mode)
                
                # OCR识别
                ocr_result = await self._ocr_pool.recognize(enhanced, timeout=5.0)
                
                if ocr_result.texts:
                    text = ''.join(ocr_result.texts)
                    all_results.append(text)
                    
                    # 计算置信度（基于文本长度和scores）
                    confidence = self._calculate_confidence(ocr_result)
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_result = (text, mode)
                    
                    # 如果置信度很高，提前返回
                    if confidence > 0.9:
                        break
                        
            except Exception as e:
                print(f"  [OCR增强] 模式 {mode} 识别失败: {e}")
                continue
        
        if best_result:
            return EnhancedOCRResult(
                text=best_result[0],
                confidence=best_confidence,
                method=best_result[1],
                all_results=all_results,
                success=True
            )
        else:
            return EnhancedOCRResult(
                text='',
                confidence=0.0,
                method='none',
                all_results=all_results,
                success=False
            )
    
    async def recognize_number(self, 
                              image: 'Image.Image',
                              pattern: str = r'\d+\.?\d*') -> EnhancedOCRResult:
        """专门识别数字和金额
        
        使用数字模式预处理 + 正则验证
        
        Args:
            image: PIL图像对象
            pattern: 数字匹配正则表达式
            
        Returns:
            EnhancedOCRResult: 识别结果
        """
        # 使用数字模式预处理
        enhanced = ImageProcessor.preprocess_for_ocr(image, mode='number')
        
        # OCR识别
        ocr_result = await self._ocr_pool.recognize(enhanced, timeout=5.0)
        
        if not ocr_result.texts:
            return EnhancedOCRResult(
                text='',
                confidence=0.0,
                method='number',
                all_results=[],
                success=False
            )
        
        # 提取所有数字
        text = ''.join(ocr_result.texts)
        numbers = re.findall(pattern, text)
        
        if numbers:
            # 返回第一个匹配的数字
            return EnhancedOCRResult(
                text=numbers[0],
                confidence=0.8,
                method='number',
                all_results=numbers,
                success=True
            )
        else:
            return EnhancedOCRResult(
                text=text,
                confidence=0.3,
                method='number',
                all_results=[text],
                success=False
            )
    
    async def recognize_amount(self, 
                              image: 'Image.Image',
                              min_value: float = 0.01,
                              max_value: float = 100.0) -> Optional[float]:
        """专门识别金额（带范围验证）
        
        优先识别带有货币符号或"元"字的金额，避免误识别其他数字
        
        Args:
            image: PIL图像对象
            min_value: 最小金额
            max_value: 最大金额
            
        Returns:
            float: 识别的金额，失败返回None
        """
        # 使用数字模式预处理
        enhanced = ImageProcessor.preprocess_for_ocr(image, mode='number')
        
        # OCR识别
        ocr_result = await self._ocr_pool.recognize(enhanced, timeout=5.0)
        
        if not ocr_result.texts:
            return None
        
        # 合并所有文本
        full_text = ''.join(ocr_result.texts)
        
        # 策略1: 优先查找带有货币符号的金额（¥、￥、$）
        currency_patterns = [
            r'[¥￥$]\s*(\d+\.?\d*)',  # ¥3.5 或 ￥3.5
            r'(\d+\.?\d*)\s*[¥￥$]',  # 3.5¥
        ]
        
        for pattern in currency_patterns:
            matches = re.findall(pattern, full_text)
            if matches:
                for match in matches:
                    try:
                        amount = float(match)
                        if min_value <= amount <= max_value:
                            return amount
                    except ValueError:
                        continue
        
        # 策略2: 查找带有"元"字的金额
        yuan_patterns = [
            r'(\d+\.?\d*)\s*元',  # 3.5元
            r'元\s*(\d+\.?\d*)',  # 元3.5（不常见）
        ]
        
        for pattern in yuan_patterns:
            matches = re.findall(pattern, full_text)
            if matches:
                for match in matches:
                    try:
                        amount = float(match)
                        if min_value <= amount <= max_value:
                            return amount
                    except ValueError:
                        continue
        
        # 策略3: 查找所有小数（优先选择在合理范围内的）
        decimal_pattern = r'\d+\.\d+'
        decimals = re.findall(decimal_pattern, full_text)
        
        if decimals:
            # 优先返回在范围内的第一个小数
            for decimal in decimals:
                try:
                    amount = float(decimal)
                    if min_value <= amount <= max_value:
                        return amount
                except ValueError:
                    continue
        
        # 策略4: 查找所有整数（作为最后的后备）
        integer_pattern = r'\d+'
        integers = re.findall(integer_pattern, full_text)
        
        if integers:
            # 优先返回在范围内的第一个整数
            for integer in integers:
                try:
                    amount = float(integer)
                    if min_value <= amount <= max_value:
                        return amount
                except ValueError:
                    continue
        
        # 所有策略都失败
        print(f"  [OCR增强] 未找到有效金额，识别文本: {full_text}")
        return None
    
    async def recognize_with_region(self,
                                   image: 'Image.Image',
                                   region: Tuple[int, int, int, int],
                                   mode: str = 'balanced') -> EnhancedOCRResult:
        """识别指定区域
        
        Args:
            image: PIL图像对象
            region: 区域坐标 (x, y, width, height)
            mode: 预处理模式
            
        Returns:
            EnhancedOCRResult: 识别结果
        """
        # 裁剪区域
        x, y, w, h = region
        cropped = ImageProcessor.crop_region(image, x, y, w, h)
        
        # 预处理
        enhanced = ImageProcessor.preprocess_for_ocr(cropped, mode=mode)
        
        # OCR识别
        ocr_result = await self._ocr_pool.recognize(enhanced, timeout=5.0)
        
        if ocr_result.texts:
            text = ''.join(ocr_result.texts)
            confidence = self._calculate_confidence(ocr_result)
            
            return EnhancedOCRResult(
                text=text,
                confidence=confidence,
                method=f'region_{mode}',
                all_results=[text],
                success=True
            )
        else:
            return EnhancedOCRResult(
                text='',
                confidence=0.0,
                method=f'region_{mode}',
                all_results=[],
                success=False
            )
    
    async def recognize_multi_scale(self,
                                   image: 'Image.Image',
                                   scales: List[float] = None) -> EnhancedOCRResult:
        """多尺度识别（尝试不同缩放比例）
        
        Args:
            image: PIL图像对象
            scales: 缩放比例列表
            
        Returns:
            EnhancedOCRResult: 识别结果
        """
        if scales is None:
            scales = [0.8, 1.0, 1.2]
        
        all_results = []
        best_result = None
        best_confidence = 0.0
        
        for scale in scales:
            try:
                # 缩放图像
                width, height = image.size
                new_size = (int(width * scale), int(height * scale))
                scaled = image.resize(new_size, Image.LANCZOS)
                
                # 预处理
                enhanced = ImageProcessor.preprocess_for_ocr(scaled, mode='balanced')
                
                # OCR识别
                ocr_result = await self._ocr_pool.recognize(enhanced, timeout=5.0)
                
                if ocr_result.texts:
                    text = ''.join(ocr_result.texts)
                    all_results.append(text)
                    
                    confidence = self._calculate_confidence(ocr_result)
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_result = (text, scale)
                        
            except Exception as e:
                print(f"  [OCR增强] 缩放 {scale} 识别失败: {e}")
                continue
        
        if best_result:
            return EnhancedOCRResult(
                text=best_result[0],
                confidence=best_confidence,
                method=f'scale_{best_result[1]}',
                all_results=all_results,
                success=True
            )
        else:
            return EnhancedOCRResult(
                text='',
                confidence=0.0,
                method='multi_scale',
                all_results=all_results,
                success=False
            )
    
    async def find_text_with_fuzzy(self,
                                  image: 'Image.Image',
                                  target_text: str,
                                  similarity_threshold: float = 0.7) -> bool:
        """模糊匹配文本（容错识别）
        
        Args:
            image: PIL图像对象
            target_text: 目标文本
            similarity_threshold: 相似度阈值 (0-1)
            
        Returns:
            bool: 是否找到匹配文本
        """
        # 使用文本模式识别
        enhanced = ImageProcessor.preprocess_for_ocr(image, mode='text')
        ocr_result = await self._ocr_pool.recognize(enhanced, timeout=5.0)
        
        if not ocr_result.texts:
            return False
        
        # 检查每个识别的文本
        for text in ocr_result.texts:
            similarity = self._calculate_similarity(text, target_text)
            if similarity >= similarity_threshold:
                return True
        
        return False
    
    def _calculate_confidence(self, ocr_result: OCRResult) -> float:
        """计算识别置信度
        
        Args:
            ocr_result: OCR识别结果
            
        Returns:
            float: 置信度 (0-1)
        """
        if not ocr_result.texts:
            return 0.0
        
        # 如果有scores，使用平均分数
        if ocr_result.scores:
            try:
                avg_score = sum(ocr_result.scores) / len(ocr_result.scores)
                return min(avg_score, 1.0)
            except:
                pass
        
        # 否则基于文本长度估算
        total_length = sum(len(text) for text in ocr_result.texts)
        if total_length > 0:
            return min(0.5 + total_length * 0.05, 0.9)
        
        return 0.3
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的相似度（简单版本）
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            float: 相似度 (0-1)
        """
        # 移除空格
        text1 = text1.replace(' ', '')
        text2 = text2.replace(' ', '')
        
        # 完全匹配
        if text1 == text2:
            return 1.0
        
        # 包含关系
        if text2 in text1 or text1 in text2:
            return 0.8
        
        # 计算字符重叠率
        set1 = set(text1)
        set2 = set(text2)
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0


# 全局单例
_ocr_enhancer = None


def get_ocr_enhancer() -> OCREnhancer:
    """获取全局OCR增强器实例
    
    Returns:
        OCREnhancer: 全局单例实例
    """
    global _ocr_enhancer
    if _ocr_enhancer is None:
        _ocr_enhancer = OCREnhancer()
    return _ocr_enhancer
