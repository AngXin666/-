"""
增强的广告页检测器
Enhanced Ad Detector
"""

import asyncio
import os
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
from dataclasses import dataclass

# 导入OCR图像预处理模块
from ..ocr_image_processor import enhance_for_ocr


@dataclass
class AdDetectionResult:
    """广告检测结果"""
    is_ad: bool  # 是否为广告页
    confidence: float  # 置信度 (0.0-1.0)
    details: Dict[str, Any]  # 检测详情
    method: str  # 检测方法 (template/ocr/layout/hybrid)


class EnhancedAdDetector:
    """增强的广告页检测器
    
    使用多种特征识别广告页：
    1. 模板匹配 - 使用SSIM算法匹配已知广告页模板
    2. OCR关键词 - 识别"跳过"、"关闭"、"秒后"等关键词
    3. 布局特征 - 检测全屏图片、视频播放器等特征
    4. 组合判断 - 综合多种特征计算总置信度
    """
    
    # OCR关键词模式
    SKIP_KEYWORDS = ["跳过", "Skip", "SKIP", "跳过广告"]
    CLOSE_KEYWORDS = ["关闭", "×", "X", "Close"]
    COUNTDOWN_KEYWORDS = ["秒后", "秒", "s后", "S后"]
    AD_KEYWORDS = ["广告", "AD", "推广", "赞助", "Sponsored"]
    
    # 置信度阈值
    TEMPLATE_THRESHOLD = 0.7  # 模板匹配阈值
    OCR_THRESHOLD = 0.6  # OCR关键词阈值
    LAYOUT_THRESHOLD = 0.5  # 布局特征阈值
    COMBINED_THRESHOLD = 0.6  # 组合判断阈值
    
    # 权重配置
    TEMPLATE_WEIGHT = 0.4  # 模板匹配权重
    OCR_WEIGHT = 0.4  # OCR关键词权重
    LAYOUT_WEIGHT = 0.2  # 布局特征权重
    
    def __init__(self, adb_bridge=None, template_dir: str = "templates/ads"):
        """初始化增强广告检测器
        
        Args:
            adb_bridge: ADB桥接器实例
            template_dir: 广告页模板目录
        """
        # 禁用所有OCR相关的日志输出
        import logging
        logging.getLogger('RapidOCR').setLevel(logging.CRITICAL)
        logging.getLogger('ppocr').setLevel(logging.CRITICAL)
        logging.getLogger('onnxruntime').setLevel(logging.CRITICAL)
        
        self.adb = adb_bridge
        self.template_dir = template_dir
        self._templates = []
        self._load_templates()
    
    def _load_templates(self):
        """加载广告页模板"""
        if not os.path.exists(self.template_dir):
            print(f"[广告检测] 模板目录不存在: {self.template_dir}")
            return
        
        try:
            from PIL import Image
            import glob
            
            template_files = glob.glob(os.path.join(self.template_dir, "*.png"))
            template_files.extend(glob.glob(os.path.join(self.template_dir, "*.jpg")))
            
            for template_path in template_files:
                try:
                    img = Image.open(template_path)
                    self._templates.append({
                        'path': template_path,
                        'image': np.array(img),
                        'name': os.path.basename(template_path)
                    })
                except Exception as e:
                    print(f"[广告检测] 加载模板失败 {template_path}: {e}")
            
            print(f"[广告检测] 已加载 {len(self._templates)} 个广告页模板")
        except ImportError:
            print(f"[广告检测] PIL未安装，无法加载模板")
    
    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """计算两张图片的SSIM相似度
        
        Args:
            img1: 图片1
            img2: 图片2
            
        Returns:
            SSIM相似度 (0.0-1.0)
        """
        try:
            from skimage.metrics import structural_similarity as ssim
            from PIL import Image
            
            # 转换为灰度图
            if len(img1.shape) == 3:
                img1_gray = np.mean(img1, axis=2).astype(np.uint8)
            else:
                img1_gray = img1
            
            if len(img2.shape) == 3:
                img2_gray = np.mean(img2, axis=2).astype(np.uint8)
            else:
                img2_gray = img2
            
            # 调整大小使其一致
            if img1_gray.shape != img2_gray.shape:
                img1_pil = Image.fromarray(img1_gray)
                img1_pil = img1_pil.resize((img2_gray.shape[1], img2_gray.shape[0]))
                img1_gray = np.array(img1_pil)
            
            # 计算SSIM
            similarity = ssim(img1_gray, img2_gray)
            return float(similarity)
        except ImportError:
            print(f"[广告检测] scikit-image未安装，无法计算SSIM")
            return 0.0
        except Exception as e:
            print(f"[广告检测] SSIM计算失败: {e}")
            return 0.0
    
    async def _detect_by_template(self, screenshot: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """使用模板匹配检测广告页
        
        Args:
            screenshot: 屏幕截图
            
        Returns:
            (置信度, 详情)
        """
        if not self._templates:
            return 0.0, {'method': 'template', 'message': '无可用模板'}
        
        max_similarity = 0.0
        best_match = None
        
        for template in self._templates:
            similarity = self._calculate_ssim(screenshot, template['image'])
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = template['name']
        
        details = {
            'method': 'template',
            'max_similarity': max_similarity,
            'best_match': best_match,
            'threshold': self.TEMPLATE_THRESHOLD
        }
        
        return max_similarity, details
    
    async def _detect_by_ocr(self, screenshot: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """使用OCR关键词检测广告页
        
        Args:
            screenshot: 屏幕截图
            
        Returns:
            (置信度, 详情)
        """
        try:
            # 禁用RapidOCR的日志输出
            import logging
            logging.getLogger('RapidOCR').setLevel(logging.CRITICAL)
            logging.getLogger('ppocr').setLevel(logging.CRITICAL)
            
            from rapidocr import RapidOCR
            from PIL import Image
            
            # 转换为PIL Image
            if isinstance(screenshot, np.ndarray):
                img = Image.fromarray(screenshot)
            else:
                img = screenshot
            
            # 图像预处理以提高OCR识别准确率
            # 1. 转换为灰度图并增强对比度
            gray_img = img.convert('L')
            gray_img = enhance_for_ocr(gray_img)
            
            # 2. 可选：二值化处理（对于低对比度文字效果更好）
            # gray_array = np.array(gray_img)
            # threshold = 128
            # binary_array = np.where(gray_array > threshold, 255, 0).astype(np.uint8)
            # gray_img = Image.fromarray(binary_array)
            
            # 执行OCR
            ocr = RapidOCR()
            ocr_result = ocr(gray_img)
            
            # 获取识别结果 - RapidOCROutput对象
            if not ocr_result or not hasattr(ocr_result, 'txts'):
                return 0.0, {'method': 'ocr', 'message': '未识别到文字', 'total_texts': 0, 'all_texts': []}
            
            # 提取所有文本
            texts = ocr_result.txts
            scores_list = ocr_result.scores if hasattr(ocr_result, 'scores') else []
            
            # 检测关键词
            found_keywords = {
                'skip': [],
                'close': [],
                'countdown': [],
                'ad': []
            }
            
            for text in texts:
                text_upper = text.upper()
                
                # 跳过关键词
                for keyword in self.SKIP_KEYWORDS:
                    if keyword.upper() in text_upper:
                        found_keywords['skip'].append(text)
                
                # 关闭关键词
                for keyword in self.CLOSE_KEYWORDS:
                    if keyword.upper() in text_upper:
                        found_keywords['close'].append(text)
                
                # 倒计时关键词
                for keyword in self.COUNTDOWN_KEYWORDS:
                    if keyword in text:
                        found_keywords['countdown'].append(text)
                
                # 广告关键词
                for keyword in self.AD_KEYWORDS:
                    if keyword.upper() in text_upper:
                        found_keywords['ad'].append(text)
            
            # 计算置信度
            # 核心规则：必须同时有"跳过"和"关闭"才是广告
            score = 0.0
            
            # 检查是否同时存在跳过和广告标识
            has_skip = bool(found_keywords['skip'])
            has_ad = bool(found_keywords['ad'])
            
            if has_skip and has_ad:
                # 同时有跳过和广告，基础分0.8
                score = 0.8
                
                # 如果还有关闭按钮，额外加0.1
                if found_keywords['close']:
                    score += 0.1
                
                # 如果还有倒计时，额外加0.1
                if found_keywords['countdown']:
                    score += 0.1
            else:
                # 不同时具备跳过和广告，不是广告
                score = 0.0
            
            # 限制在0-1范围
            score = min(score, 1.0)
            
            details = {
                'method': 'ocr',
                'found_keywords': found_keywords,
                'score': score,
                'threshold': self.OCR_THRESHOLD,
                'total_texts': len(texts),
                'all_texts': texts  # 添加所有识别的文本
            }
            
            return score, details
            
        except ImportError:
            return 0.0, {'method': 'ocr', 'message': 'RapidOCR未安装'}
        except Exception as e:
            return 0.0, {'method': 'ocr', 'message': f'OCR失败: {str(e)}'}
    
    async def _detect_by_layout(self, screenshot: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """使用布局特征检测广告页
        
        Args:
            screenshot: 屏幕截图
            
        Returns:
            (置信度, 详情)
        """
        try:
            from PIL import Image
            import cv2
            
            # 转换为OpenCV格式
            if isinstance(screenshot, Image.Image):
                screenshot = np.array(screenshot)
            
            height, width = screenshot.shape[:2]
            
            # 特征1: 检测全屏图片（颜色分布均匀度低）
            # 计算颜色直方图的标准差
            if len(screenshot.shape) == 3:
                gray = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
            else:
                gray = screenshot
            
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_std = np.std(hist)
            
            # 标准差越大，说明颜色分布越不均匀，越可能是广告图片
            fullscreen_score = min(hist_std / 1000.0, 1.0)
            
            # 特征2: 检测边缘密度（广告通常边缘较多）
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (width * height)
            edge_score = min(edge_density * 10, 1.0)
            
            # 特征3: 检测亮度分布（广告通常较亮）
            mean_brightness = np.mean(gray)
            brightness_score = mean_brightness / 255.0
            
            # 综合评分
            layout_score = (fullscreen_score * 0.4 + edge_score * 0.3 + brightness_score * 0.3)
            
            details = {
                'method': 'layout',
                'fullscreen_score': fullscreen_score,
                'edge_score': edge_score,
                'brightness_score': brightness_score,
                'combined_score': layout_score,
                'threshold': self.LAYOUT_THRESHOLD
            }
            
            return layout_score, details
            
        except ImportError:
            return 0.0, {'method': 'layout', 'message': 'OpenCV未安装'}
        except Exception as e:
            return 0.0, {'method': 'layout', 'message': f'布局检测失败: {str(e)}'}
    
    async def detect_ad_page(
        self,
        device_id: str,
        screenshot: Optional[np.ndarray] = None
    ) -> AdDetectionResult:
        """检测是否为广告页（组合判断）
        
        Args:
            device_id: 设备ID
            screenshot: 屏幕截图（可选，如果不提供则自动获取）
            
        Returns:
            广告检测结果
        """
        # 获取截图
        if screenshot is None:
            if self.adb is None:
                return AdDetectionResult(
                    is_ad=False,
                    confidence=0.0,
                    details={'error': 'ADB桥接器未初始化'},
                    method='none'
                )
            
            try:
                from PIL import Image
                from io import BytesIO
                
                screenshot_data = await self.adb.screencap(device_id)
                if not screenshot_data:
                    return AdDetectionResult(
                        is_ad=False,
                        confidence=0.0,
                        details={'error': '获取截图失败'},
                        method='none'
                    )
                
                screenshot = np.array(Image.open(BytesIO(screenshot_data)))
            except Exception as e:
                return AdDetectionResult(
                    is_ad=False,
                    confidence=0.0,
                    details={'error': f'截图处理失败: {str(e)}'},
                    method='none'
                )
        
        # 执行三种检测方法
        template_score, template_details = await self._detect_by_template(screenshot)
        ocr_score, ocr_details = await self._detect_by_ocr(screenshot)
        layout_score, layout_details = await self._detect_by_layout(screenshot)
        
        # OCR优先判断：如果OCR检测到"跳过"和"广告"，直接判定为广告
        # 这是最可靠的判断依据
        if ocr_score >= self.OCR_THRESHOLD:
            is_ad = True
            total_score = ocr_score
            primary_method = 'ocr'
            scores = {
                'template': template_score,
                'ocr': ocr_score,
                'layout': layout_score
            }
        else:
            # 如果OCR未达到阈值，使用加权综合判断
            total_score = (
                template_score * self.TEMPLATE_WEIGHT +
                ocr_score * self.OCR_WEIGHT +
                layout_score * self.LAYOUT_WEIGHT
            )
            is_ad = total_score >= self.COMBINED_THRESHOLD
            
            # 确定主要检测方法
            scores = {
                'template': template_score,
                'ocr': ocr_score,
                'layout': layout_score
            }
            primary_method = max(scores, key=scores.get)
        
        # 组合详情
        combined_details = {
            'template': template_details,
            'ocr': ocr_details,
            'layout': layout_details,
            'scores': scores,
            'weights': {
                'template': self.TEMPLATE_WEIGHT,
                'ocr': self.OCR_WEIGHT,
                'layout': self.LAYOUT_WEIGHT
            },
            'total_score': total_score,
            'threshold': self.COMBINED_THRESHOLD
        }
        
        return AdDetectionResult(
            is_ad=is_ad,
            confidence=total_score,
            details=combined_details,
            method=f'hybrid({primary_method})'
        )
