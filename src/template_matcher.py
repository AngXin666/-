"""
模板匹配模块 - 使用截图模板识别页面
Template Matcher Module - Identify pages using screenshot templates
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
from io import BytesIO

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class TemplateMatcher:
    """模板匹配器 - 通过对比截图模板识别页面"""
    
    # 需要排除的动态区域（轮播图等）- 针对 540x960 分辨率
    # 格式：{'模板名称': [(x, y, width, height), ...]}
    EXCLUDE_REGIONS = {
        '首页.png': [(0, 100, 540, 200)],  # 排除顶部轮播图区域
        # 注意：文章详情页不使用模板匹配，因为内容变化太大
        # 改用 OCR 识别（通过"返回"按钮等关键词）
    }
    
    # 关键特征区域（只匹配这些区域，提升速度和准确率）
    # 格式：{'模板名称': [(x, y, width, height), ...]}
    # 这些区域是页面独有的、不变的特征（如搜索框、品牌标题、导航栏等）
    KEY_FEATURE_REGIONS = {
        '首页.png': [
            (0, 0, 540, 100),      # 顶部区域：品牌标题 + 搜索框
            (0, 920, 540, 40),     # 底部导航栏
        ],
        '已登陆个人页.png': [
            (0, 0, 540, 150),      # 顶部个人信息区域
            (0, 920, 540, 40),     # 底部导航栏
        ],
        '未登陆个人页.png': [
            (0, 0, 540, 150),      # 顶部区域
            (0, 920, 540, 40),     # 底部导航栏
        ],
    }
    
    def __init__(self, template_dir: str = "dist/JT"):
        """初始化模板匹配器
        
        Args:
            template_dir: 模板图片目录
        """
        self.template_dir = Path(template_dir)
        self.templates = {}
        self._load_templates()
    
    def _load_templates(self):
        """加载所有模板图片（支持中文路径和加密文件）"""
        if not self.template_dir.exists():
            print(f"  [模板匹配] 模板目录不存在: {self.template_dir}")
            return
        
        # 支持的图片格式
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
        
        for image_file in self.template_dir.iterdir():
            # 检查是否是加密文件
            is_encrypted = image_file.suffix == '.encrypted'
            
            # 获取原始扩展名
            if is_encrypted:
                original_ext = image_file.stem.split('.')[-1] if '.' in image_file.stem else ''
                original_ext = f'.{original_ext}'
            else:
                original_ext = image_file.suffix.lower()
            
            # 只处理图片文件
            if original_ext not in image_extensions:
                continue
            
            try:
                # 读取文件数据
                with open(image_file, 'rb') as f:
                    image_data = f.read()
                
                # 如果是加密文件，先解密
                if is_encrypted:
                    try:
                        # 尝试绝对导入（兼容打包后的 EXE）
                        try:
                            from crypto_utils import crypto
                        except ImportError:
                            from .crypto_utils import crypto
                        
                        image_data = crypto.decrypt_file_content(image_data)
                    except Exception as e:
                        print(f"  [模板匹配] 解密失败 {image_file.name}: {e}")
                        continue
                
                # 将字节数据转换为 numpy 数组
                nparr = np.frombuffer(image_data, np.uint8)
                # 解码图片（保持原始彩色格式）
                template = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if template is not None:
                    # 使用原始文件名（不含 .encrypted 后缀）作为模板名称
                    if is_encrypted:
                        template_name = image_file.stem  # 去掉 .encrypted
                    else:
                        template_name = image_file.name
                    
                    self.templates[template_name] = template
                    # print(f"  [模板匹配] 已加载模板: {template_name}")
            except Exception as e:
                print(f"  [模板匹配] 加载模板失败 {image_file.name}: {e}")
        
        if self.templates:
            print(f"  [模板匹配] 共加载 {len(self.templates)} 个模板")
        else:
            print(f"  [模板匹配] 未找到任何模板图片")
    
    def match_screenshot(self, screenshot_data: bytes, threshold: float = 0.7) -> Optional[dict]:
        """匹配截图与模板（返回最佳匹配）
        
        Args:
            screenshot_data: 截图数据（bytes）
            threshold: 匹配阈值（0-1），默认0.7
            
        Returns:
            {'template_name': str, 'similarity': float} 或 None
        """
        all_matches = self.match_all_templates(screenshot_data)
        
        if all_matches and all_matches[0]['similarity'] >= threshold:
            return all_matches[0]
        
        return None
    
    def match_with_priority(self, screenshot_data: bytes, 
                           priority_templates: List[str],
                           threshold: float = 0.7) -> Optional[dict]:
        """按优先级列表匹配模板（只匹配指定的模板，按顺序）
        
        Args:
            screenshot_data: 截图数据（bytes）
            priority_templates: 优先匹配的模板名称列表（按优先级排序）
            threshold: 匹配阈值（0-1），默认0.7
            
        Returns:
            {'template_name': str, 'similarity': float} 或 None
        """
        if not self.templates or not HAS_PIL:
            return None
        
        try:
            # 将截图转换为 OpenCV 格式
            image = Image.open(BytesIO(screenshot_data))
            screenshot = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # 按优先级顺序匹配
            for template_name in priority_templates:
                if template_name not in self.templates:
                    continue
                
                template = self.templates[template_name]
                
                # 确保模板和截图尺寸一致
                if template.shape != screenshot.shape:
                    template_resized = cv2.resize(template, (screenshot.shape[1], screenshot.shape[0]))
                else:
                    template_resized = template
                
                # 计算相似度（优先使用关键特征区域）
                key_regions = self.KEY_FEATURE_REGIONS.get(template_name)
                exclude_regions = self.EXCLUDE_REGIONS.get(template_name)
                
                similarity = self._calculate_similarity(
                    screenshot, 
                    template_resized,
                    key_feature_regions=key_regions,
                    exclude_regions=exclude_regions
                )
                
                # 如果达到阈值，立即返回
                if similarity >= threshold:
                    return {
                        'template_name': template_name,
                        'similarity': similarity
                    }
            
            return None
                
        except Exception as e:
            print(f"  [模板匹配] 优先级匹配失败: {e}")
            return None
    
    def match_all_templates(self, screenshot_data: bytes) -> List[dict]:
        """匹配截图与所有模板（返回所有结果，按相似度排序）
        
        Args:
            screenshot_data: 截图数据（bytes）
            
        Returns:
            [{'template_name': str, 'similarity': float}, ...] 按相似度降序排列
        """
        if not self.templates:
            return []
        
        if not HAS_PIL:
            return []
        
        try:
            # 将截图转换为 OpenCV 格式
            image = Image.open(BytesIO(screenshot_data))
            screenshot = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            matches = []
            
            # 遍历所有模板
            for template_name, template in self.templates.items():
                # 确保模板和截图尺寸一致
                if template.shape != screenshot.shape:
                    # 调整模板大小以匹配截图
                    template_resized = cv2.resize(template, (screenshot.shape[1], screenshot.shape[0]))
                else:
                    template_resized = template
                
                # 计算相似度（优先使用关键特征区域）
                key_regions = self.KEY_FEATURE_REGIONS.get(template_name)
                exclude_regions = self.EXCLUDE_REGIONS.get(template_name)
                
                similarity = self._calculate_similarity(
                    screenshot, 
                    template_resized,
                    key_feature_regions=key_regions,
                    exclude_regions=exclude_regions
                )
                
                matches.append({
                    'template_name': template_name,
                    'similarity': similarity
                })
            
            # 按相似度降序排序
            matches.sort(key=lambda x: x['similarity'], reverse=True)
            
            return matches
                
        except Exception as e:
            print(f"  [模板匹配] 匹配失败: {e}")
            return []
    
    def _calculate_similarity(self, img1: np.ndarray, img2: np.ndarray,
                             key_feature_regions: Optional[List[Tuple[int, int, int, int]]] = None,
                             exclude_regions: Optional[List[Tuple[int, int, int, int]]] = None,
                             use_enhancement: bool = True) -> float:
        """计算两张图片的相似度（支持关键特征区域匹配和排除动态区域）
        
        Args:
            img1: 图片1（OpenCV格式）
            img2: 图片2（OpenCV格式）
            key_feature_regions: 关键特征区域列表 [(x, y, width, height), ...]
                                如果指定，只比较这些区域（大幅提升速度）
            exclude_regions: 需要排除的区域列表 [(x, y, width, height), ...]
            use_enhancement: 是否使用对比度增强（默认True，2倍增强）
            
        Returns:
            相似度（0-1）
        """
        try:
            # 转换为灰度图
            if len(img1.shape) == 3:
                gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            else:
                gray1 = img1
                
            if len(img2.shape) == 3:
                gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            else:
                gray2 = img2
            
            # 对比度增强（2倍）- 提高特征区分度
            if use_enhancement:
                try:
                    from PIL import Image, ImageEnhance
                    # 转换为PIL图像
                    pil1 = Image.fromarray(gray1)
                    pil2 = Image.fromarray(gray2)
                    # 增强对比度
                    gray1 = np.array(ImageEnhance.Contrast(pil1).enhance(2.0))
                    gray2 = np.array(ImageEnhance.Contrast(pil2).enhance(2.0))
                except Exception as e:
                    print(f"  [模板匹配] 对比度增强失败: {e}，使用原图")
            
            # 策略1: 如果指定了关键特征区域，只比较这些区域（最快）
            if key_feature_regions:
                similarities = []
                total_pixels = 0
                
                for x, y, w, h in key_feature_regions:
                    # 提取关键区域
                    region1 = gray1[y:y+h, x:x+w]
                    region2 = gray2[y:y+h, x:x+w]
                    
                    # 计算该区域的相似度
                    try:
                        from skimage.metrics import structural_similarity
                        region_sim = structural_similarity(region1, region2)
                    except ImportError:
                        # 使用MSE作为备用
                        mse = np.mean((region1.astype(float) - region2.astype(float)) ** 2)
                        region_sim = 1.0 - (mse / (255.0 ** 2))
                    
                    # 按区域大小加权
                    region_pixels = w * h
                    similarities.append(region_sim * region_pixels)
                    total_pixels += region_pixels
                
                # 加权平均
                if total_pixels > 0:
                    similarity = sum(similarities) / total_pixels
                    return max(0.0, min(1.0, similarity))
            
            # 策略2: 如果有需要排除的区域，创建遮罩
            if exclude_regions:
                # 创建全白遮罩（255表示参与比较）
                mask = np.ones_like(gray1, dtype=np.uint8) * 255
                
                # 将排除区域设为黑色（0表示不参与比较）
                for x, y, w, h in exclude_regions:
                    mask[y:y+h, x:x+w] = 0
                
                # 将排除区域设为相同的灰色值，避免影响SSIM计算
                mean_val = 128  # 使用中性灰色
                for x, y, w, h in exclude_regions:
                    gray1[y:y+h, x:x+w] = mean_val
                    gray2[y:y+h, x:x+w] = mean_val
            
            # 策略3: 全图比较（最慢）
            try:
                from skimage.metrics import structural_similarity
                similarity = structural_similarity(gray1, gray2)
                return max(0.0, min(1.0, similarity))
            except ImportError:
                # 如果没有安装scikit-image，使用备用方法：均方误差（MSE）
                mse = np.mean((gray1.astype(float) - gray2.astype(float)) ** 2)
                similarity = 1.0 - (mse / (255.0 ** 2))
                return max(0.0, min(1.0, similarity))
                
        except Exception as e:
            print(f"  [模板匹配] 计算相似度失败: {e}")
            return 0.0
    
    def match_region(self, screenshot_data: bytes, template_name: str, 
                     region: Optional[Tuple[int, int, int, int]] = None,
                     threshold: float = 0.8) -> Optional[Tuple[int, int]]:
        """在截图中查找模板的位置
        
        Args:
            screenshot_data: 截图数据（bytes）
            template_name: 模板名称
            region: 搜索区域 (x, y, width, height)，None表示全屏搜索
            threshold: 匹配阈值（0-1），默认0.8
            
        Returns:
            (x, y) 匹配位置的中心点坐标，或 None
        """
        if template_name not in self.templates:
            print(f"  [模板匹配] 模板不存在: {template_name}")
            return None
        
        if not HAS_PIL:
            return None
        
        try:
            # 将截图转换为 OpenCV 格式
            image = Image.open(BytesIO(screenshot_data))
            screenshot = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # 如果指定了区域，裁剪截图
            if region:
                x, y, w, h = region
                screenshot = screenshot[y:y+h, x:x+w]
                offset_x, offset_y = x, y
            else:
                offset_x, offset_y = 0, 0
            
            # 获取模板
            template = self.templates[template_name]
            
            # 转换为灰度图
            screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            
            # 模板匹配
            result = cv2.matchTemplate(screenshot_gray, template_gray, cv2.TM_CCOEFF_NORMED)
            
            # 找到最佳匹配位置
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val >= threshold:
                # 计算中心点坐标
                h, w = template_gray.shape
                center_x = max_loc[0] + w // 2 + offset_x
                center_y = max_loc[1] + h // 2 + offset_y
                
                print(f"  [模板匹配] ✓ 找到 {template_name} 位置: ({center_x}, {center_y}), 相似度: {max_val:.2f}")
                return (center_x, center_y)
            else:
                print(f"  [模板匹配] 未找到 {template_name} (最高相似度: {max_val:.2f})")
                return None
                
        except Exception as e:
            print(f"  [模板匹配] 区域匹配失败: {e}")
            return None


# 全局单例
_template_matcher = None


def get_template_matcher(template_dir: str = "dist/JT") -> TemplateMatcher:
    """获取模板匹配器单例
    
    Args:
        template_dir: 模板图片目录
        
    Returns:
        TemplateMatcher实例
    """
    global _template_matcher
    if _template_matcher is None:
        _template_matcher = TemplateMatcher(template_dir)
    return _template_matcher
