"""
增强广告检测器单元测试
"""

import pytest
import numpy as np
from src.ad_detection.enhanced_ad_detector import EnhancedAdDetector, AdDetectionResult


class TestEnhancedAdDetector:
    """测试增强广告检测器"""
    
    def test_detector_initialization(self):
        """测试检测器初始化"""
        detector = EnhancedAdDetector()
        assert detector is not None
        assert detector.template_dir == "templates/ads"
        assert detector.TEMPLATE_THRESHOLD == 0.7
        assert detector.OCR_THRESHOLD == 0.6
        assert detector.LAYOUT_THRESHOLD == 0.5
        assert detector.COMBINED_THRESHOLD == 0.6
    
    def test_weights_sum_to_one(self):
        """测试权重总和为1"""
        detector = EnhancedAdDetector()
        total_weight = (
            detector.TEMPLATE_WEIGHT +
            detector.OCR_WEIGHT +
            detector.LAYOUT_WEIGHT
        )
        assert abs(total_weight - 1.0) < 0.01, "权重总和应该为1"
    
    def test_ssim_calculation_same_image(self):
        """测试SSIM计算 - 相同图片应该返回1.0"""
        detector = EnhancedAdDetector()
        
        # 创建一个简单的测试图片
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # 相同图片的SSIM应该接近1.0
        similarity = detector._calculate_ssim(img, img)
        assert similarity > 0.99, f"相同图片的SSIM应该接近1.0，实际: {similarity}"
    
    def test_ssim_calculation_different_images(self):
        """测试SSIM计算 - 不同图片应该返回较低值"""
        detector = EnhancedAdDetector()
        
        # 创建两个不同的测试图片
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)  # 全黑
        img2 = np.ones((100, 100, 3), dtype=np.uint8) * 255  # 全白
        
        # 完全不同图片的SSIM应该很低
        similarity = detector._calculate_ssim(img1, img2)
        assert similarity < 0.5, f"完全不同图片的SSIM应该很低，实际: {similarity}"
    
    @pytest.mark.anyio
    async def test_detect_by_template_no_templates(self):
        """测试模板匹配 - 无模板时应返回0"""
        detector = EnhancedAdDetector()
        detector._templates = []  # 清空模板
        
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        score, details = await detector._detect_by_template(img)
        
        assert score == 0.0
        assert details['method'] == 'template'
        assert '无可用模板' in details['message']
    
    @pytest.mark.anyio
    async def test_detect_by_ocr_no_text(self):
        """测试OCR检测 - 无文字时应返回0"""
        detector = EnhancedAdDetector()
        
        # 创建一个纯色图片（没有文字）
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        score, details = await detector._detect_by_ocr(img)
        
        # 无文字时得分应该很低或为0
        assert score <= 0.1
        assert details['method'] == 'ocr'
    
    @pytest.mark.anyio
    async def test_detect_by_layout(self):
        """测试布局检测 - 应该返回有效结果"""
        detector = EnhancedAdDetector()
        
        # 创建一个测试图片
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        score, details = await detector._detect_by_layout(img)
        
        # 应该返回0-1之间的分数
        assert 0.0 <= score <= 1.0
        assert details['method'] == 'layout'
        assert 'fullscreen_score' in details
        assert 'edge_score' in details
        assert 'brightness_score' in details
    
    def test_keyword_lists(self):
        """测试关键词列表定义"""
        detector = EnhancedAdDetector()
        
        # 验证关键词列表不为空
        assert len(detector.SKIP_KEYWORDS) > 0
        assert len(detector.CLOSE_KEYWORDS) > 0
        assert len(detector.COUNTDOWN_KEYWORDS) > 0
        assert len(detector.AD_KEYWORDS) > 0
        
        # 验证包含预期的关键词
        assert "跳过" in detector.SKIP_KEYWORDS
        assert "关闭" in detector.CLOSE_KEYWORDS
        assert "秒后" in detector.COUNTDOWN_KEYWORDS
        assert "广告" in detector.AD_KEYWORDS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
