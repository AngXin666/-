"""
页面状态检测模块 - 基于 OCR 识别
Page State Detection Module - OCR-based detection
"""

import asyncio
from enum import Enum
from typing import Optional, Tuple, List
from dataclasses import dataclass
from io import BytesIO

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from rapidocr import RapidOCR
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

from .adb_bridge import ADBBridge
from .ocr_image_processor import enhance_for_ocr


class PageState(Enum):
    """页面状态枚举"""
    UNKNOWN = "unknown"           # 未知页面
    AD = "ad"                     # 广告页
    HOME = "home"                 # 首页/主页
    PROFILE = "profile"          # 我的页面（未登录）
    PROFILE_LOGGED = "profile_logged"  # 我的页面（已登录）
    LOGIN = "login"              # 登录页
    LOGIN_ERROR = "login_error"  # 登录错误弹窗
    LOADING = "loading"          # 加载中
    POPUP = "popup"              # 弹窗
    USER_AGREEMENT = "user_agreement"  # 用户协议弹窗
    HOME_ANNOUNCEMENT = "home_announcement"  # 主页公告弹窗


@dataclass
class PageDetectionResult:
    """页面检测结果"""
    state: PageState
    confidence: float  # 置信度 0-1
    details: str       # 详细信息
    texts: List[str] = None  # OCR 识别的文本
    detection_method: str = "ocr"  # 检测方式 (template/ocr/hybrid)
    detection_time: float = 0.0  # 检测耗时（秒）
    cached: bool = False  # 是否来自缓存


class PageDetectorOCR:
    """页面状态检测器 - 基于 OCR"""
    
    # 各页面的关键词
    HOME_KEYWORDS = ["首页", "分类", "购物车", "我的"]
    PROFILE_KEYWORDS = ["我的", "商城订单", "待付款", "待发货", "待收货"]
    PROFILE_LOGGED_KEYWORDS = ["ID:", "余额", "积分", "普通会员", "VIP"]
    LOGIN_KEYWORDS = ["登录", "手机号", "密码", "用户协议", "忘记密码"]
    LOGIN_ERROR_KEYWORDS = ["友情提示", "手机号不存在", "密码错误", "确定"]
    USER_AGREEMENT_KEYWORDS = ["用户协议", "隐私政策", "同意", "不同意"]
    ANNOUNCEMENT_KEYWORDS = ["公告", "活动", "关闭", "×"]
    AD_KEYWORDS = ["跳过", "广告", "立即下载", "了解更多"]
    LOADING_KEYWORDS = ["溪盟商城", "加载中"]
    
    def __init__(self, adb: ADBBridge):
        """初始化页面检测器
        
        Args:
            adb: ADB 桥接器实例
        """
        self.adb = adb
        self._ocr = None
        self._last_screenshot = None
        self._last_texts = None
        
        if HAS_OCR:
            # 禁用日志输出
            import logging
            logging.getLogger("rapidocr").setLevel(logging.WARNING)
            self._ocr = RapidOCR()
    
    async def _get_screenshot(self, device_id: str) -> Optional['Image.Image']:
        """获取屏幕截图"""
        if not HAS_PIL:
            return None
        
        try:
            screenshot_data = await self.adb.screencap(device_id)
            if not screenshot_data:
                return None
            
            image = Image.open(BytesIO(screenshot_data))
            self._last_screenshot = image
            return image
        except Exception:
            return None
    
    def _ocr_image(self, image: 'Image.Image') -> List[str]:
        """OCR 识别图片"""
        if not self._ocr:
            return []
        
        try:
            # 使用OCR图像预处理模块增强图像
            # 页面检测主要识别关键词文字，对比度增强可以提高文字边缘清晰度
            enhanced_image = enhance_for_ocr(image)
            
            result = self._ocr(enhanced_image)
            if result and result.txts:
                return list(result.txts)
        except Exception:
            pass
        return []
    
    def _count_keywords(self, texts: List[str], keywords: List[str]) -> int:
        """统计关键词出现次数"""
        count = 0
        text_str = " ".join(texts)
        for keyword in keywords:
            if keyword in text_str:
                count += 1
        return count
    
    def _has_keyword(self, texts: List[str], keywords: List[str]) -> bool:
        """检查是否包含任一关键词"""
        text_str = " ".join(texts)
        for keyword in keywords:
            if keyword in text_str:
                return True
        return False
    
    async def detect_page(self, device_id: str) -> PageDetectionResult:
        """检测当前页面状态
        
        Args:
            device_id: 设备 ID
            
        Returns:
            页面检测结果
        """
        if not HAS_PIL or not HAS_OCR:
            return PageDetectionResult(
                state=PageState.UNKNOWN,
                confidence=0.0,
                details="OCR 未安装"
            )
        
        image = await self._get_screenshot(device_id)
        if not image:
            return PageDetectionResult(
                state=PageState.UNKNOWN,
                confidence=0.0,
                details="无法截取屏幕"
            )
        
        # OCR 识别
        texts = self._ocr_image(image)
        self._last_texts = texts
        
        if not texts:
            # 没有识别到文字，可能是加载中或广告
            return PageDetectionResult(
                state=PageState.LOADING,
                confidence=0.5,
                details="未识别到文字，可能是加载中",
                texts=[]
            )
        
        # 1. 检测登录错误弹窗（优先级最高）
        if self._has_keyword(texts, ["友情提示"]) and self._has_keyword(texts, ["确定"]):
            return PageDetectionResult(
                state=PageState.LOGIN_ERROR,
                confidence=0.95,
                details="检测到登录错误弹窗（友情提示）",
                texts=texts
            )
        
        # 2. 检测用户协议弹窗
        if self._has_keyword(texts, ["用户协议"]) and self._has_keyword(texts, ["隐私政策"]):
            if self._has_keyword(texts, ["同意", "不同意", "确定"]):
                return PageDetectionResult(
                    state=PageState.USER_AGREEMENT,
                    confidence=0.95,
                    details="检测到用户协议弹窗",
                    texts=texts
                )
        
        # 3. 检测主页公告弹窗
        if self._has_keyword(texts, ["公告", "活动"]) and self._has_keyword(texts, ["×", "关闭", "X"]):
            return PageDetectionResult(
                state=PageState.HOME_ANNOUNCEMENT,
                confidence=0.9,
                details="检测到主页公告弹窗",
                texts=texts
            )
        
        # 4. 检测登录页面
        login_score = self._count_keywords(texts, self.LOGIN_KEYWORDS)
        if login_score >= 3:
            return PageDetectionResult(
                state=PageState.LOGIN,
                confidence=0.9,
                details=f"检测到登录页面（{login_score}个关键词匹配）",
                texts=texts
            )
        
        # 5. 检测我的页面（已登录）
        if self._has_keyword(texts, ["ID:"]) and self._has_keyword(texts, ["余额"]):
            return PageDetectionResult(
                state=PageState.PROFILE_LOGGED,
                confidence=0.95,
                details="检测到我的页面（已登录，有ID和余额）",
                texts=texts
            )
        
        # 6. 检测我的页面（未登录）
        if self._has_keyword(texts, ["请登录", "登录/注册"]):
            if self._has_keyword(texts, self.PROFILE_KEYWORDS):
                return PageDetectionResult(
                    state=PageState.PROFILE,
                    confidence=0.9,
                    details="检测到我的页面（未登录）",
                    texts=texts
                )
        
        # 7. 检测首页
        home_score = self._count_keywords(texts, self.HOME_KEYWORDS)
        if home_score >= 3:
            # 检查是否有弹窗覆盖
            if self._has_keyword(texts, ["×", "关闭"]):
                return PageDetectionResult(
                    state=PageState.POPUP,
                    confidence=0.8,
                    details="检测到首页有弹窗",
                    texts=texts
                )
            return PageDetectionResult(
                state=PageState.HOME,
                confidence=0.85,
                details=f"检测到首页（{home_score}个关键词匹配）",
                texts=texts
            )
        
        # 8. 检测广告页
        if self._has_keyword(texts, ["跳过"]):
            return PageDetectionResult(
                state=PageState.AD,
                confidence=0.9,
                details="检测到广告页（有跳过按钮）",
                texts=texts
            )
        
        # 9. 检测加载页
        if self._has_keyword(texts, ["溪盟商城"]) and len(texts) <= 3:
            return PageDetectionResult(
                state=PageState.LOADING,
                confidence=0.8,
                details="检测到加载页（溪盟商城logo）",
                texts=texts
            )
        
        # 10. 通用弹窗检测
        if self._has_keyword(texts, ["确定", "取消", "关闭", "×"]):
            return PageDetectionResult(
                state=PageState.POPUP,
                confidence=0.7,
                details="检测到可能的弹窗",
                texts=texts
            )
        
        return PageDetectionResult(
            state=PageState.UNKNOWN,
            confidence=0.0,
            details=f"未能识别页面，识别到{len(texts)}个文本",
            texts=texts
        )
    
    async def wait_for_page(self, device_id: str, target_state: PageState,
                           timeout: int = 30, check_interval: float = 1.0) -> bool:
        """等待指定页面出现"""
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            result = await self.detect_page(device_id)
            if result.state == target_state:
                return True
            await asyncio.sleep(check_interval)
        
        return False
    
    async def is_on_home(self, device_id: str) -> bool:
        """检查是否在主页"""
        result = await self.detect_page(device_id)
        return result.state == PageState.HOME
    
    async def is_on_login(self, device_id: str) -> bool:
        """检查是否在登录页"""
        result = await self.detect_page(device_id)
        return result.state == PageState.LOGIN
    
    async def is_logged_in(self, device_id: str) -> bool:
        """检查是否已登录"""
        result = await self.detect_page(device_id)
        return result.state == PageState.PROFILE_LOGGED
    
    async def has_popup(self, device_id: str) -> Tuple[bool, Optional[PageState]]:
        """检查是否有弹窗
        
        Returns:
            (是否有弹窗, 弹窗类型)
        """
        result = await self.detect_page(device_id)
        popup_states = [
            PageState.POPUP,
            PageState.USER_AGREEMENT,
            PageState.HOME_ANNOUNCEMENT,
            PageState.LOGIN_ERROR
        ]
        if result.state in popup_states:
            return True, result.state
        return False, None
