"""
页面状态检测模块 - 基于像素颜色检测（不依赖 OCR）
Page State Detection Module - Pixel-based detection (no OCR dependency)
"""

import asyncio
from enum import Enum
from typing import Optional, Tuple
from dataclasses import dataclass
from io import BytesIO

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from .adb_bridge import ADBBridge


class PageState(Enum):
    """页面状态枚举"""
    UNKNOWN = "unknown"           # 未知页面
    LAUNCHER = "launcher"         # Android桌面/启动器
    AD = "ad"                     # 广告页
    HOME = "home"                 # 首页/主页
    PROFILE = "profile"          # 我的页面（未登录）
    PROFILE_LOGGED = "profile_logged"  # 我的页面（已登录）
    LOGIN = "login"              # 登录页
    LOGIN_ERROR = "login_error"  # 登录错误弹窗（手机号不存在/密码错误）
    LOADING = "loading"          # 加载中
    POPUP = "popup"              # 通用弹窗
    CHECKIN = "checkin"          # 每日签到页面
    CHECKIN_POPUP = "checkin_popup"  # 签到弹窗
    WARMTIP = "warmtip"          # 温馨提示弹窗
    STARTUP_POPUP = "startup_popup"  # 启动页服务弹窗
    HOME_NOTICE = "home_notice"  # 首页公告弹窗
    HOME_ERROR_POPUP = "home_error_popup"  # 首页异常代码弹窗
    POINTS_PAGE = "points_page"  # 积分页（登录后跳转）
    SPLASH = "splash"            # 启动页
    # 新增页面状态
    TRANSFER = "transfer"        # 转账页
    TRANSFER_CONFIRM = "transfer_confirm"  # 转账确认弹窗
    WALLET = "wallet"            # 钱包页
    TRANSACTION_HISTORY = "transaction_history"  # 交易流水
    CATEGORY = "category"        # 分类页
    SEARCH = "search"            # 搜索页
    ARTICLE = "article"          # 文章页
    SETTINGS = "settings"        # 设置页
    COUPON = "coupon"            # 优惠劵页
    PROFILE_AD = "profile_ad"    # 个人页广告
    
    @property
    def chinese_name(self) -> str:
        """获取中文名称"""
        name_map = {
            "unknown": "未知页面",
            "launcher": "Android桌面",
            "ad": "广告页",
            "home": "首页",
            "profile": "个人页（未登录）",
            "profile_logged": "个人页（已登录）",
            "login": "登录页",
            "login_error": "登录错误",
            "loading": "加载中",
            "popup": "弹窗",
            "checkin": "签到页",
            "checkin_popup": "签到弹窗",
            "warmtip": "温馨提示",
            "startup_popup": "启动页服务弹窗",
            "home_notice": "首页公告",
            "home_error_popup": "首页异常代码弹窗",
            "points_page": "积分页",
            "splash": "启动页",
            "transfer": "转账页",
            "transfer_confirm": "转账确认弹窗",
            "wallet": "钱包页",
            "transaction_history": "交易流水",
            "category": "分类页",
            "search": "搜索页",
            "article": "文章页",
            "settings": "设置页",
            "coupon": "优惠劵页",
            "profile_ad": "个人页广告",
        }
        return name_map.get(self.value, self.value)


@dataclass
class PageDetectionResult:
    """页面检测结果"""
    state: PageState
    confidence: float  # 置信度 0-1
    details: str       # 详细信息
    detection_method: str = "template"  # 检测方式 (template/ocr/hybrid)
    detection_time: float = 0.0  # 检测耗时（秒）
    cached: bool = False  # 是否来自缓存


class PageDetector:
    """页面状态检测器 - 基于像素颜色"""
    
    # 540x960 分辨率下的坐标
    
    # 广告页跳过按钮位置和颜色（绿色圆形按钮）
    # 跳过按钮约在 (500, 50) 位置，绿色背景
    SKIP_BUTTON_POS = (500, 50)
    SKIP_BUTTON_COLOR = (76, 175, 80)  # 绿色
    SKIP_BUTTON_TOLERANCE = 40
    
    # 底部导航栏位置（用于判断主页/我的页面）
    BOTTOM_NAV_Y = 950  # 底部导航栏 Y 坐标
    
    # 主页特征：底部导航栏 + 顶部青色
    HOME_TOP_COLOR = (79, 193, 186)  # 青色
    
    # 登录按钮颜色（青色）
    LOGIN_BUTTON_COLOR = (79, 220, 200)
    
    def __init__(self, adb: ADBBridge):
        """初始化页面检测器
        
        Args:
            adb: ADB 桥接器实例
        """
        self.adb = adb
        self._last_screenshot = None
    
    async def _get_screenshot(self, device_id: str) -> Optional['Image.Image']:
        """获取屏幕截图
        
        Args:
            device_id: 设备 ID
            
        Returns:
            PIL Image 对象
        """
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
    
    def _get_pixel(self, image: 'Image.Image', x: int, y: int) -> Optional[Tuple[int, int, int]]:
        """获取像素颜色
        
        Args:
            image: PIL Image
            x: X 坐标
            y: Y 坐标
            
        Returns:
            (R, G, B) 颜色
        """
        try:
            if 0 <= x < image.width and 0 <= y < image.height:
                pixel = image.getpixel((x, y))
                if len(pixel) >= 3:
                    return (pixel[0], pixel[1], pixel[2])
        except Exception:
            pass
        return None
    
    def _color_match(self, color1: Tuple[int, int, int], 
                     color2: Tuple[int, int, int], 
                     tolerance: int) -> bool:
        """检查两个颜色是否匹配"""
        return (abs(color1[0] - color2[0]) <= tolerance and
                abs(color1[1] - color2[1]) <= tolerance and
                abs(color1[2] - color2[2]) <= tolerance)
    
    def _is_green(self, color: Tuple[int, int, int]) -> bool:
        """检查是否是绿色（跳过按钮的颜色）"""
        r, g, b = color
        # 绿色特征：G 通道最高，且明显高于 R 和 B
        return g > 100 and g > r + 20 and g > b + 20
    
    def _is_white(self, color: Tuple[int, int, int], threshold: int = 240) -> bool:
        """检查是否是白色"""
        return color[0] > threshold and color[1] > threshold and color[2] > threshold
    
    def _count_dark_pixels(self, image: 'Image.Image', x1: int, y1: int, 
                           x2: int, y2: int, threshold: int = 80) -> int:
        """统计区域内深色像素数量（用于检测文字）
        
        Args:
            image: PIL Image
            x1, y1: 左上角坐标
            x2, y2: 右下角坐标
            threshold: 深色阈值，RGB都小于此值视为深色
            
        Returns:
            深色像素数量
        """
        count = 0
        for y in range(y1, y2):
            for x in range(x1, x2):
                color = self._get_pixel(image, x, y)
                if color:
                    r, g, b = color
                    if r < threshold and g < threshold and b < threshold:
                        count += 1
        return count
    
    async def is_splash_page(self, device_id: str) -> bool:
        """检测是否在启动页（白色背景 + 中间有 logo）
        
        Args:
            device_id: 设备 ID
            
        Returns:
            是否是启动页
        """
        image = await self._get_screenshot(device_id)
        if not image:
            return False
        
        # 启动页特征：大部分区域是白色
        white_count = 0
        check_points = [
            (100, 100), (270, 100), (440, 100),  # 顶部
            (100, 300), (440, 300),               # 中上
            (100, 700), (270, 700), (440, 700),  # 中下
            (100, 900), (270, 900), (440, 900),  # 底部
        ]
        
        for x, y in check_points:
            color = self._get_pixel(image, x, y)
            if color and self._is_white(color):
                white_count += 1
        
        # 如果大部分点都是白色，可能是启动页
        # 但还需要排除登录页（登录页底部也是白色但有按钮）
        if white_count >= 8:
            # 检查中间是否有 logo（非白色区域）
            center_color = self._get_pixel(image, 270, 400)
            if center_color and not self._is_white(center_color, 200):
                return True
        
        return False
    
    async def is_ad_page(self, device_id: str) -> Tuple[bool, Optional[Tuple[int, int]]]:
        """检测是否在广告页面（通过检测绿色跳过按钮）
        
        Args:
            device_id: 设备 ID
            
        Returns:
            (是否是广告页, 跳过按钮位置)
        """
        image = await self._get_screenshot(device_id)
        if not image:
            return False, None
        
        # 检查右上角区域是否有绿色跳过按钮
        # 扫描右上角区域 (450-520, 30-80)
        for x in range(450, 520, 10):
            for y in range(30, 80, 10):
                color = self._get_pixel(image, x, y)
                if color and self._is_green(color):
                    # 找到绿色区域，返回中心位置
                    return True, (x, y)
        
        return False, None
    
    def _is_cyan(self, color: Tuple[int, int, int]) -> bool:
        """检查是否是青色（logo/状态栏的颜色）"""
        r, g, b = color
        # 青色特征：G 和 B 通道高，R 通道相对低
        # 例如 RGB(18, 165, 152), RGB(79, 193, 186), RGB(113, 236, 225)
        # RGB(16, 157, 147) 也是青色
        # 放宽条件：R < 150, G > 140, B > 140, 且 G 和 B 都明显高于 R
        return g > 140 and b > 140 and r < 150 and (g > r + 30 or b > r + 30)
    
    async def detect_page(self, device_id: str) -> PageDetectionResult:
        """检测当前页面状态
        
        Args:
            device_id: 设备 ID
            
        Returns:
            页面检测结果
        """
        if not HAS_PIL:
            return PageDetectionResult(
                state=PageState.UNKNOWN,
                confidence=0.0,
                details="PIL 未安装"
            )
        
        image = await self._get_screenshot(device_id)
        if not image:
            return PageDetectionResult(
                state=PageState.UNKNOWN,
                confidence=0.0,
                details="无法截取屏幕"
            )
        
        # 0. 最优先：检测纯白屏/灰屏（加载中状态）
        # 这个必须最先检测，但要排除有内容的白色页面（如登录页）
        white_count = 0
        gray_count = 0
        check_points = [
            (100, 200), (270, 200), (440, 200),
            (100, 400), (270, 400), (440, 400),
            (100, 600), (270, 600), (440, 600),
            (100, 800), (270, 800), (440, 800),
        ]
        for x, y in check_points:
            color = self._get_pixel(image, x, y)
            if color:
                if self._is_white(color, 250):
                    white_count += 1
                elif self._is_white(color, 230):  # 灰白色
                    gray_count += 1
        
        # 如果大部分是白色或灰白色，可能是加载中
        # 但需要进一步检查是否有内容（文字、按钮等）
        if white_count + gray_count >= 10:
            # 检查关键位置是否有非白色内容
            # 登录页面的青色按钮位置 (270, 720)
            login_btn = self._get_pixel(image, 270, 720)
            has_login_btn = login_btn and not self._is_white(login_btn, 200)
            
            # 检查是否有深色文字（标题、标签等）
            title_area = self._get_pixel(image, 270, 60)
            has_title = title_area and not self._is_white(title_area, 200)
            
            # 如果有按钮或标题，说明不是纯白屏，是有内容的页面
            if not has_login_btn and not has_title:
                return PageDetectionResult(
                    state=PageState.LOADING,
                    confidence=0.7,
                    details="检测到白屏/灰屏（加载中）"
                )
        
        # 1. 先检测主页公告弹窗（顶部青色 + 底部灰色遮罩 + 中间白色弹窗）
        top_cyan_color = self._get_pixel(image, 270, 70)
        bottom_color = self._get_pixel(image, 270, 950)
        center_color = self._get_pixel(image, 270, 400)
        
        if top_cyan_color and bottom_color and center_color:
            has_cyan_top = self._is_cyan(top_cyan_color)
            # 底部灰色遮罩 (约 170-190)
            is_gray_bottom = (160 < bottom_color[0] < 200 and 
                             160 < bottom_color[1] < 200 and 
                             160 < bottom_color[2] < 200)
            # 中间白色弹窗
            is_white_center = self._is_white(center_color, 250)
            
            if has_cyan_top and is_gray_bottom and is_white_center:
                return PageDetectionResult(
                    state=PageState.POPUP,
                    confidence=0.95,
                    details="检测到主页公告弹窗（青色顶部+灰色遮罩+白色弹窗）"
                )
        
        # 1. 检测主页/我的页面（底部白色导航栏 + 顶部有青色区域）
        has_bottom_nav = bottom_color and self._is_white(bottom_color)
        has_cyan_area = top_cyan_color and self._is_cyan(top_cyan_color)
        
        if has_bottom_nav and has_cyan_area:
            # 底部白色导航栏 + 顶部有青色区域 = 主页或我的页面
            
            # 先检测是否有主页公告弹窗（中间有白色弹窗区域）
            # 弹窗特征：(270, 350-450) 区域是白色，且有X关闭按钮区域
            popup_area1 = self._get_pixel(image, 270, 350)
            popup_area2 = self._get_pixel(image, 270, 450)
            popup_area3 = self._get_pixel(image, 270, 400)
            
            if popup_area1 and popup_area2 and popup_area3:
                is_popup_white1 = self._is_white(popup_area1, 250)
                is_popup_white2 = self._is_white(popup_area2, 250)
                is_popup_white3 = self._is_white(popup_area3, 250)
                
                # 如果中间区域都是白色，说明有弹窗
                if is_popup_white1 and is_popup_white2 and is_popup_white3:
                    return PageDetectionResult(
                        state=PageState.POPUP,
                        confidence=0.9,
                        details="检测到主页公告弹窗（中间白色区域）"
                    )
            
            # 检测底部导航栏"我的"图标是否高亮（判断是否在我的页面）
            my_tab_color = self._get_pixel(image, 450, 900)
            is_on_my_tab = my_tab_color and self._is_cyan(my_tab_color)
            
            if is_on_my_tab:
                # 在我的页面，进一步判断是否已登录
                # 检测ID区域 (y=155-165) - 已登录有深色ID文字，未登录是青色背景
                # 扫描ID区域的深色像素数量
                id_dark_pixels = self._count_dark_pixels(image, 100, 155, 200, 165)
                
                # 已登录时ID区域有大量深色像素（文字），未登录时几乎没有
                is_logged_in = id_dark_pixels > 50
                
                if is_logged_in:
                    return PageDetectionResult(
                        state=PageState.PROFILE_LOGGED,
                        confidence=0.95,
                        details=f"检测到我的页面（已登录，ID区域深色像素={id_dark_pixels}）"
                    )
                else:
                    return PageDetectionResult(
                        state=PageState.PROFILE,
                        confidence=0.9,
                        details=f"检测到我的页面（未登录，ID区域深色像素={id_dark_pixels}）"
                    )
            else:
                # 在主页
                return PageDetectionResult(
                    state=PageState.HOME,
                    confidence=0.85,
                    details="检测到主页（底部导航+青色区域）"
                )
        
        # 2. 检测登录错误弹窗（优先检测，避免被误判为广告页）
        # 弹窗特征：灰色遮罩 + 白色弹窗 + 青色确定按钮
        overlay_color = self._get_pixel(image, 100, 300)
        dialog_bg = self._get_pixel(image, 270, 500)
        confirm_btn = self._get_pixel(image, 428, 552)  # 确定按钮实际位置
        
        if overlay_color and dialog_bg and confirm_btn:
            or_r, or_g, or_b = overlay_color
            # 遮罩是灰色（RGB约102,102,102）
            is_overlay_gray = 80 < or_r < 130 and 80 < or_g < 130 and 80 < or_b < 130
            # 弹窗背景是白色
            is_dialog_white = self._is_white(dialog_bg, 240)
            # 确定按钮是深青色 RGB(0, 133, 119)
            cr, cg, cb = confirm_btn
            is_confirm_cyan = cr < 50 and 100 < cg < 180 and 80 < cb < 160
            
            if is_overlay_gray and is_dialog_white and is_confirm_cyan:
                return PageDetectionResult(
                    state=PageState.LOGIN_ERROR,
                    confidence=0.95,
                    details="检测到登录错误弹窗（友情提示）"
                )
        
        # 2b. 检测通用弹窗（用户协议等）
        # 特征：顶部和底部是灰色遮罩 (102,102,102)，中间有内容
        top_overlay = self._get_pixel(image, 270, 30)
        bottom_overlay = self._get_pixel(image, 270, 950)
        center_content = self._get_pixel(image, 270, 480)
        
        if top_overlay and bottom_overlay and center_content:
            # 检查顶部和底部是否都是灰色遮罩
            is_top_gray = (90 < top_overlay[0] < 120 and 
                          90 < top_overlay[1] < 120 and 
                          90 < top_overlay[2] < 120)
            is_bottom_gray = (90 < bottom_overlay[0] < 120 and 
                             90 < bottom_overlay[1] < 120 and 
                             90 < bottom_overlay[2] < 120)
            # 中间不是灰色（有内容）
            is_center_not_gray = not (90 < center_content[0] < 120 and 
                                      90 < center_content[1] < 120 and 
                                      90 < center_content[2] < 120)
            
            if is_top_gray and is_bottom_gray and is_center_not_gray:
                return PageDetectionResult(
                    state=PageState.POPUP,
                    confidence=0.9,
                    details="检测到弹窗（灰色遮罩+中间内容）"
                )
        
        # 2c. 检测主页弹窗（浅灰色背景，底部白色导航栏，中间有青色区域）
        # 特征：顶部浅灰(235,235,235)，底部白色，(270,100)位置有青色
        if top_overlay and bottom_overlay:
            is_top_light_gray = (top_overlay[0] > 220 and top_overlay[1] > 220 and 
                                 top_overlay[2] > 220 and top_overlay[0] < 250)
            is_bottom_white = self._is_white(bottom_overlay, 250)
            
            # 检查(270, 100)是否有青色（主页特征）
            cyan_check = self._get_pixel(image, 270, 100)
            has_cyan_behind = cyan_check and self._is_cyan(cyan_check)
            
            if is_top_light_gray and is_bottom_white and has_cyan_behind:
                return PageDetectionResult(
                    state=PageState.POPUP,
                    confidence=0.9,
                    details="检测到主页弹窗（浅灰背景+底部白色+青色区域）"
                )
        
        # 3. 检测广告页
        # 广告页特征：顶部有彩色内容（不是白色、不是青色、不是深灰/黑色、不是浅灰、不是中灰遮罩）
        top_color = self._get_pixel(image, 270, 30)
        if top_color:
            r, g, b = top_color
            is_white_top = self._is_white(top_color, 240)
            is_light_gray = r > 200 and g > 200 and b > 200  # 浅灰色也不是广告
            is_mid_gray = 90 < r < 130 and 90 < g < 130 and 90 < b < 130  # 中灰色（弹窗遮罩）
            is_cyan_top = self._is_cyan(top_color)
            is_dark_top = r < 80 and g < 80 and b < 80  # 深灰/黑色
            
            # 如果顶部不是白色、不是浅灰、不是中灰、不是青色、不是深色，那就是广告页
            if not is_white_top and not is_light_gray and not is_mid_gray and not is_cyan_top and not is_dark_top:
                return PageDetectionResult(
                    state=PageState.AD,
                    confidence=0.8,
                    details=f"检测到广告页（顶部彩色 RGB{top_color}）"
                )
            
            # 如果顶部是深色，检查内容区域
            if is_dark_top:
                content_colors = []
                for x, y in [(270, 300), (270, 400), (270, 500)]:
                    color = self._get_pixel(image, x, y)
                    if color:
                        content_colors.append(color)
                
                # 检查是否有彩色内容
                has_colorful_content = False
                for color in content_colors:
                    cr, cg, cb = color
                    max_diff = max(abs(cr-cg), abs(cg-cb), abs(cr-cb))
                    if max_diff > 30 and not self._is_white(color) and max(cr, cg, cb) > 50:
                        has_colorful_content = True
                        break
                
                if has_colorful_content:
                    return PageDetectionResult(
                        state=PageState.AD,
                        confidence=0.8,
                        details="检测到广告页（深色顶部+彩色内容）"
                    )
        
        # 2b. 检测绿色跳过按钮（备用检测）
        for x in range(450, 520, 10):
            for y in range(30, 80, 10):
                color = self._get_pixel(image, x, y)
                if color and self._is_green(color):
                    return PageDetectionResult(
                        state=PageState.AD,
                        confidence=0.9,
                        details=f"检测到绿色跳过按钮 ({x}, {y})"
                    )
        
        # 3. 检测启动页（白色背景 + 中上部有青色 logo）
        # 启动页特征：(270, 200) 位置有青色 logo，大部分区域白色
        logo_color = self._get_pixel(image, 270, 200)
        if logo_color and self._is_cyan(logo_color):
            # 检查周围是否大部分是白色
            white_count = 0
            splash_check_points = [
                (100, 200), (440, 200),  # logo 两侧
                (100, 400), (270, 400), (440, 400),  # 中间
                (100, 600), (270, 600), (440, 600),  # 下方
            ]
            for x, y in splash_check_points:
                color = self._get_pixel(image, x, y)
                if color and self._is_white(color):
                    white_count += 1
            
            if white_count >= 6:
                return PageDetectionResult(
                    state=PageState.LOADING,
                    confidence=0.9,
                    details="检测到启动页（白色背景+青色logo）"
                )
        
        # 5. 检测登录页面（多点检测）
        # 登录页面特征：
        # - 顶部白色背景
        # - (260, 60) 有深色标题文字
        # - (60, 280) 有黑色手机号标签
        # - (60, 540) 有灰色协议勾选框
        # - (270, 720) 有青色登录按钮
        login_features = 0
        login_details = []
        
        # 检测点1: 顶部白色
        top_color = self._get_pixel(image, 270, 30)
        if top_color and self._is_white(top_color):
            login_features += 1
            login_details.append("顶部白色")
        
        # 检测点2: 标题文字区域有深色
        title_color = self._get_pixel(image, 260, 60)
        if title_color:
            r, g, b = title_color
            if r < 80 and g < 80 and b < 80:  # 深色文字
                login_features += 1
                login_details.append("标题文字")
        
        # 检测点3: 手机号标签区域有黑色
        phone_label = self._get_pixel(image, 60, 280)
        if phone_label:
            r, g, b = phone_label
            if r < 50 and g < 50 and b < 50:  # 黑色文字
                login_features += 1
                login_details.append("手机号标签")
        
        # 检测点4: 协议勾选框区域有灰色
        checkbox_color = self._get_pixel(image, 60, 540)
        if checkbox_color:
            r, g, b = checkbox_color
            if 100 < r < 180 and 100 < g < 180 and 100 < b < 180:  # 灰色
                login_features += 1
                login_details.append("协议勾选框")
        
        # 检测点5: 登录按钮是青色（最重要的特征）
        login_btn_color = self._get_pixel(image, 270, 720)
        if login_btn_color:
            r, g, b = login_btn_color
            if g > 200 and b > 180 and r < 100:  # 青色
                login_features += 2  # 青色按钮权重更高
                login_details.append("青色登录按钮")
        
        # 检测点6: 协议链接是青色
        link_color = self._get_pixel(image, 240, 640)
        if link_color:
            r, g, b = link_color
            if g > 200 and b > 180 and r < 100:  # 青色
                login_features += 1
                login_details.append("青色协议链接")
        
        # 降低阈值：满足3个以上特征（或有青色按钮）就判定为登录页面
        if login_features >= 3:
            return PageDetectionResult(
                state=PageState.LOGIN,
                confidence=0.9,
                details=f"检测到登录页面（{login_features}分，特征：{', '.join(login_details)}）"
            )
        
        # 6. 检测加载页（底部有logo的白色页面）
        # 检查底部是否有灰色文字（溪盟商城logo区域）
        bottom_logo_text = self._get_pixel(image, 270, 895)
        if bottom_logo_text:
            r, g, b = bottom_logo_text
            # 灰色文字特征：R≈G≈B，且在 100-150 范围
            is_gray_text = (100 < r < 150 and 100 < g < 150 and 100 < b < 150 and
                           abs(r-g) < 20 and abs(g-b) < 20)
            if is_gray_text:
                return PageDetectionResult(
                    state=PageState.LOADING,
                    confidence=0.8,
                    details="检测到加载页（底部有溪盟商城logo）"
                )
        
        return PageDetectionResult(
            state=PageState.UNKNOWN,
            confidence=0.0,
            details="未能识别页面"
        )
    
    async def wait_for_page(self, device_id: str, target_state: PageState,
                           timeout: int = 15, check_interval: float = 1.0) -> bool:
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
        """检查是否已登录（通过检测ID区域）"""
        result = await self.detect_page(device_id)
        return result.state == PageState.PROFILE_LOGGED
