"""
混合页面检测模块 - 像素快速检测 + OCR 精确识别
Hybrid Page Detection Module - Pixel fast detection + OCR precise recognition
"""

import asyncio
from typing import Optional, Tuple, List
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
from .page_detector import PageDetector, PageState, PageDetectionResult
from .ocr_thread_pool import get_ocr_pool


class PageDetectorHybrid:
    """混合页面检测器 - 像素 + OCR"""
    
    # 弹窗按钮坐标 (540x960)
    POPUP_BUTTONS = {
        'user_agreement': (270, 600),      # 服务协议弹窗"同意并接受"（实际测量坐标）
        'user_agreement_alt': (270, 608),  # 服务协议弹窗备用坐标
        'home_announcement': (270, 690),   # 主页公告弹窗（底部中央按钮）
        'login_error': (436, 557),         # 登录错误确定按钮
        'generic': (270, 600),             # 通用弹窗
    }
    
    # 签到弹窗关闭按钮坐标（MuMu模拟器 540x960）
    CHECKIN_POPUP_CLOSE = [
        (270, 812),  # 中心位置
        (278, 811),  # 右偏
        (274, 811),  # 中右
    ]
    
    def __init__(self, adb: ADBBridge, log_callback=None):
        """初始化混合检测器
        
        Args:
            adb: ADB桥接对象
            log_callback: 可选的日志回调函数
        """
        self.adb = adb
        self.pixel_detector = PageDetector(adb)
        self._ocr_pool = get_ocr_pool() if HAS_OCR else None
        self._last_screenshot = None
        self._log_callback = log_callback
        
        # 初始化检测缓存（TTL=2秒 - 增加缓存时间）
        from .performance.detection_cache import DetectionCache
        self._detection_cache = DetectionCache(ttl=2.0)
        
        # 初始化模板匹配器（使用单例，避免重复加载）
        from .template_matcher import get_template_matcher
        import os
        import sys
        
        # 获取模板目录路径（支持打包后的 EXE）
        if getattr(sys, 'frozen', False):
            # 打包后的 EXE，模板文件被解压到临时目录
            # PyInstaller 会把文件解压到 sys._MEIPASS 目录
            base_dir = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
            self._log(f"[模板匹配] 运行模式: 打包后的 EXE")
            self._log(f"[模板匹配] sys._MEIPASS: {getattr(sys, '_MEIPASS', '未设置')}")
            self._log(f"[模板匹配] sys.executable: {sys.executable}")
            self._log(f"[模板匹配] base_dir: {base_dir}")
            template_dir = os.path.join(base_dir, 'dist', 'JT')
        else:
            # 开发环境，使用项目根目录下的 dist/JT
            base_dir = os.path.dirname(os.path.dirname(__file__))
            self._log(f"[模板匹配] 运行模式: 开发环境")
            self._log(f"[模板匹配] base_dir: {base_dir}")
            template_dir = os.path.join(base_dir, 'dist', 'JT')
        
        # 调试信息：显示路径
        self._log(f"[模板匹配] 检查模板目录: {template_dir}")
        self._log(f"[模板匹配] 目录是否存在: {os.path.exists(template_dir)}")
        
        if os.path.exists(template_dir):
            try:
                # 使用单例获取模板匹配器（只在第一次调用时加载模板）
                self._template_matcher = get_template_matcher(template_dir)
                if self._template_matcher and self._template_matcher.templates:
                    # 只在第一次初始化时输出日志
                    if not hasattr(get_template_matcher, '_logged'):
                        self._log(f"[模板匹配] ✓ 模板匹配器初始化成功，共 {len(self._template_matcher.templates)} 个模板")
                        get_template_matcher._logged = True
                else:
                    self._log(f"[模板匹配] ⚠️ 模板匹配器初始化成功，但未加载任何模板")
                    self._template_matcher = None
            except Exception as e:
                self._log(f"[模板匹配] ✗ 模板匹配器初始化失败: {e}")
                import traceback
                traceback.print_exc()
                self._template_matcher = None
        else:
            self._log(f"[模板匹配] ✗ 模板目录不存在，模板匹配功能不可用")
            self._log(f"[模板匹配] 提示：请确保 dist/JT 目录存在并包含模板图片")
            self._template_matcher = None
        
        # 模板到页面状态的映射（排除包含个人信息的页面和内容变化大的页面）
        self._template_to_state = {
            '首页.png': PageState.HOME,
            '登陆.png': PageState.LOGIN,
            '签到.png': PageState.CHECKIN,
            '签到弹窗.png': PageState.POPUP,
            '首页公告.png': PageState.POPUP,
            '温馨提示.png': PageState.POPUP,
            '用户名或密码错误.png': PageState.POPUP,
            '手机号码不存在.png': PageState.POPUP,
            '启动页服务弹窗.png': PageState.POPUP,  # 用户协议弹窗
            '广告.png': PageState.AD,
            '加载卡死白屏.png': PageState.LOADING,
            '模拟器桌面.png': PageState.LAUNCHER,
            '商品列表.png': PageState.UNKNOWN,
            '已登陆个人页.png': PageState.PROFILE_LOGGED,  # 已登录的个人页面
            '未登陆个人页.png': PageState.PROFILE,  # 未登录的个人页面
            '积分页.png': PageState.POINTS_PAGE,  # 积分页（登录后跳转的页面，需要返回）
            '我的钱包.png': PageState.UNKNOWN,  # 钱包页面（转账流程）
            '余额互转.png': PageState.UNKNOWN,  # 转账页面（转账流程）
            '交易流水.png': PageState.UNKNOWN,  # 交易流水页面（转账流程）
            # 排除的页面：
            # 1. 包含个人信息的页面（每次数据不同）：
            #    注意：'我的钱包.png', '交易流水.png', '余额互转.png' 虽然包含个人信息，
            #    但在转账流程中需要识别，所以添加到映射中（映射为 UNKNOWN 状态）
            # 2. 内容变化大的页面（容易误判，改用 OCR 识别）：
            #    '文章详情（1）.png', '文章详情（2）.png', '文章详情（3）.png', 
            #    '文章详情（4）.png', '文章详情（5）.png', '文章详情（6）.png'
            #    这些页面只有顶部标题栏相同，文章内容每次都不同，
            #    使用模板匹配会导致首页被误判为文章详情页
        }
    
    def _log(self, msg: str):
        """输出日志
        
        Args:
            msg: 日志消息
        """
        if self._log_callback:
            self._log_callback(msg)
        else:
            print(msg)
    
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
    
    def get_last_screenshot_texts(self) -> list:
        """获取最后一次截图的OCR识别文本
        
        Returns:
            识别到的文本列表，如果没有截图或识别失败则返回空列表
        """
        if not self.pixel_detector._last_screenshot:
            return []
        
        try:
            return self._ocr_image(self.pixel_detector._last_screenshot)
        except Exception as e:
            print(f"    [OCR] ⚠️ 获取OCR文本失败: {e}")
            return []
    
    def _ocr_image(self, image: 'Image.Image', timeout: float = 5.0, preprocess: bool = True) -> list:
        """OCR 识别图片（使用线程池，带超时保护和图像预处理）
        
        Args:
            image: PIL图片对象
            timeout: 超时时间（秒），默认5秒
            preprocess: 是否进行图像预处理（默认True）
            
        Returns:
            识别到的文本列表
        """
        if not self._ocr_pool:
            return []
        
        try:
            # 图像预处理（提升识别准确率）
            # 使用统一的OCR图像预处理模块：灰度图 + 对比度增强2倍
            if preprocess:
                try:
                    from .ocr_image_processor import enhance_for_ocr
                    image = enhance_for_ocr(image)
                except Exception as e:
                    print(f"    [OCR] ⚠️ 图像预处理失败: {e}，使用原图")
                    pass  # 预处理失败，使用原图
            
            # 直接调用线程池的同步方法
            result = self._ocr_pool._ocr_sync(image, use_cache=True)
            
            if result and result.texts:
                return result.texts
            return []
            
        except Exception as e:
            print(f"    [OCR] ⚠️ OCR识别失败: {e}")
            return []
    
    def _should_use_ocr(self, pixel_result: PageDetectionResult, use_ocr: bool = False) -> bool:
        """判断是否需要使用 OCR
        
        根据检测结果和场景决定是否需要 OCR：
        1. 页面类型检测优先用模板匹配（已在 detect_page 中处理）
        2. 按钮位置识别必须用 OCR
        3. 模板匹配置信度低时降级 OCR
        
        优化策略：减少不必要的OCR调用
        - 只在真正需要时才使用OCR
        - 提高模板匹配和像素检测的置信度阈值
        
        Args:
            pixel_result: 像素检测结果
            use_ocr: 是否明确要求使用 OCR
            
        Returns:
            是否需要使用 OCR
        """
        # 明确要求使用 OCR
        if use_ocr:
            return True
        
        # 检测到弹窗（需要确认弹窗类型和按钮位置）
        if pixel_result.state == PageState.POPUP:
            return True
        
        # 检测到 LOADING（可能是活动页面，需要 OCR 确认）
        if pixel_result.state == PageState.LOADING:
            return True
        
        # 检测到 UNKNOWN（不确定的页面，需要 OCR 识别）
        if pixel_result.state == PageState.UNKNOWN:
            return True
        
        # 优化：提高置信度阈值，只有在置信度很低时才使用OCR
        # 从0.7提高到0.6，减少OCR调用
        if pixel_result.confidence < 0.6:
            return True
        
        # 其他情况不需要 OCR
        return False
    
    async def _detect_by_template(self, device_id: str) -> Optional[PageDetectionResult]:
        """使用模板匹配检测页面（快速但不够灵活）
        
        Args:
            device_id: 设备ID
            
        Returns:
            检测结果，如果没有匹配或置信度低返回 None（会降级到 OCR）
        """
        if not self._template_matcher:
            # 只在第一次输出警告，避免日志刷屏
            if not hasattr(self, '_template_warning_shown'):
                print(f"    [模板匹配] ⚠️ 模板匹配器未初始化，将使用 OCR 识别")
                self._template_warning_shown = True
            return None
        
        try:
            # 获取截图
            screenshot_data = await self.adb.screencap(device_id)
            if not screenshot_data:
                print(f"    [模板匹配] ⚠️ 获取截图失败")
                return None
            
            # 匹配模板
            best_match = self._template_matcher.match_screenshot(screenshot_data)
            
            # 优化：降低相似度阈值到80%，提高模板匹配命中率，减少OCR调用
            # 原阈值85%，现在80%
            if best_match and best_match['similarity'] >= 0.80:
                template_name = best_match['template_name']
                similarity = best_match['similarity']
                
                # 查找对应的页面状态
                if template_name in self._template_to_state:
                    state = self._template_to_state[template_name]
                    print(f"    [模板匹配] ✓ 匹配成功: {template_name} (相似度: {similarity:.2%}) -> {state.value}")
                    return PageDetectionResult(
                        state=state,
                        confidence=similarity,
                        details=f"模板匹配: {template_name} (相似度: {similarity:.2%})"
                    )
                else:
                    print(f"    [模板匹配] ⚠️ 模板 {template_name} 未在映射中 (相似度: {similarity:.2%})")
            else:
                if best_match:
                    # 相似度低于阈值，返回 None 让系统降级到 OCR
                    # 只在相似度接近阈值时输出（避免日志刷屏）
                    if best_match['similarity'] >= 0.75:
                        print(f"    [模板匹配] 相似度过低: {best_match['template_name']} ({best_match['similarity']:.2%} < 85%)，降级到 OCR")
            
            return None
            
        except Exception as e:
            print(f"    [模板匹配] ✗ 匹配失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def detect_page(self, device_id: str, use_ocr: bool = False, use_template: bool = True) -> PageDetectionResult:
        """检测当前页面状态（优化版：缓存 -> 模板匹配 -> 像素检测 -> OCR）
        
        Args:
            device_id: 设备 ID
            use_ocr: 是否使用 OCR（默认只用像素检测）
            use_template: 是否使用模板匹配（默认True，优先级最高）
            
        Returns:
            页面检测结果
        """
        import time
        start_time = time.time()
        
        # 策略0: 检查缓存（最快）
        cached_result = self._detection_cache.get(device_id)
        if cached_result is not None:
            # 缓存命中，直接返回
            cached_result.cached = True
            cached_result.detection_time = time.time() - start_time
            print(f"    [检测] ✓ 缓存命中: {cached_result.state.value} (耗时: {cached_result.detection_time*1000:.1f}ms)")
            return cached_result
        
        # 策略1: 模板匹配（最快，准确率高，但不够灵活）
        if use_template and self._template_matcher:
            template_result = await self._detect_by_template(device_id)
            # 优化：降低置信度阈值到0.80，提高模板匹配命中率
            if template_result and template_result.confidence >= 0.80:
                # 模板匹配成功且置信度高，直接返回
                template_result.detection_method = "template"
                template_result.detection_time = time.time() - start_time
                template_result.cached = False
                print(f"    [检测] 使用模板匹配结果: {template_result.state.value} (耗时: {template_result.detection_time*1000:.1f}ms)")
                
                # 更新缓存
                self._detection_cache.set(device_id, template_result)
                return template_result
        
        # 策略2: 像素检测（快速）
        result = await self.pixel_detector.detect_page(device_id)
        
        # 使用智能判断是否需要 OCR
        need_ocr = self._should_use_ocr(result, use_ocr)
        
        # 策略3: OCR 识别（慢但准确）
        if need_ocr and self._ocr_pool and self.pixel_detector._last_screenshot:
            texts = self._ocr_image(self.pixel_detector._last_screenshot)
            if texts:
                # 用 OCR 结果增强检测
                result = self._enhance_with_ocr(result, texts)
                result.detection_method = "hybrid"  # 像素+OCR混合
            else:
                result.detection_method = "pixel"
        else:
            result.detection_method = "pixel"
        
        # 记录检测耗时
        result.detection_time = time.time() - start_time
        result.cached = False
        
        # 更新缓存
        self._detection_cache.set(device_id, result)
        
        print(f"    [检测] 完成: {result.state.value} (方式: {result.detection_method}, 耗时: {result.detection_time*1000:.1f}ms)")
        
        return result
    
    async def detect_page_with_priority(self, device_id: str, 
                                       expected_templates: List[str],
                                       use_cache: bool = True) -> PageDetectionResult:
        """使用优先级模板列表检测页面（快速模式）
        
        当我们知道接下来可能出现哪些页面时，只匹配这些模板，大幅提升速度。
        
        Args:
            device_id: 设备 ID
            expected_templates: 期望的模板名称列表（按优先级排序）
            use_cache: 是否使用缓存（默认True）
            
        Returns:
            页面检测结果
        """
        import time
        start_time = time.time()
        
        # 策略0: 检查缓存（最快）
        if use_cache:
            cached_result = self._detection_cache.get(device_id)
            if cached_result is not None:
                cached_result.cached = True
                cached_result.detection_time = time.time() - start_time
                self._log(f"    [快速检测] ✓ 缓存命中: {cached_result.state.value} (耗时: {cached_result.detection_time*1000:.1f}ms)")
                return cached_result
        
        # 策略1: 优先级模板匹配（只匹配指定的模板）
        if not self._template_matcher:
            # 降级到普通检测
            return await self.detect_page(device_id, use_ocr=False, use_template=False)
        
        try:
            # 获取截图
            screenshot_data = await self.adb.screencap(device_id)
            if not screenshot_data:
                self._log(f"    [快速检测] ⚠️ 获取截图失败")
                return await self.detect_page(device_id, use_ocr=False, use_template=False)
            
            # 按优先级匹配模板
            best_match = self._template_matcher.match_with_priority(
                screenshot_data, 
                expected_templates,
                threshold=0.80  # 优化：降低阈值到80%
            )
            
            if best_match:
                template_name = best_match['template_name']
                similarity = best_match['similarity']
                
                # 查找对应的页面状态
                if template_name in self._template_to_state:
                    state = self._template_to_state[template_name]
                    result = PageDetectionResult(
                        state=state,
                        confidence=similarity,
                        details=f"快速匹配: {template_name} (相似度: {similarity:.2%})",
                        detection_method="template_priority",
                        detection_time=time.time() - start_time,
                        cached=False
                    )
                    
                    self._log(f"    [快速检测] ✓ 匹配成功: {template_name} (相似度: {similarity:.2%}) -> {state.value} (耗时: {result.detection_time*1000:.1f}ms)")
                    
                    # 更新缓存
                    if use_cache:
                        self._detection_cache.set(device_id, result)
                    
                    return result
            
            # 优先级匹配失败，降级到普通检测
            self._log(f"    [快速检测] 优先级模板未匹配，降级到普通检测")
            return await self.detect_page(device_id, use_ocr=False, use_template=True)
            
        except Exception as e:
            self._log(f"    [快速检测] ✗ 失败: {e}")
            # 降级到普通检测
            return await self.detect_page(device_id, use_ocr=False, use_template=False)
    
    def _enhance_with_ocr(self, pixel_result: PageDetectionResult, 
                          texts: list) -> PageDetectionResult:
        """用 OCR 结果增强检测"""
        text_str = " ".join(texts)
        
        # 0. 检测Android桌面/启动器（最高优先级）
        # Android桌面特征：有"设置"、"应用商店"、"文件管理"等系统应用图标
        launcher_keywords = ["设置", "应用商店", "文件管理", "浏览器", "相机", "图库", "时钟", "日历"]
        launcher_count = sum(1 for kw in launcher_keywords if kw in text_str)
        
        # 0. 检测Android桌面/启动器（最高优先级）
        # Android桌面特征：有"设置"、"应用商店"、"文件管理"等系统应用图标
        launcher_keywords = ["设置", "应用商店", "文件管理", "浏览器", "相机", "图库", "时钟", "日历", "系统应用"]
        launcher_count = sum(1 for kw in launcher_keywords if kw in text_str)
        
        # 如果识别到2个或以上的系统应用名称，或者有明显的桌面特征，判定为Android桌面
        # 明显的桌面特征：同时有多个应用名称（不是底部导航栏）
        has_multiple_apps = sum(1 for app in ["溪盟商城", "溪盟山泉", "千国汇盟", "空灵诗篇", "风影游侠", "热血江湖", "鹅鸭杀", "灵妖劫"] if app in text_str) >= 3
        
        if launcher_count >= 2 or has_multiple_apps:
            return PageDetectionResult(
                state=PageState.LAUNCHER,
                confidence=0.95,
                details=f"Android桌面（OCR确认，识别到{launcher_count}个系统应用，{has_multiple_apps}个应用图标）"
            )
        
        # 1. 检测登录错误弹窗（最高优先级）
        if "友情提示" in text_str:
            if "手机号不存在" in text_str:
                return PageDetectionResult(
                    state=PageState.LOGIN_ERROR,
                    confidence=0.98,
                    details="登录错误：手机号不存在"
                )
            if "密码错误" in text_str:
                return PageDetectionResult(
                    state=PageState.LOGIN_ERROR,
                    confidence=0.98,
                    details="登录错误：密码错误"
                )
            return PageDetectionResult(
                state=PageState.LOGIN_ERROR,
                confidence=0.95,
                details="登录错误弹窗（OCR确认）"
            )
        
        # 1.5. 检测签到次数用完弹窗
        if "温馨提示" in text_str and "没有签到次数" in text_str:
            return PageDetectionResult(
                state=PageState.POPUP,
                confidence=0.98,
                details="签到次数已用完弹窗（OCR确认）"
            )
        
        # 1.6. 检测签到奖励弹窗（新增）
        # 签到奖励弹窗特征：根据用户提供的截图，弹窗上的文字是："恭喜你签到成功"、"¥1.14"、"知道了"
        # 关键特征：有"恭喜" + 有"成功" + 有金额符号或"知道了"按钮
        # 不使用"签到"关键词，因为签到页面本身也有这个字
        has_congrats = "恭喜" in text_str
        has_success = "成功" in text_str
        has_know_button = "知道了" in text_str or "知道" in text_str
        has_amount = "¥" in text_str or "￥" in text_str
        
        # 必须同时满足：恭喜 + 成功 + (知道了 或 金额符号)
        if has_congrats and has_success and (has_know_button or has_amount):
            return PageDetectionResult(
                state=PageState.POPUP,
                confidence=0.95,
                details="签到奖励弹窗（OCR确认）"
            )
        
        # 2. 检测用户协议弹窗
        # 排除登录页面（登录页面底部也有用户协议文字）
        if any(kw in text_str for kw in ["用户协议", "隐私政策", "服务协议", "隐私协议"]):
            # 如果同时有"登录"关键词，且没有"同意并接受"，则不是弹窗
            if "登录" in text_str and "同意并接受" not in text_str:
                pass  # 不是弹窗，继续检测
            else:
                return PageDetectionResult(
                    state=PageState.POPUP,
                    confidence=0.95,
                    details="用户协议弹窗（OCR确认）"
                )
        
        # 3. 检测主页公告弹窗（改进检测逻辑）
        # 主页弹窗特征：有主页元素（首页/分类/购物车/我的）+ 有弹窗关键词（公告/活动等）
        home_keywords = ["首页", "分类", "购物车", "我的"]
        popup_keywords = ["公告", "活动", "恭喜", "领取", "提现通道"]
        
        # 底部导航栏关键词（提前定义，后面会用到）
        nav_keywords = ["首页", "分类", "购物车", "我的"]
        nav_count = sum(1 for kw in nav_keywords if kw in text_str)
        
        has_home = any(kw in text_str for kw in home_keywords)
        has_popup = any(kw in text_str for kw in popup_keywords)
        
        if has_home and has_popup:
            return PageDetectionResult(
                state=PageState.POPUP,
                confidence=0.95,
                details="主页公告弹窗（OCR确认）"
            )
        
        # 4. 通用弹窗检测（有关闭按钮）
        # 注意：单独的"X"字符可能是页面内容，需要更严格的判断
        # 只有在没有底部导航栏的情况下，且有明确的关闭按钮标识，才判定为弹窗
        has_close_button = "×" in text_str or "关闭" in text_str
        # 排除页面内容中的"X"（通常"X"单独出现且周围有其他内容时不是关闭按钮）
        if has_close_button and not has_home and nav_count < 2:
            return PageDetectionResult(
                state=PageState.POPUP,
                confidence=0.90,
                details="通用弹窗（OCR确认）"
            )
        
        # 4. 检测已登录状态
        if "ID:" in text_str and "余额" in text_str:
            return PageDetectionResult(
                state=PageState.PROFILE_LOGGED,
                confidence=0.98,
                details="我的页面（已登录，OCR确认）"
            )
        
        # 5. 检测未登录状态
        if "请登录" in text_str or "登录/注册" in text_str:
            return PageDetectionResult(
                state=PageState.PROFILE,
                confidence=0.95,
                details="我的页面（未登录，OCR确认）"
            )
        
        # 6. 检测登录页面（优先级提高，在检测首页之前）
        login_keywords = ["手机号", "密码", "登录", "用户协议"]
        login_count = sum(1 for kw in login_keywords if kw in text_str)
        # 如果有登录关键词，且没有"友情提示"（排除错误弹窗），则判定为登录页
        if login_count >= 3 and "友情提示" not in text_str:
            return PageDetectionResult(
                state=PageState.LOGIN,
                confidence=0.95,
                details=f"登录页面（OCR确认，{login_count}个关键词）"
            )
        
        # 7. 检测首页（更精确的判断）
        # 首页特征：必须有"每日签到" + 底部导航栏
        has_daily_checkin = "每日签到" in text_str
        has_ximeng = "溪盟山泉" in text_str or "溪盟" in text_str
        
        # nav_count 已在前面定义
        
        # 首页判断：必须有"每日签到"，且有底部导航栏（至少3个）
        # 注意："溪盟"可能出现在商品名称中，不能作为首页的唯一判断依据
        if has_daily_checkin and nav_count >= 3:
            return PageDetectionResult(
                state=PageState.HOME,
                confidence=0.95,
                details=f"首页（OCR确认，每日签到={has_daily_checkin}，导航栏={nav_count}）"
            )
        
        # 如果只有导航栏但没有"每日签到"，可能是分类页或其他页面
        if nav_count >= 3 and not has_daily_checkin:
            # 进一步判断是否是分类页
            # 分类页特征：有"分类"关键词 + 有导航栏
            if "分类" in text_str:
                return PageDetectionResult(
                    state=PageState.UNKNOWN,
                    confidence=0.85,
                    details=f"分类页（有导航栏，需点击首页标签）"
                )
            else:
                return PageDetectionResult(
                    state=PageState.UNKNOWN,
                    confidence=0.7,
                    details=f"其他页面（有导航栏但无每日签到）"
                )
        
        # 8. 检测商品列表页（异常页面）
        # 商品列表页特征：有"商品列表"、"搜索商品"、排序选项等
        product_list_keywords = ["商品列表", "搜索商品", "累计成交"]
        if any(kw in text_str for kw in product_list_keywords):
            return PageDetectionResult(
                state=PageState.UNKNOWN,
                confidence=0.90,
                details="商品列表页（异常页面，需要返回）"
            )
        
        # 9. 检测产品套餐页（异常页面）
        # 产品套餐页特征：有"选项1"、"选项2"、"立即购买"等
        package_keywords = ["选项1", "选项2", "套餐"]
        package_count = sum(1 for kw in package_keywords if kw in text_str)
        has_buy_button = "立即购买" in text_str
        
        if package_count >= 2 and has_buy_button:
            return PageDetectionResult(
                state=PageState.UNKNOWN,
                confidence=0.90,
                details="产品套餐页（异常页面，需要返回）"
            )
        
        # 10. 检测每日签到/抽奖页面（同一个页面）
        # 签到页面特征：有"每天签到"、"立即签到"、"签到任务"、"总次数"等
        # 注意：签到完成后可能还在签到页面，需要能识别出来
        checkin_keywords = ["每天签到", "立即签到", "签到任务", "总次数", "签到成功", "已签到"]
        if any(kw in text_str for kw in checkin_keywords):
            return PageDetectionResult(
                state=PageState.CHECKIN,
                confidence=0.95,
                details="每日签到/抽奖页面（OCR确认）"
            )
        
        # 11. 检测商品详情页（异常页面）
        # 商品详情页特征：有价格、加入购物车、立即购买等
        product_keywords = ["加入购物车", "立即购买", "收藏", "分享"]
        product_count = sum(1 for kw in product_keywords if kw in text_str)
        has_price = "￥" in text_str or "¥" in text_str
        
        if product_count >= 2 and has_price:
            return PageDetectionResult(
                state=PageState.UNKNOWN,
                confidence=0.90,
                details="商品详情页（异常页面，需要返回）"
            )
        
        # 12. 检测文章列表/内容页面（异常页面）
        article_keywords = ["文章列表", "全部"]
        if any(kw in text_str for kw in article_keywords):
            return PageDetectionResult(
                state=PageState.UNKNOWN,
                confidence=0.85,
                details="文章列表页面（异常页面，需要返回）"
            )
        
        # 13. 检测活动页面（登录后可能跳转到的页面）
        activity_keywords = ["积分商城", "抽奖", "领取", "暂无相关记录", "积分"]
        if any(kw in text_str for kw in activity_keywords):
            # 但不是我的页面（我的页面也有积分）
            if "余额" not in text_str and "商城订单" not in text_str:
                return PageDetectionResult(
                    state=PageState.UNKNOWN,
                    confidence=0.8,
                    details="活动页面（需要按返回键）"
                )
        
        # 14. 检测广告页
        # 广告页特征：有"跳过"文字 + "广告"标识
        has_skip = "跳过" in text_str or "skip" in text_str.lower()
        has_ad_label = "广告" in text_str
        
        # 更严格的广告页判断：必须同时有"跳过"和"广告"标识
        if has_skip and has_ad_label:
            return PageDetectionResult(
                state=PageState.AD,
                confidence=0.95,
                details="广告页（OCR确认：跳过+广告标识）"
            )
        
        # 如果只有"跳过"但没有"广告"标识，可能是其他页面
        # 降低置信度，避免误判
        if has_skip and not has_ad_label:
            # 检查是否有底部导航栏（如果有，可能不是广告）
            nav_count = sum(1 for kw in nav_keywords if kw in text_str)
            if nav_count >= 3:
                # 有底部导航栏，不是广告页
                pass
            else:
                return PageDetectionResult(
                    state=PageState.AD,
                    confidence=0.7,
                    details="疑似广告页（仅有跳过按钮）"
                )
        
        # 14.5. 检测抽奖页面（新增）
        # 抽奖页面特征：必须同时满足多个条件
        # 1. 有"抽奖"相关关键词
        # 2. 有"立即抽奖"或"开始抽奖"按钮
        # 3. 有剩余次数显示（或"次数已用完"）
        if "抽奖" in text_str or "幸运抽奖" in text_str:
            # 检查是否有抽奖按钮
            has_draw_button = any(kw in text_str for kw in ["立即抽奖", "开始抽奖", "免费抽奖", "抽一次"])
            
            # 检查是否有次数信息
            has_times_info = any(kw in text_str for kw in ["剩余", "次", "机会", "次数已用完", "今日"])
            
            if has_draw_button and has_times_info:
                return PageDetectionResult(
                    state=PageState.UNKNOWN,  # 不新增状态，用UNKNOWN + details
                    confidence=0.95,
                    details="抽奖页面（OCR确认：按钮+次数）"
                )
        
        # 15. 检测加载页
        if "溪盟商城" in text_str and len(texts) <= 5:
            return PageDetectionResult(
                state=PageState.LOADING,
                confidence=0.8,
                details="加载页（OCR确认）"
            )
        
        # 16. 检测具体弹窗类型
        if pixel_result.state == PageState.POPUP:
            return PageDetectionResult(
                state=PageState.POPUP,
                confidence=0.9,
                details=f"弹窗（OCR确认，识别到{len(texts)}个文本）"
            )
        
        return pixel_result
    
    async def detect_ad_skip_button(self, device_id: str) -> Optional[Tuple[int, int]]:
        """检测广告跳过按钮位置
        
        Returns:
            跳过按钮坐标，未找到返回 None
        """
        if not self._ocr_pool:
            return None
        
        image = await self._get_screenshot(device_id)
        if not image:
            return None
        
        try:
            ocr_result = self._ocr_pool.ocr(image)
            if ocr_result and ocr_result.boxes and ocr_result.txts:
                # 查找跳过按钮
                skip_keywords = ["跳过", "skip", "Skip", "SKIP"]
                
                for box, text in zip(ocr_result.boxes, ocr_result.txts):
                    if any(kw in text for kw in skip_keywords):
                        # 计算中心点
                        x_coords = [p[0] for p in box]
                        y_coords = [p[1] for p in box]
                        x_center = int(sum(x_coords) / 4)
                        y_center = int(sum(y_coords) / 4)
                        
                        # 跳过按钮通常在右上角（x > 400, y < 100）
                        if x_center > 400 and y_center < 150:
                            print(f"    [广告] 找到跳过按钮: '{text}' at ({x_center}, {y_center})")
                            return (x_center, y_center)
        except Exception as e:
            print(f"    [广告] 检测跳过按钮失败: {e}")
        
        return None
    
    async def skip_ad(self, device_id: str) -> bool:
        """跳过广告
        
        Returns:
            是否成功跳过
        """
        # 先检测是否是广告页
        result = await self.detect_page(device_id, use_ocr=True)
        if result.state != PageState.AD:
            print(f"    [广告] 当前不是广告页: {result.state.value}")
            return False
        
        # 查找跳过按钮
        skip_pos = await self.detect_ad_skip_button(device_id)
        
        if skip_pos:
            print(f"    [广告] 点击跳过按钮: {skip_pos}")
            await self.adb.tap(device_id, skip_pos[0], skip_pos[1])
            await asyncio.sleep(1)
            return True
        else:
            print(f"    [广告] 未找到跳过按钮，等待广告自动关闭...")
            # 等待5秒看广告是否自动关闭
            await asyncio.sleep(5)
            return False
    
    async def detect_popup_type(self, device_id: str) -> Tuple[Optional[str], Optional[Tuple[int, int]]]:
        """检测弹窗类型并返回关闭按钮坐标
        
        Returns:
            (弹窗类型, 关闭按钮坐标) 或 (None, None)
        """
        if not self._ocr_pool:
            # 没有 OCR，返回通用弹窗坐标
            return "generic", self.POPUP_BUTTONS['generic']
        
        image = await self._get_screenshot(device_id)
        if not image:
            return "generic", self.POPUP_BUTTONS['generic']
        
        texts = self._ocr_image(image)
        text_str = " ".join(texts) if texts else ""
        
        print(f"    [OCR] 识别到: {texts[:5] if texts else '无'}...")
        
        # 登录错误弹窗（最高优先级 - 必须先检测）
        if "友情提示" in text_str:
            return "login_error", self.POPUP_BUTTONS['login_error']
        
        # 用户协议弹窗 - 检测更多关键词
        if any(kw in text_str for kw in ["用户协议", "隐私政策", "服务协议", "隐私协议"]):
            # 但要排除登录页面（登录页面底部也有用户协议文字）
            if "登录" not in text_str or "同意并接受" in text_str:
                return "user_agreement", self.POPUP_BUTTONS['user_agreement']
        
        # 主页公告弹窗 - 检测更多关键词
        if any(kw in text_str for kw in ["公告", "活动", "恭喜", "领取", "×"]):
            return "home_announcement", self.POPUP_BUTTONS['home_announcement']
        
        # 通用弹窗（有确定/关闭/取消按钮）
        if any(kw in text_str for kw in ["确定", "关闭", "取消", "知道了", "我知道了"]):
            return "generic", self.POPUP_BUTTONS['generic']
        
        # 如果识别不出来，返回通用弹窗坐标（尝试关闭）
        return "unknown", self.POPUP_BUTTONS['generic']
    
    async def find_popup_close_button(self, device_id: str, timeout: float = 5.0) -> Optional[Tuple[int, int]]:
        """使用OCR查找弹窗关闭按钮位置
        
        Args:
            device_id: 设备ID
            timeout: OCR超时时间（秒），默认5秒
        
        Returns:
            关闭按钮坐标，未找到返回 None
        """
        if not self._ocr_pool:
            return None
        
        image = await self._get_screenshot(device_id)
        if not image:
            return None
        
        try:
            # 使用 OCR 线程池进行识别（带超时）
            ocr_result = await self._ocr_pool.recognize(image, timeout=timeout, use_cache=True)
            
            if ocr_result and ocr_result.texts:
                # 查找关闭按钮关键词
                # 严格规则：只能点击同意类按钮，不能点击拒绝类按钮
                agree_keywords = ["同意并接受", "同意并登录", "同意", "知道了", "我知道了", "确定", "立即领取", "去看看"]
                reject_keywords = ["暂不同意", "不同意", "取消", "拒绝"]
                
                # 检查是否有拒绝按钮
                has_reject_button = any(
                    any(keyword in str(text) for keyword in reject_keywords)
                    for text in ocr_result.texts
                )
                
                if has_reject_button:
                    print(f"    [弹窗] 检测到拒绝按钮，需要找到同意按钮")
                
                # 查找同意按钮
                for keyword in agree_keywords:
                    for text in ocr_result.texts:
                        text_str = str(text) if text is not None else ""
                        if keyword in text_str:
                            print(f"    [弹窗] 找到关闭按钮关键词: '{text_str}'")
                            # 由于没有位置信息，返回 None 让调用者使用预设位置
                            # 但至少我们知道有这个按钮
                            return None
                
                if has_reject_button:
                    print(f"    [弹窗] ⚠️ 检测到拒绝按钮但未找到同意按钮")
                    return None
                
        except Exception as e:
            print(f"    [弹窗] OCR查找关闭按钮失败: {e}")
        
        return None
    
    async def close_popup(self, device_id: str, timeout: float = 15.0) -> bool:
        """自动关闭弹窗（带超时保护和重试机制）
        
        Args:
            device_id: 设备ID
            timeout: 总超时时间（秒），默认15秒
        
        Returns:
            是否成功关闭
        """
        import asyncio
        
        try:
            # 使用 asyncio.wait_for 为整个关闭流程添加超时
            return await asyncio.wait_for(
                self._close_popup_impl(device_id),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            print(f"    [弹窗] ✗ 关闭弹窗超时（{timeout}秒）")
            return False
    
    async def _close_popup_impl(self, device_id: str) -> bool:
        """关闭弹窗的实际实现（带重试验证）"""
        from .retry_helper import retry_until_success
        
        # 优先使用模板匹配结果来判断弹窗类型
        popup_type = None
        button_pos = None
        
        # 检查是否有缓存的检测结果（包含模板名称）
        cached_result = self._detection_cache.get(device_id)
        if cached_result and cached_result.state == PageState.POPUP:
            # 从检测结果的details中提取模板名称
            if "首页公告.png" in cached_result.details:
                popup_type = "home_announcement"
                button_pos = None  # 首页公告弹窗点击外部关闭
                print(f"    [弹窗] 类型: {popup_type} (模板匹配)")
            elif "启动页服务弹窗.png" in cached_result.details:
                popup_type = "user_agreement"
                button_pos = self.POPUP_BUTTONS['user_agreement']
                print(f"    [弹窗] 类型: {popup_type} (模板匹配)")
            elif "签到弹窗.png" in cached_result.details:
                popup_type = "checkin_popup"
                button_pos = None  # 签到弹窗使用专用坐标列表
                print(f"    [弹窗] 类型: {popup_type} (模板匹配)")
            elif "温馨提示.png" in cached_result.details:
                popup_type = "generic"
                button_pos = self.POPUP_BUTTONS['generic']
                print(f"    [弹窗] 类型: {popup_type} (模板匹配)")
            elif "用户名或密码错误.png" in cached_result.details or "手机号码不存在.png" in cached_result.details:
                popup_type = "login_error"
                button_pos = self.POPUP_BUTTONS['login_error']
                print(f"    [弹窗] 类型: {popup_type} (模板匹配)")
        
        # 如果模板匹配没有识别出类型，使用OCR检测
        if popup_type is None:
            popup_type, button_pos = await self.detect_popup_type(device_id)
            print(f"    [弹窗] 类型: {popup_type} (OCR检测)")
        
        # 如果是首页公告弹窗，点击弹窗外部关闭（更可靠）
        if popup_type == "home_announcement":
            print(f"    [弹窗] 首页公告弹窗，点击外部区域关闭...")
            # 点击屏幕顶部区域（弹窗外）
            await self.adb.tap(device_id, 270, 100)
            await asyncio.sleep(2)
            
            # 检查是否成功关闭
            result = await self.detect_page(device_id, use_ocr=True)
            if result.state != PageState.POPUP:
                print(f"    [弹窗] ✓ 成功关闭首页公告弹窗")
                return True
            else:
                print(f"    [弹窗] ⚠️ 点击外部未关闭，尝试其他位置...")
                # 尝试点击底部区域
                await self.adb.tap(device_id, 270, 850)
                await asyncio.sleep(2)
                
                result = await self.detect_page(device_id, use_ocr=True)
                if result.state != PageState.POPUP:
                    print(f"    [弹窗] ✓ 成功关闭首页公告弹窗")
                    return True
        
        # 检查是否是签到奖励弹窗（通过OCR识别关键词）
        is_checkin_popup = (popup_type == "checkin_popup")
        if not is_checkin_popup and self._last_screenshot and self._ocr_pool:
            texts = self._ocr_image(self._last_screenshot)
            text_str = ''.join(texts)
            # 签到弹窗特征：有"恭喜"、"成功"、"知道了"等关键词
            if ("恭喜" in text_str and "成功" in text_str) or "知道了" in text_str:
                is_checkin_popup = True
                print(f"    [弹窗] 检测到签到奖励弹窗 (OCR确认)")
        
        # 如果是签到弹窗，使用专用坐标
        if is_checkin_popup:
            print(f"    [弹窗] 使用签到弹窗专用坐标...")
            for i, (x, y) in enumerate(self.CHECKIN_POPUP_CLOSE, 1):
                print(f"    [弹窗] 尝试位置 {i}/3: ({x}, {y})")
                await self.adb.tap(device_id, x, y)
                await asyncio.sleep(2)
                
                # 检查是否成功关闭
                result = await self.detect_page(device_id, use_ocr=True)
                if result.state != PageState.POPUP:
                    print(f"    [弹窗] ✓ 成功关闭签到弹窗")
                    return True
            
            print(f"    [弹窗] ⚠️ 签到弹窗专用坐标都失败，尝试其他方法...")
        
        # 先用OCR查找关闭按钮
        close_button_pos = await self.find_popup_close_button(device_id)
        
        if close_button_pos:
            print(f"    [弹窗] 使用OCR找到的按钮位置: {close_button_pos}")
            
            # 使用重试机制，点击后验证是否成功关闭
            async def check_closed():
                result = await self.detect_page(device_id, use_ocr=True)
                return result.state != PageState.POPUP
            
            success = await retry_until_success(
                self.adb.tap,
                check_closed,
                max_attempts=2,
                delay=2.0,
                device_id=device_id,
                x=close_button_pos[0],
                y=close_button_pos[1]
            )
            
            if success:
                print(f"    [弹窗] ✓ 成功关闭")
                return True
            else:
                print(f"    [弹窗] ⚠️ 点击后仍是弹窗，尝试其他方法...")
        else:
            print(f"    [弹窗] 未通过OCR找到关闭按钮，使用预设位置...")
        
        # 如果OCR没找到或点击失败，使用预设位置
        print(f"    [弹窗] 预设按钮: {button_pos}")
        
        if button_pos:
            await self.adb.tap(device_id, button_pos[0], button_pos[1])
            await asyncio.sleep(2)  # 增加等待时间
            
            # 检查是否成功关闭
            result = await self.detect_page(device_id, use_ocr=True)
            if result.state != PageState.POPUP:
                print(f"    [弹窗] ✓ 成功关闭")
                return True
            else:
                print(f"    [弹窗] ⚠️ 预设位置点击失败，仍是弹窗")
            
            # 如果是未知类型或首页公告弹窗，尝试多个位置
            if popup_type in ["unknown", "home_announcement", "user_agreement"]:
                print(f"    [弹窗] 尝试其他预设位置...")
                # 尝试其他常见位置（按成功率从高到低排序）
                alternative_positions = [
                    (270, 608),  # 备用位置1
                    (270, 620),  # 稍微靠下
                    (270, 650),  # 更靠下的位置
                    (270, 550),  # 更靠上
                ]
                
                for pos in alternative_positions:
                    await self.adb.tap(device_id, pos[0], pos[1])
                    await asyncio.sleep(1.5)
                    
                    # 检查是否成功关闭
                    result = await self.detect_page(device_id, use_ocr=True)
                    if result.state != PageState.POPUP:
                        print(f"    [弹窗] ✓ 成功关闭（位置: {pos}）")
                        return True
                    else:
                        print(f"    [弹窗] ⚠️ 位置 {pos} 失败，仍是弹窗")
                
                # 如果所有位置都失败，尝试按返回键
                print(f"    [弹窗] 所有位置都失败，尝试按返回键...")
                await self.adb.press_back(device_id)
                await asyncio.sleep(1.5)
                
                result = await self.detect_page(device_id, use_ocr=True)
                if result.state != PageState.POPUP:
                    print(f"    [弹窗] ✓ 成功关闭（返回键）")
                    return True
                else:
                    print(f"    [弹窗] ✗ 返回键也失败，弹窗无法关闭")
                    return False  # 明确返回失败
        
        print(f"    [弹窗] ✗ 无法关闭弹窗")
        return False  # 明确返回失败
    
    async def wait_for_page(self, device_id: str, target_state: PageState,
                           timeout: int = 30, check_interval: float = 1.0,
                           auto_close_popup: bool = True) -> bool:
        """等待指定页面出现
        
        Args:
            device_id: 设备 ID
            target_state: 目标页面状态
            timeout: 超时时间（秒）
            check_interval: 检测间隔（秒）
            auto_close_popup: 是否自动关闭弹窗
        """
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            result = await self.detect_page(device_id)
            
            if result.state == target_state:
                return True
            
            # 自动关闭弹窗
            if auto_close_popup and result.state == PageState.POPUP:
                await self.close_popup(device_id)
            
            await asyncio.sleep(check_interval)
        
        return False
    
    async def is_logged_in(self, device_id: str) -> bool:
        """检查是否已登录（使用 OCR 确认）"""
        result = await self.detect_page(device_id, use_ocr=True)
        return result.state == PageState.PROFILE_LOGGED
    
    def clear_cache(self, device_id: str = None):
        """清除检测缓存
        
        Args:
            device_id: 设备ID，如果为None则清除所有缓存
        """
        self._detection_cache.clear(device_id)
    
    def get_cache_stats(self) -> dict:
        """获取缓存统计信息
        
        Returns:
            包含缓存统计的字典
        """
        return self._detection_cache.get_stats()
