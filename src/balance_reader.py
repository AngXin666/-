"""
余额读取模块 - 使用 OCR 识别账户信息
Balance Reader Module - OCR-based account info extraction
"""

import re
from typing import Optional
from dataclasses import dataclass
from io import BytesIO

# 在导入 RapidOCR 之前设置日志级别
import logging
for logger_name in ['rapidocr', 'RapidOCR', 'ppocr', 'onnxruntime']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

try:
    from rapidocr import RapidOCR
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from .adb_bridge import ADBBridge
from .page_detector import PageDetector, PageState


@dataclass
class AccountInfo:
    """账户信息"""
    user_id: Optional[str] = None
    balance: Optional[str] = None
    points: Optional[str] = None
    coupons: Optional[str] = None
    vouchers: Optional[str] = None  # 优惠券


class BalanceReader:
    """余额读取器 - 基于 OCR"""
    
    # 540x960 分辨率下的坐标
    TAB_MY = (450, 920)  # "我的" 标签
    
    def __init__(self, adb: ADBBridge):
        """初始化余额读取器
        
        Args:
            adb: ADB 桥接器实例
        """
        self.adb = adb
        self.page_detector = PageDetector(adb)
        self._ocr = None
        
        if HAS_OCR:
            self._ocr = RapidOCR()
    
    async def get_account_info(self, device_id: str, 
                               navigate_to_profile: bool = True) -> Optional[AccountInfo]:
        """获取账户信息
        
        Args:
            device_id: 设备 ID
            navigate_to_profile: 是否自动导航到我的页面
            
        Returns:
            账户信息，失败返回 None
        """
        if not HAS_OCR or not HAS_PIL:
            return None
        
        # 确保在我的页面
        if navigate_to_profile:
            result = await self.page_detector.detect_page(device_id)
            if result.state == PageState.HOME:
                await self.adb.tap(device_id, self.TAB_MY[0], self.TAB_MY[1])
                import asyncio
                await asyncio.sleep(2)
        
        # 截图
        screenshot = await self.adb.screencap(device_id)
        if not screenshot:
            return None
        
        img = Image.open(BytesIO(screenshot))
        
        # OCR 识别（添加超时保护）
        import asyncio
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self._ocr, img),
                timeout=10.0
            )
        except asyncio.TimeoutError:
            print("  [余额读取] OCR识别超时（10秒）")
            return None
        except Exception as e:
            print(f"  [余额读取] OCR识别异常: {e}")
            return None
        
        if not result or not result.txts:
            return None
        
        return self._parse_account_info(result.txts)
    
    def _parse_account_info(self, texts: list) -> AccountInfo:
        """解析 OCR 结果
        
        Args:
            texts: OCR 识别的文本列表
            
        Returns:
            账户信息
        """
        info = AccountInfo()
        
        # 找到 "余额" 的位置
        balance_idx = -1
        for i, text in enumerate(texts):
            if text == "余额":
                balance_idx = i
                break
        
        if balance_idx > 0:
            # 余额、积分、抵扣券、优惠券的值在它们标签之前
            # 找到标签前面的数字
            numbers = []
            for j in range(balance_idx - 1, -1, -1):
                if re.match(r'^[\d.]+$', texts[j]) and ":" not in texts[j]:
                    numbers.insert(0, texts[j])
                    if len(numbers) >= 4:
                        break
            
            # 按顺序分配：余额、积分、抵扣券、优惠券
            if len(numbers) >= 1:
                info.balance = numbers[0]
            if len(numbers) >= 2:
                info.points = numbers[1]
            if len(numbers) >= 3:
                info.coupons = numbers[2]
            if len(numbers) >= 4:
                info.vouchers = numbers[3]
        
        # 用户ID
        for text in texts:
            if text.startswith("ID:"):
                info.user_id = text.replace("ID:", "")
                break
        
        return info
    
    async def get_balance(self, device_id: str, navigate_to_profile: bool = True) -> Optional[str]:
        """获取余额（简化接口）
        
        Args:
            device_id: 设备 ID
            navigate_to_profile: 是否自动导航到我的页面（默认True）
            
        Returns:
            余额字符串，失败返回 None
        """
        info = await self.get_account_info(device_id, navigate_to_profile=navigate_to_profile)
        return info.balance if info else None
