"""
签到页面信息读取模块
Check-in Page Reader Module
"""

import re
import asyncio
from typing import Optional, Dict, Tuple, List
from io import BytesIO
from PIL import Image

try:
    from rapidocr import RapidOCR
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

from .adb_bridge import ADBBridge
from .ocr_image_processor import enhance_for_ocr
from .ocr_thread_pool import get_ocr_pool


class CheckinPageReader:
    """签到页面信息读取器"""
    
    def __init__(self, adb: ADBBridge):
        """初始化读取器
        
        Args:
            adb: ADB桥接对象
        """
        self.adb = adb
        
        # 使用全局 OCR 线程池（替代单独的 OCR 实例）
        self._ocr_pool = get_ocr_pool() if HAS_OCR else None
    
    async def get_checkin_info(self, device_id: str) -> Dict[str, any]:
        """获取签到页面信息
        
        Args:
            device_id: 设备ID
            
        Returns:
            dict: 签到信息
                - total_times: int, 总次数（总共可以签到的次数）
                - daily_remaining_times: int, 当日剩余次数（今天还可以签到的次数）
                - can_checkin: bool, 是否可以签到
                - checkin_button_pos: tuple, 签到按钮位置
                - raw_text: str, 原始文本（用于调试）
        """
        result = {
            'total_times': None,
            'daily_remaining_times': None,
            'can_checkin': False,
            'checkin_button_pos': None,
            'raw_text': ''
        }
        
        if not self._ocr_pool:
            return result
        
        try:
            # 获取截图
            screenshot = await self.adb.screencap(device_id)
            if not screenshot:
                return result
            
            img = Image.open(BytesIO(screenshot))
            
            # 使用全屏OCR识别（简单快速）
            # 使用OCR图像预处理模块增强图像（灰度图 + 对比度增强2倍）
            enhanced_img = enhance_for_ocr(img)
            
            # 使用 OCR 线程池识别（异步，带超时）
            ocr_result = await self._ocr_pool.recognize(enhanced_img, timeout=2.0)  # 优化：减少超时 10秒→2秒
            
            # 检查返回值
            if not ocr_result or not ocr_result.texts:
                return result
            
            texts = ocr_result.texts
            # 修复：正确处理numpy数组
            boxes = ocr_result.boxes if ocr_result.boxes is not None and len(ocr_result.boxes) > 0 else []
            
            # 合并所有文本用于调试
            result['raw_text'] = ' '.join(texts)
            
            # 调试：输出所有识别到的文本
            print(f"[签到页面OCR] 识别到 {len(texts)} 段文本:")
            for idx, txt in enumerate(texts):
                print(f"  [{idx}] {txt}")
            
            # 解析签到信息
            self._parse_checkin_times(texts, result)
            
            # 查找签到按钮（在循环外单独处理）
            for i, text in enumerate(texts):
                if '立即签到' in text or '点击签到' in text or '签到' in text:
                    result['can_checkin'] = True
                    
                    # 获取按钮位置
                    if i < len(boxes):
                        box = boxes[i]
                        x_coords = [p[0] for p in box]
                        y_coords = [p[1] for p in box]
                        center_x = int(sum(x_coords) / len(x_coords))
                        center_y = int(sum(y_coords) / len(y_coords))
                        result['checkin_button_pos'] = (center_x, center_y)
                        break
            
            return result
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"[签到页面OCR] 发生错误: {str(e)}")
            print(f"[签到页面OCR] 错误详情:\n{error_details}")
            result['raw_text'] = f"Error: {str(e)}"
            return result
    
    def _parse_checkin_times(self, texts: list, result: dict):
        """解析签到次数信息（提取为独立方法，便于复用）
        
        Args:
            texts: OCR识别的文本列表
            result: 结果字典（会被修改）
        """
        # 解析签到信息
        for i, text in enumerate(texts):
            # 解析总次数和当日剩余次数
            # 格式1: "您总次数为108,您当日还有1次签到任务"
            # 格式2: "您总次数为107，您当日还有0次签到任务"
            match = re.search(r'总次数为(\d+)[,，].*?当日还有(\d+)次', text)
            if match:
                result['total_times'] = int(match.group(1))
                result['daily_remaining_times'] = int(match.group(2))
                continue
            
            # 备用格式: 分开的文本
            # "总次数: 108" 或 "总次数：108" 或 "总次数为108"
            match = re.search(r'总次数[:：为]\s*(\d+)', text)
            if match:
                result['total_times'] = int(match.group(1))
                continue
            
            # 新增：更宽松的总次数匹配（处理OCR识别错误）
            # "总次数108" 或 "总次 数108" 或 "总 次数108"
            match = re.search(r'总\s*次\s*数\s*[:：为]?\s*(\d+)', text)
            if match:
                result['total_times'] = int(match.group(1))
                continue
            
            # 新增：匹配单独的数字（如果前面有"总次数"相关文字）
            if '总' in text and '次' in text:
                # 在同一行或下一行查找数字
                match = re.search(r'(\d+)', text)
                if match and result['total_times'] is None:
                    print(f"[签到页面OCR] 从文本 '{text}' 中提取总次数: {match.group(1)}")
                    result['total_times'] = int(match.group(1))
                    continue
            
            # "当日还有1次" 或 "当日剩余: 1"
            match = re.search(r'当日(?:还有|剩余)[:：]?\s*(\d+)次?', text)
            if match:
                result['daily_remaining_times'] = int(match.group(1))
                continue
        
        # 新增：跨文本匹配（处理总次数和数字分开识别的情况）
        full_text = ' '.join(texts)
        if result['total_times'] is None:
            # 匹配 "总次数" 后面跟着数字（可能有空格或其他字符）
            match = re.search(r'总\s*次\s*数\s*[:：为]?\s*(\d+)', full_text)
            if match:
                result['total_times'] = int(match.group(1))
                print(f"[签到页面OCR] 跨文本匹配总次数: {result['total_times']}")
        
        if result['daily_remaining_times'] is None:
            # 匹配 "当日还有" 或 "当日剩余" 后面跟着数字
            match = re.search(r'当日(?:还有|剩余)\s*[:：]?\s*(\d+)', full_text)
            if match:
                result['daily_remaining_times'] = int(match.group(1))
                print(f"[签到页面OCR] 跨文本匹配剩余次数: {result['daily_remaining_times']}")
    
    async def can_checkin_today(self, device_id: str) -> bool:
        """检查今天是否还可以签到
        
        Args:
            device_id: 设备ID
            
        Returns:
            bool: 是否可以签到
        """
        info = await self.get_checkin_info(device_id)
        return info['can_checkin'] and (info['daily_remaining_times'] or 0) > 0

    async def extract_checkin_reward(self, device_id: str) -> Optional[float]:
        """提取单次签到奖励金额
        
        从签到结果页面或弹窗中提取奖励金额。
        支持多种格式：
        - "恭喜获得 1.50 元"
        - "奖励: 1.20元"
        - "+1.00"
        - "1.50元"
        
        Args:
            device_id: 设备ID
            
        Returns:
            float: 奖励金额，如果提取失败返回None
        """
        if not self._ocr_pool:
            return None
        
        try:
            # 获取截图
            screenshot = await self.adb.screencap(device_id)
            if not screenshot:
                return None
            
            img = Image.open(BytesIO(screenshot))
            
            # 使用OCR图像预处理模块增强图像（灰度图 + 对比度增强2倍）
            enhanced_img = enhance_for_ocr(img)
            
            # 使用 OCR 线程池识别（异步，带超时）
            ocr_result = await self._ocr_pool.recognize(enhanced_img, timeout=2.0)  # 优化：减少超时 10秒→2秒
            
            # 检查返回值
            if not ocr_result or not ocr_result.texts:
                return None
            
            texts = ocr_result.texts
            
            # 合并所有文本
            full_text = ' '.join(texts)
            
            # 尝试多种模式匹配
            patterns = [
                r'恭喜获得\s*([0-9]+\.?[0-9]*)\s*元',  # "恭喜获得 1.50 元"
                r'奖励[:：]\s*([0-9]+\.?[0-9]*)\s*元',  # "奖励: 1.20元"
                r'\+\s*([0-9]+\.?[0-9]*)',              # "+1.00"
                r'获得\s*([0-9]+\.?[0-9]*)\s*元',       # "获得 1.50 元"
                r'([0-9]+\.?[0-9]*)\s*元',              # "1.50元"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, full_text)
                if match:
                    try:
                        reward = float(match.group(1))
                        return reward
                    except ValueError:
                        continue
            
            return None
            
        except Exception as e:
            print(f"  ⚠️ 提取签到奖励失败: {str(e)}")
            return None
    
    async def perform_multiple_checkins(self, device_id: str, max_checkins: int = 10) -> List[float]:
        """执行多次签到并收集所有奖励
        
        循环执行签到操作，直到没有剩余次数或达到最大尝试次数。
        每次签到后提取奖励金额并累积。
        
        Args:
            device_id: 设备ID
            max_checkins: 最大签到次数（防止无限循环）
            
        Returns:
            List[float]: 所有签到奖励的列表，例如 [1.00, 1.20, 1.30]
        """
        rewards = []
        consecutive_failures = 0  # 连续失败次数
        
        try:
            for attempt in range(max_checkins):
                try:
                    # 获取当前签到信息
                    checkin_info = await self.get_checkin_info(device_id)
                    
                    # 检查是否还有剩余次数
                    remaining = checkin_info.get('daily_remaining_times', 0)
                    if remaining <= 0:
                        print(f"  ✓ 签到完成，当日无剩余次数")
                        break
                    
                    # 检查是否可以签到
                    if not checkin_info.get('can_checkin'):
                        print(f"  ⚠️ 未找到签到按钮")
                        consecutive_failures += 1
                        if consecutive_failures >= 3:
                            print(f"  ✗ 连续3次未找到签到按钮，停止签到")
                            break
                        await asyncio.sleep(2)
                        continue
                    
                    # 获取签到按钮位置
                    button_pos = checkin_info.get('checkin_button_pos')
                    if not button_pos:
                        print(f"  ⚠️ 未找到签到按钮位置")
                        consecutive_failures += 1
                        if consecutive_failures >= 3:
                            print(f"  ✗ 连续3次未找到签到按钮位置，停止签到")
                            break
                        await asyncio.sleep(2)
                        continue
                    
                    print(f"  [签到 {len(rewards) + 1}] 剩余次数: {remaining}")
                    
                    # 点击签到按钮
                    try:
                        await self.adb.tap(device_id, button_pos[0], button_pos[1])
                    except Exception as e:
                        print(f"  ⚠️ 点击签到按钮失败: {str(e)}")
                        consecutive_failures += 1
                        if consecutive_failures >= 3:
                            print(f"  ✗ 连续3次点击失败，停止签到")
                            break
                        await asyncio.sleep(2)
                        continue
                    
                    # 等待签到结果显示
                    await asyncio.sleep(2)
                    
                    # 提取奖励金额
                    try:
                        reward = await self.extract_checkin_reward(device_id)
                        if reward is not None:
                            rewards.append(reward)
                            print(f"  ✓ 获得奖励: {reward:.2f} 元")
                            consecutive_failures = 0  # 重置连续失败计数
                        else:
                            # 即使提取失败，也记录0.0，表示签到已执行
                            rewards.append(0.0)
                            print(f"  ⚠️ 未能提取奖励金额，记录为 0.00 元")
                            consecutive_failures += 1
                    except Exception as e:
                        print(f"  ⚠️ 提取奖励金额出错: {str(e)}")
                        rewards.append(0.0)
                        consecutive_failures += 1
                    
                    # 等待一下，让页面更新
                    await asyncio.sleep(1)
                    
                    # 尝试关闭可能出现的弹窗
                    try:
                        # 点击屏幕中心或其他位置来关闭弹窗
                        await self.adb.tap(device_id, 270, 960)  # 屏幕中心位置
                        await asyncio.sleep(1)
                    except Exception as e:
                        print(f"  ⚠️ 关闭弹窗失败: {str(e)}")
                    
                    # 如果连续失败次数过多，警告但继续
                    if consecutive_failures >= 3:
                        print(f"  ⚠️ 警告：连续失败 {consecutive_failures} 次，可能出现问题")
                        print(f"  继续尝试...")
                    
                except Exception as e:
                    print(f"  ⚠️ 签到过程出错: {str(e)}")
                    consecutive_failures += 1
                    if consecutive_failures >= 5:
                        print(f"  ✗ 连续5次出错，停止签到")
                        break
                    await asyncio.sleep(2)
                    continue
            
            # 返回收集到的奖励
            if rewards:
                print(f"\n  签到总结:")
                print(f"  ✓ 成功签到 {len(rewards)} 次")
                print(f"  ✓ 总奖励: {sum(rewards):.2f} 元")
            else:
                print(f"\n  ⚠️ 未能完成任何签到")
            
            return rewards
            
        except Exception as e:
            print(f"  ✗ 执行多次签到时出错: {str(e)}")
            # 即使出错，也返回已收集的奖励
            if rewards:
                print(f"  ⚠️ 已收集 {len(rewards)} 次签到奖励")
            return rewards
