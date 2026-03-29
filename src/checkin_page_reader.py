"""
签到页面信息读取模块
Check-in Page Reader Module
"""

import re
import asyncio
from typing import Optional, Dict, Tuple, List
from io import BytesIO
from PIL import Image

# 在导入 RapidOCR 之前设置日志级别
import logging
for logger_name in ['rapidocr', 'RapidOCR', 'ppocr', 'onnxruntime']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

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
        
        # [2026-03-11] 新增：获取YOLO检测器用于快速定位签到次数区域
        from .model_manager import ModelManager
        model_manager = ModelManager.get_instance()
        self._yolo_detector = model_manager.get_page_detector_integrated()
        
        # [2026-02-21] 删除学习器：移除 OCRRegionLearner
    
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
            
            # [2026-03-11] 新增：YOLO+OCR混合方案优化性能
            # 优先尝试使用YOLO快速定位签到次数区域，然后只对该区域进行OCR
            yolo_success = False
            if self._yolo_detector:
                try:
                    # 使用YOLO检测签到页面元素
                    yolo_result = await self._yolo_detector.detect_elements_yolo(
                        device_id, 
                        model_key="签到页",  # 使用签到页YOLO模型
                        target_classes=["签到次数", "签到按钮"]  # 检测签到次数和签到按钮
                    )
                    
                    if yolo_result and yolo_result.get('签到次数'):
                        # 找到签到次数区域，进行区域OCR
                        checkin_times_region = yolo_result['签到次数'][0]  # 取第一个检测结果
                        
                        # 提取区域坐标并扩展边界（增加10像素边距提高识别率）
                        x1, y1, x2, y2 = checkin_times_region['bbox']
                        margin = 10
                        x1 = max(0, x1 - margin)
                        y1 = max(0, y1 - margin)
                        x2 = min(img.width, x2 + margin)
                        y2 = min(img.height, y2 + margin)
                        
                        # 裁剪签到次数区域
                        times_region = img.crop((x1, y1, x2, y2))
                        
                        # 对区域进行OCR识别（使用相同的图像增强）
                        enhanced_region = enhance_for_ocr(times_region)
                        
                        # 使用OCR线程池识别区域（超时时间减少到2秒）
                        ocr_result = await self._ocr_pool.recognize(enhanced_region, timeout=2.0)
                        
                        if ocr_result and ocr_result.texts is not None and len(ocr_result.texts) > 0:
                            texts = ocr_result.texts
                            result['raw_text'] = ' '.join(texts)
                            
                            # 解析签到次数信息
                            self._parse_checkin_times(texts, result)
                            yolo_success = True
                            
                            # [2026-03-12] 优化日志：移除CMD控制台的YOLO+OCR技术信息
                    
                    # 处理签到按钮检测
                    if yolo_result and yolo_result.get('签到按钮'):
                        result['can_checkin'] = True
                        button_info = yolo_result['签到按钮'][0]  # 取第一个检测结果
                        # 计算按钮中心点
                        x1, y1, x2, y2 = button_info['bbox']
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        result['checkin_button_pos'] = (center_x, center_y)
                        
                except Exception as e:
                    print(f"[签到页面YOLO] ⚠️ YOLO检测失败，降级到全屏OCR: {e}")
            
            # [2026-03-29] 修改原因：YOLO检测失败时不再降级到全屏OCR，避免加剧阻塞
            # 如果YOLO失败，直接返回空结果
            if not yolo_success:
                print(f"[签到页面OCR] YOLO检测失败，跳过OCR识别")
                return result
                
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
            # [2026-03-01] 禁用调试输出：减少日志冗余
            # print(f"[签到页面OCR] 发生错误: {str(e)}")
            # print(f"[签到页面OCR] 错误详情:\n{error_details}")
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
                    # [2026-03-01] 禁用调试输出：减少日志冗余
                    # print(f"[签到页面OCR] 从文本 '{text}' 中提取总次数: {match.group(1)}")
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
                # [2026-03-01] 禁用调试输出：减少日志冗余
                # print(f"[签到页面OCR] 跨文本匹配总次数: {result['total_times']}")
        
        if result['daily_remaining_times'] is None:
            # 匹配 "当日还有" 或 "当日剩余" 后面跟着数字
            match = re.search(r'当日(?:还有|剩余)\s*[:：]?\s*(\d+)', full_text)
            if match:
                result['daily_remaining_times'] = int(match.group(1))
                # [2026-03-01] 禁用调试输出：减少日志冗余
                # print(f"[签到页面OCR] 跨文本匹配剩余次数: {result['daily_remaining_times']}")
    
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
            
            # [2026-03-05] 修复数组比较错误：检查 texts 是否为 None 或长度为 0
            if not ocr_result or ocr_result.texts is None or len(ocr_result.texts) == 0:
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
                    
                    # 尝试关闭可能出现的弹窗（使用统一的弹窗关闭方法）
                    try:
                        from .model_manager import ModelManager
                        model_manager = ModelManager.get_instance()
                        detector = model_manager.get_page_detector_integrated()
                        if detector:
                            await detector.close_popup(device_id)
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
