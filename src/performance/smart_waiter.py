"""
智能等待器
Smart Waiter - 真正的页面变化检测
"""

import asyncio
import hashlib
from typing import List, Optional


class SmartWaiter:
    """智能等待器
    
    核心理念：
    1. 高频轮询检测页面变化（不是等待固定时间）
    2. 检测到变化后立即用深度学习确认
    3. 超时只是防止卡死的保护机制，不影响正常检测
    """
    
    def __init__(self):
        """初始化智能等待器"""
        # 获取静默日志记录器（用于详细调试信息）
        from ..logger import get_silent_logger
        self._debug_logger = get_silent_logger()
        
        # 上一次截图的哈希值（用于检测页面变化）
        self._last_screenshot_hash = {}
    
    async def _get_screenshot_hash(self, device_id: str, adb_bridge) -> Optional[str]:
        """获取截图的哈希值（用于快速检测页面变化）
        
        Args:
            device_id: 设备ID
            adb_bridge: ADB桥接对象
            
        Returns:
            截图的MD5哈希值，失败返回None
        """
        try:
            # 获取截图数据
            screenshot_data = await adb_bridge.screencap(device_id)
            if not screenshot_data:
                return None
            
            # 计算MD5哈希（快速）
            return hashlib.md5(screenshot_data).hexdigest()
        except Exception as e:
            self._debug_logger.warning(f"[SmartWaiter] 获取截图哈希失败: {e}")
            return None
    
    async def _detect_page_change(self, device_id: str, adb_bridge) -> bool:
        """检测页面是否变化（轻量级检测）
        
        Args:
            device_id: 设备ID
            adb_bridge: ADB桥接对象
            
        Returns:
            True表示页面已变化，False表示页面未变化
        """
        # 获取当前截图哈希
        current_hash = await self._get_screenshot_hash(device_id, adb_bridge)
        if not current_hash:
            return True  # 获取失败，假设页面已变化
        
        # 获取上一次的哈希
        last_hash = self._last_screenshot_hash.get(device_id)
        
        # 更新哈希
        self._last_screenshot_hash[device_id] = current_hash
        
        # 如果是第一次检测，假设页面已变化
        if last_hash is None:
            return True
        
        # 比较哈希值
        return current_hash != last_hash
    
    async def wait_for_page_change(
        self,
        device_id: str,
        detector,  # PageDetectorIntegrated
        expected_states: List,  # List[PageState]
        max_wait: float = 15.0,  # 超时保护（防止卡死）
        poll_interval: float = 0.1,  # 高频轮询
        log_callback=None,
        ignore_loading: bool = True,
        stability_check: bool = True,
        stability_count: int = 2,
        adb_bridge=None  # ADB桥接对象（可选）
    ) -> Optional[any]:
        """等待页面变化到期望状态（两阶段检测：轻量级变化检测 + 深度学习确认）
        
        工作流程：
        1. 高频轮询（0.1秒）检测页面是否变化（轻量级：图像哈希）
        2. 检测到变化 → 用深度学习确认页面类型
        3. 如果是期望状态 → 连续确认N次后立即返回
        4. 超时保护：15秒后强制返回（防止卡死）
        
        优化：
        - 轻量级检测：每次轮询都执行（快速，低开销）
        - 深度学习确认：只在检测到变化时执行（准确，高开销）
        - 大幅减少深度学习调用次数，提升性能
        
        支持的检测器：
        - PageDetectorIntegrated（整合检测器，GPU加速）
        
        Args:
            device_id: 设备ID
            detector: 页面检测器（混合检测器或整合检测器）
            expected_states: 期望的页面状态列表
            max_wait: 超时保护时间（秒，默认15秒）
            poll_interval: 轮询间隔（秒，默认0.1秒）
            log_callback: 日志回调函数
            ignore_loading: 是否忽略loading状态
            stability_check: 是否进行稳定性检测
            stability_count: 稳定性检测次数
            
        Returns:
            检测到的页面结果，超时返回None
        """
        from ..page_detector import PageState
        
        start_time = asyncio.get_event_loop().time()
        
        # 页面变化检测
        last_state = None
        last_confidence = 0.0
        same_state_count = 0
        
        # 统计信息
        total_checks = 0
        lightweight_checks = 0  # 轻量级检测次数
        dl_checks = 0  # 深度学习检测次数
        state_changes = 0
        
        # 检测器类型
        detector_type = type(detector).__name__
        
        # 详细日志：记录检测器类型（仅写入文件）
        self._debug_logger.debug(f"[SmartWaiter] 检测器类型: {detector_type}")
        
        # 获取ADB桥接对象
        if adb_bridge is None:
            # 尝试从detector获取ADB实例
            if hasattr(detector, 'adb'):
                adb_bridge = detector.adb
                self._debug_logger.info(f"[SmartWaiter] ✓ 从detector.adb获取ADB实例")
            elif hasattr(detector, '_adb'):
                adb_bridge = detector._adb
                self._debug_logger.info(f"[SmartWaiter] ✓ 从detector._adb获取ADB实例")
            else:
                # 如果detector没有ADB实例，禁用轻量级检测
                self._debug_logger.warning(f"[SmartWaiter] ⚠️ 无法获取ADB实例，禁用轻量级检测")
                adb_bridge = None
        else:
            self._debug_logger.info(f"[SmartWaiter] ✓ 使用传入的ADB实例")
        
        # 清除上一次的截图哈希
        if adb_bridge:
            self._last_screenshot_hash.pop(device_id, None)
        
        # 开始等待前，先清除缓存，确保第一次检测就是最新状态
        if detector_type == 'PageDetectorIntegrated' and hasattr(detector, '_detection_cache'):
            detector._detection_cache.clear(device_id)
            self._debug_logger.debug(f"[SmartWaiter] 已清除检测缓存")
        
        # 自适应轮询间隔
        current_poll_interval = poll_interval
        max_poll_interval = 0.3  # 最大轮询间隔（秒）
        
        # 强制深度学习检测间隔（即使页面未变化，也要定期检测）
        force_dl_check_interval = 1.0  # 每1秒强制检测一次
        last_dl_check_time = start_time
        
        while asyncio.get_event_loop().time() - start_time < max_wait:
            total_checks += 1
            current_time = asyncio.get_event_loop().time()
            
            # ============================================================
            # 第一阶段：轻量级检测页面是否变化（快速）
            # ============================================================
            if adb_bridge:
                lightweight_checks += 1
                page_changed = await self._detect_page_change(device_id, adb_bridge)
                
                # 调试日志：输出轻量级检测结果（每10次输出一次）
                if lightweight_checks % 10 == 1:
                    self._debug_logger.debug(
                        f"[SmartWaiter] 轻量级检测 #{lightweight_checks}: "
                        f"页面{'已变化' if page_changed else '未变化'}"
                    )
            else:
                # 没有ADB实例，跳过轻量级检测，直接深度学习
                page_changed = True
                if total_checks == 1:
                    self._debug_logger.info(f"[SmartWaiter] 无ADB实例，跳过轻量级检测")
            
            # 判断是否需要深度学习检测
            time_since_last_dl = current_time - last_dl_check_time
            force_dl_check = time_since_last_dl >= force_dl_check_interval
            
            if not page_changed and not force_dl_check:
                # 页面未变化且未到强制检测时间，继续轮询
                await asyncio.sleep(current_poll_interval)
                continue
            
            # 调试日志：输出深度学习检测原因
            if force_dl_check and not page_changed:
                self._debug_logger.debug(
                    f"[SmartWaiter] 强制深度学习检测（距上次{time_since_last_dl:.1f}秒）"
                )
            
            # ============================================================
            # 第二阶段：深度学习确认页面类型（准确）
            # ============================================================
            dl_checks += 1
            last_dl_check_time = current_time
            
            # 根据检测器类型调用不同的方法
            if detector_type == 'PageDetectorIntegrated':
                # 整合检测器（GPU加速，只检测页面类型，不检测元素）
                result = await detector.detect_page(device_id, use_cache=False, detect_elements=False)
            else:
                # 混合检测器（使用深度学习，最准确）
                result = await detector.detect_page(device_id, use_ocr=False, use_dl=True)
            
            if not result or not result.state:
                await asyncio.sleep(poll_interval)
                continue
            
            current_state = result.state
            current_confidence = result.confidence
            
            # 【调试】输出每次深度学习检测到的页面状态
            self._debug_logger.debug(
                f"[SmartWaiter] 深度学习检测 #{dl_checks}: {current_state.value} (置信度{current_confidence:.2%})"
            )
            
            # 检测到状态变化
            if current_state != last_state:
                state_changes += 1
                if log_callback and last_state is not None:
                    # 客户端：简洁信息
                    log_callback(f"页面变化: {last_state.value} → {current_state.value}")
                    
                    # 详细日志：包含耗时和置信度（仅写入文件）
                    elapsed = asyncio.get_event_loop().time() - start_time
                    self._debug_logger.debug(
                        f"[SmartWaiter] 页面变化: {last_state.value} → {current_state.value} "
                        f"(耗时{elapsed:.2f}秒, 置信度{current_confidence:.2%})"
                    )
                
                # 重置稳定性计数
                same_state_count = 1
                last_state = current_state
                last_confidence = current_confidence
                
                # 状态变化后，重置为快速轮询
                current_poll_interval = poll_interval
                
                # 清除缓存，确保下次检测到最新状态
                if detector_type == 'PageDetectorIntegrated' and hasattr(detector, '_detection_cache'):
                    detector._detection_cache.clear(device_id)
            else:
                # 状态相同，增加计数
                same_state_count += 1
                
                # 如果状态持续不变，逐渐增加轮询间隔（减少CPU占用）
                if same_state_count > 3:
                    current_poll_interval = min(current_poll_interval * 1.5, max_poll_interval)
            
            # 检查是否是期望状态
            if current_state in expected_states:
                # 如果启用稳定性检测
                if stability_check:
                    if same_state_count >= stability_count:
                        # 页面已稳定在期望状态，立即返回
                        elapsed = asyncio.get_event_loop().time() - start_time
                        if log_callback:
                            # 客户端：简洁信息
                            log_callback(f"✓ 页面稳定在 {current_state.value}，耗时{elapsed:.2f}秒")
                        
                        # 详细日志：包含统计信息（仅写入文件）
                        self._debug_logger.debug(
                            f"[SmartWaiter] ✓ 页面稳定在 {current_state.value} "
                            f"(连续{same_state_count}次, 置信度{current_confidence:.2%}, 耗时{elapsed:.2f}秒)"
                        )
                        self._debug_logger.debug(
                            f"[SmartWaiter] 统计: 总检测{total_checks}次, "
                            f"轻量级{lightweight_checks}次, 深度学习{dl_checks}次, 状态变化{state_changes}次"
                        )
                        return result
                    else:
                        # 还需要继续确认
                        if log_callback:
                            log_callback(f"确认中: {current_state.value}（{same_state_count}/{stability_count}次）")
                        
                        # 详细日志（仅写入文件）
                        self._debug_logger.debug(
                            f"[SmartWaiter] 确认中: {current_state.value} "
                            f"({same_state_count}/{stability_count}次, 置信度{current_confidence:.2%})"
                        )
                else:
                    # 不需要稳定性检测，直接返回
                    elapsed = asyncio.get_event_loop().time() - start_time
                    if log_callback:
                        log_callback(f"✓ 检测到 {current_state.value}，耗时{elapsed:.2f}秒")
                    
                    # 详细日志：包含统计信息（仅写入文件）
                    self._debug_logger.debug(
                        f"[SmartWaiter] 统计: 总检测{total_checks}次, "
                        f"轻量级{lightweight_checks}次, 深度学习{dl_checks}次, 状态变化{state_changes}次"
                    )
                    return result
            
            # 如果是loading状态且ignore_loading=True
            elif ignore_loading and current_state == PageState.LOADING:
                # loading是过渡状态，重置稳定性计数
                same_state_count = 0
                if log_callback and total_checks % 30 == 1:  # 每30次打印一次（约3秒）
                    elapsed = asyncio.get_event_loop().time() - start_time
                    # 客户端：简洁信息
                    log_callback(f"页面加载中...（{elapsed:.0f}秒）")
                
                # 详细日志：每次都记录（仅写入文件）
                if total_checks % 10 == 1:  # 每10次记录一次
                    elapsed = asyncio.get_event_loop().time() - start_time
                    self._debug_logger.debug(
                        f"[SmartWaiter] 页面加载中 (已等待{elapsed:.1f}秒, 检测{total_checks}次)"
                    )
            
            # 继续轮询
            await asyncio.sleep(current_poll_interval)
        
        # 超时保护触发
        elapsed = asyncio.get_event_loop().time() - start_time
        if log_callback:
            # 客户端：简洁信息
            log_callback(f"⚠️ 等待超时（{elapsed:.1f}秒）")
        
        # 详细日志：包含统计信息（仅写入文件）
        self._debug_logger.warning(
            f"[SmartWaiter] ⚠️ 超时保护触发 (耗时{elapsed:.1f}秒)"
        )
        self._debug_logger.warning(
            f"[SmartWaiter] 统计: 总检测{total_checks}次, "
            f"轻量级{lightweight_checks}次, 深度学习{dl_checks}次, 状态变化{state_changes}次"
        )
        if last_state:
            self._debug_logger.warning(
                f"[SmartWaiter] 最后状态: {last_state.value} (置信度{last_confidence:.2%})"
            )
        
        return None
    
    async def wait_for_condition(
        self,
        condition_func,
        max_wait: float = 5.0,
        poll_interval: float = 0.5,
        log_callback=None
    ) -> bool:
        """等待条件满足
        
        Args:
            condition_func: 条件函数，返回True表示条件满足
            max_wait: 最大等待时间（秒）
            poll_interval: 轮询间隔（秒）
            log_callback: 可选的日志回调函数
            
        Returns:
            条件是否在超时前满足
        """
        start_time = asyncio.get_event_loop().time()
        attempt = 0
        
        while asyncio.get_event_loop().time() - start_time < max_wait:
            attempt += 1
            
            # 检查条件
            if await condition_func() if asyncio.iscoroutinefunction(condition_func) else condition_func():
                # 条件满足，立即返回
                if log_callback:
                    elapsed = asyncio.get_event_loop().time() - start_time
                    log_callback(f"✓ 条件满足，耗时 {elapsed:.2f}秒（尝试{attempt}次）")
                return True
            
            # 条件未满足，等待后继续
            await asyncio.sleep(poll_interval)
        
        # 超时
        if log_callback:
            log_callback(f"⚠️ 等待超时（{max_wait}秒），条件未满足")
        return False


# ============================================================================
# 全局便捷函数
# ============================================================================

# 全局单例
_global_waiter = SmartWaiter()


async def wait_for_page(
    device_id: str,
    detector,
    expected_states: List,
    log_callback=None
) -> Optional[any]:
    """全局便捷函数：等待页面变化
    
    可以在项目任何地方直接调用，无需创建实例。
    内置最佳实践参数：
    - max_wait=15.0 (超时保护，不是等待时间)
    - poll_interval=0.1 (高频轮询，0.1秒/次)
    - stability_check=True (稳定性检测)
    - stability_count=1 (检测到即返回，不需要多次确认)
    - ignore_loading=True (忽略loading状态)
    
    实际耗时通常0.5-2秒，不是15秒！
    
    Args:
        device_id: 设备ID
        detector: 页面检测器
        expected_states: 期望的页面状态列表
        log_callback: 日志回调函数
        
    Returns:
        检测到的页面结果，超时返回None
        
    示例：
        from src.performance.smart_waiter import wait_for_page
        
        # 简单调用
        result = await wait_for_page(device_id, detector, [PageState.HOME])
        
        # 带日志
        result = await wait_for_page(
            device_id, detector, [PageState.PROFILE_LOGGED],
            log_callback=lambda msg: print(f"[等待] {msg}")
        )
    """
    return await _global_waiter.wait_for_page_change(
        device_id=device_id,
        detector=detector,
        expected_states=expected_states,
        max_wait=15.0,
        poll_interval=0.1,  # 恢复为0.1秒，快速检测页面变化
        log_callback=log_callback,
        ignore_loading=True,
        stability_check=True,
        stability_count=1  # 检测到即返回
    )
