"""
登录处理模块 - 从个人未登录页到登录成功回到个人页
Login Handler Module - From profile page (not logged in) to profile page (logged in)
"""

import asyncio
from typing import Optional, Dict, Tuple, Union
from .adb_bridge import ADBBridge
from .page_detector_hybrid import PageDetectorHybrid, PageState
from .performance.smart_waiter import wait_for_page
from .account_cache import get_account_cache


class LoginHandler:
    """登录处理器
    
    负责：
    1. 从个人未登录页点击登录按钮
    2. 输入账号密码
    3. 点击登录
    4. 等待登录成功
    5. 读取用户ID和昵称
    6. 保存/更新缓存
    7. 返回个人页（已登录状态）
    """
    
    # 登录页坐标（540x960分辨率）
    COORDS = {
        'LOGIN_BUTTON_PROFILE': (270, 200),      # 个人页的登录按钮
        'PHONE_INPUT': (270, 466),               # 手机号输入框
        'PASSWORD_INPUT': (270, 532),            # 密码输入框
        'AGREEMENT_CHECK': (89, 591),            # 协议勾选框
        'LOGIN_BUTTON': (270, 687),              # 登录按钮
        'ERROR_CONFIRM': (430, 588),             # 错误确认按钮
    }
    
    def __init__(self, adb: ADBBridge, detector: Union['PageDetectorHybrid', 'PageDetectorIntegrated'], profile_reader=None):
        """初始化登录处理器
        
        Args:
            adb: ADB桥接对象
            detector: 页面检测器
            profile_reader: ProfileReader对象（用于读取用户信息）
        """
        self.adb = adb
        self.detector = detector
        self.profile_reader = profile_reader
        self.cache = get_account_cache()
    
    async def login_from_profile(
        self,
        device_id: str,
        phone: str,
        password: str,
        max_wait: float = 30.0
    ) -> Tuple[bool, str]:
        """从个人未登录页执行登录
        
        Args:
            device_id: 设备ID
            phone: 手机号
            password: 密码
            max_wait: 最大等待时间（秒）
            
        Returns:
            tuple: (是否成功, 消息)
        """
        print(f"[登录] 开始登录流程...")
        
        try:
            # 1. 确认当前在个人未登录页
            print(f"[登录] 检测当前页面...")
            page_result = await self.detector.detect_page(device_id, use_ocr=False, use_dl=True)
            
            if not page_result or page_result.state != PageState.PROFILE:
                return False, f"当前不在个人未登录页: {page_result.state.value if page_result else 'Unknown'}"
            
            print(f"[登录] ✓ 已在个人未登录页")
            
            # 2. 点击登录按钮
            print(f"[登录] 点击登录按钮...")
            await self.adb.tap(device_id, *self.COORDS['LOGIN_BUTTON_PROFILE'])
            
            # 智能等待到达登录页
            await wait_for_page(
                device_id,
                self.detector,
                [PageState.LOGIN],
                log_callback=lambda msg: print(f"  [智能等待] {msg}")
            )
            
            # 确认到达登录页
            page_result = await self.detector.detect_page(device_id, use_ocr=False, use_dl=True)
            if not page_result or page_result.state != PageState.LOGIN:
                return False, "无法到达登录页"
            
            print(f"[登录] ✓ 已到达登录页")
            
            # 3. 输入手机号
            print(f"[登录] 输入手机号...")
            await self.adb.tap(device_id, *self.COORDS['PHONE_INPUT'])
            await asyncio.sleep(0.5)
            
            # 清空输入框
            for _ in range(15):
                await self.adb.press_key(device_id, 'KEYCODE_DEL')
                await asyncio.sleep(0.05)
            
            # 输入手机号
            await self.adb.input_text(device_id, phone)
            await asyncio.sleep(0.5)
            print(f"[登录] ✓ 手机号已输入")
            
            # 4. 输入密码
            print(f"[登录] 输入密码...")
            await self.adb.tap(device_id, *self.COORDS['PASSWORD_INPUT'])
            await asyncio.sleep(0.5)
            
            # 清空输入框
            for _ in range(20):
                await self.adb.press_key(device_id, 'KEYCODE_DEL')
                await asyncio.sleep(0.05)
            
            # 输入密码
            await self.adb.input_text(device_id, password)
            await asyncio.sleep(0.5)
            print(f"[登录] ✓ 密码已输入")
            
            # 5. 勾选协议（如果需要）
            print(f"[登录] 勾选协议...")
            await self.adb.tap(device_id, *self.COORDS['AGREEMENT_CHECK'])
            await asyncio.sleep(0.3)
            print(f"[登录] ✓ 协议已勾选")
            
            # 6. 点击登录按钮
            print(f"[登录] 点击登录按钮...")
            await self.adb.tap(device_id, *self.COORDS['LOGIN_BUTTON'])
            
            # 7. 智能等待登录结果
            print(f"[登录] 等待登录结果...")
            start_time = asyncio.get_event_loop().time()
            
            while asyncio.get_event_loop().time() - start_time < max_wait:
                await asyncio.sleep(1)
                
                # 检测当前页面
                page_result = await self.detector.detect_page(device_id, use_ocr=False, use_dl=True)
                
                if not page_result:
                    continue
                
                current_state = page_result.state
                print(f"[登录] 当前页面: {current_state.value}")
                
                # 登录成功，到达个人已登录页
                if current_state == PageState.PROFILE_LOGGED:
                    elapsed = asyncio.get_event_loop().time() - start_time
                    print(f"[登录] ✓ 登录成功，耗时 {elapsed:.2f}秒")
                    
                    # 读取用户ID和昵称
                    if self.profile_reader:
                        print(f"[登录] 读取用户信息...")
                        try:
                            identity_data = await self.profile_reader.get_identity_only(device_id)
                            
                            nickname = identity_data.get('nickname')
                            user_id = identity_data.get('user_id')
                            
                            if nickname:
                                print(f"[登录] ✓ 昵称: {nickname}")
                            if user_id:
                                print(f"[登录] ✓ 用户ID: {user_id}")
                            
                            # 保存/更新缓存
                            if nickname or user_id:
                                print(f"[登录] 更新缓存...")
                                if nickname:
                                    self.cache.set(phone, nickname=nickname)
                                    print(f"[登录] ✓ 昵称已保存到缓存")
                                if user_id:
                                    self.cache.set(phone, user_id=user_id)
                                    print(f"[登录] ✓ 用户ID已保存到缓存")
                            else:
                                print(f"[登录] ⚠️ 无法读取用户信息")
                        except Exception as e:
                            print(f"[登录] ⚠️ 读取用户信息失败: {e}")
                    
                    return True, "登录成功"
                
                # 登录失败，仍在登录页
                if current_state == PageState.LOGIN:
                    # 可能有错误提示，尝试点击确认按钮
                    print(f"[登录] ⚠️ 仍在登录页，可能登录失败")
                    await self.adb.tap(device_id, *self.COORDS['ERROR_CONFIRM'])
                    await asyncio.sleep(1)
                    return False, "登录失败（账号或密码错误）"
                
                # 其他页面，继续等待
                await asyncio.sleep(1)
            
            # 超时
            elapsed = asyncio.get_event_loop().time() - start_time
            return False, f"登录超时（{elapsed:.1f}秒）"
            
        except Exception as e:
            return False, f"登录失败: {str(e)}"
    
    async def logout_from_profile(
        self,
        device_id: str
    ) -> Tuple[bool, str]:
        """从个人已登录页执行登出
        
        Args:
            device_id: 设备ID
            
        Returns:
            tuple: (是否成功, 消息)
        """
        print(f"[登出] 开始登出流程...")
        
        try:
            # 1. 确认当前在个人已登录页
            print(f"[登出] 检测当前页面...")
            page_result = await self.detector.detect_page(device_id, use_ocr=False, use_dl=True)
            
            if not page_result or page_result.state != PageState.PROFILE_LOGGED:
                return False, f"当前不在个人已登录页: {page_result.state.value if page_result else 'Unknown'}"
            
            print(f"[登出] ✓ 已在个人已登录页")
            
            # 2. 点击设置按钮（右上角）
            print(f"[登出] 点击设置按钮...")
            await self.adb.tap(device_id, 500, 50)  # 设置按钮坐标
            await asyncio.sleep(1)
            
            # 3. 检测是否到达设置页
            page_result = await self.detector.detect_page(device_id, use_ocr=False, use_dl=True)
            
            if not page_result or page_result.state != PageState.SETTINGS:
                return False, "无法到达设置页"
            
            print(f"[登出] ✓ 已到达设置页")
            
            # 4. 滚动到底部找到退出登录按钮
            print(f"[登出] 滚动到底部...")
            for _ in range(3):
                await self.adb.swipe(device_id, 270, 800, 270, 200, 300)
                await asyncio.sleep(0.5)
            
            # 5. 点击退出登录按钮
            print(f"[登出] 点击退出登录...")
            await self.adb.tap(device_id, 270, 850)  # 退出登录按钮坐标
            await asyncio.sleep(1)
            
            # 6. 确认退出
            print(f"[登出] 确认退出...")
            await self.adb.tap(device_id, 350, 550)  # 确认按钮坐标
            
            # 7. 智能等待返回个人未登录页
            await wait_for_page(
                device_id,
                self.detector,
                [PageState.PROFILE],
                log_callback=lambda msg: print(f"  [智能等待] {msg}")
            )
            
            # 确认到达个人未登录页
            page_result = await self.detector.detect_page(device_id, use_ocr=False, use_dl=True)
            if page_result and page_result.state == PageState.PROFILE:
                print(f"[登出] ✓ 登出成功")
                return True, "登出成功"
            else:
                return False, "登出失败"
            
        except Exception as e:
            return False, f"登出失败: {str(e)}"
