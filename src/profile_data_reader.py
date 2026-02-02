"""
登录状态检查模块 - 从首页导航到个人页并检查登录状态
Login Status Checker Module - Navigate to profile page and check login status
"""

import asyncio
from typing import Optional, Dict, Union
from .adb_bridge import ADBBridge
from .page_detector_hybrid import PageDetectorHybrid, PageState
from .navigator import Navigator
from .profile_reader import ProfileReader
from .login_handler import LoginHandler
from .performance.smart_waiter import wait_for_page


class LoginStatusChecker:
    """登录状态检查器
    
    负责：
    1. 从首页导航到个人页
    2. 检查登录状态（未登录、已登录ID匹配、已登录ID不匹配）
    3. 停留在个人页
    """
    
    def __init__(self, adb: ADBBridge, detector: Union['PageDetectorHybrid', 'PageDetectorIntegrated'], navigator: Navigator):
        """初始化登录状态检查器
        
        Args:
            adb: ADB桥接对象
            detector: 页面检测器
            navigator: 导航器
        """
        self.adb = adb
        self.detector = detector
        self.navigator = navigator
        self.profile_reader = ProfileReader(adb, detector)
        self.login_handler = LoginHandler(adb, detector, self.profile_reader)
    
    async def check_login_status(
        self,
        device_id: str,
        expected_phone: str,
        expected_user_id: Optional[str] = None
    ) -> Dict[str, any]:
        """检查登录状态（只检查登录状态和ID匹配，不读取余额等动态数据）
        
        从首页导航到个人页，检查登录状态并停留在个人页
        
        重要：此方法只检查登录状态和ID匹配，不读取余额、积分等动态数据
        如果需要读取动态数据，请单独调用 read_balance_data() 方法
        
        注意：个人页上没有显示手机号，只能通过用户ID进行匹配
        
        Args:
            device_id: 设备ID
            expected_phone: 期望的手机号（用于缓存查询，获取期望的用户ID）
            expected_user_id: 期望的用户ID（用于ID匹配检查，如果为None则从缓存获取）
            
        Returns:
            dict: 登录状态
                - status: str, 登录状态
                    - "not_logged_in": 未登录
                    - "logged_in_matched": 已登录且ID匹配
                    - "logged_in_mismatched": 已登录但ID不匹配
                - message: str, 状态消息
                - current_user_id: str, 当前登录的用户ID（如果已登录）
                - current_nickname: str, 当前登录的昵称（如果已登录）
        """
        result = {
            'status': 'unknown',
            'message': '',
            'current_user_id': None,
            'current_nickname': None
        }
        
        try:
            # 如果没有提供expected_user_id，尝试从缓存获取
            if not expected_user_id:
                from .account_cache import get_account_cache
                cache = get_account_cache()
                expected_user_id = cache.get_user_id(expected_phone)
                if expected_user_id:
                    print(f"[登录检查] 从缓存获取期望用户ID: {expected_user_id}")
            
            # 1. 确保在首页
            print(f"[登录检查] 确保在首页...")
            current_page = await self.detector.detect_page(device_id, use_ocr=False, use_dl=True)
            
            if not current_page or current_page.state != PageState.HOME:
                print(f"[登录检查] 当前不在首页: {current_page.state.value if current_page else 'Unknown'}")
                # 尝试导航到首页
                nav_success = await self.navigator.navigate_to_home(device_id)
                if not nav_success:
                    result['status'] = 'error'
                    result['message'] = "无法导航到首页"
                    return result
            
            print(f"[登录检查] ✓ 已在首页")
            
            # 2. 导航到个人页
            print(f"[登录检查] 导航到个人页...")
            nav_success = await self.navigator.navigate_to_profile(device_id)
            
            if not nav_success:
                result['status'] = 'error'
                result['message'] = "无法导航到个人页"
                return result
            
            print(f"[登录检查] ✓ 已到达个人页")
            
            # 3. 检测当前页面状态
            print(f"[登录检查] 检测登录状态...")
            page_result = await self.detector.detect_page(device_id, use_ocr=False, use_dl=True)
            
            if not page_result:
                result['status'] = 'error'
                result['message'] = "无法检测页面状态"
                return result
            
            current_state = page_result.state
            print(f"[登录检查] 当前页面状态: {current_state.value}")
            
            # 4. 判断登录状态
            if current_state == PageState.PROFILE:
                # 未登录个人页
                result['status'] = 'not_logged_in'
                result['message'] = "未登录"
                print(f"[登录检查] ✗ 未登录")
                return result
            
            elif current_state == PageState.PROFILE_LOGGED:
                # 已登录个人页，需要读取用户信息进行ID匹配
                print(f"[登录检查] 已登录，读取用户信息...")
                
                # 只读取身份信息（昵称、用户ID），不读取余额等动态数据
                # 注意：个人页上没有显示手机号，只能通过用户ID进行匹配
                identity_data = await self.profile_reader.get_identity_only(device_id)
                
                current_nickname = identity_data.get('nickname')
                current_user_id = identity_data.get('user_id')
                
                result['current_nickname'] = current_nickname
                result['current_user_id'] = current_user_id
                
                print(f"[登录检查] 当前账号信息:")
                if current_nickname:
                    print(f"  - 昵称: {current_nickname}")
                if current_user_id:
                    print(f"  - 用户ID: {current_user_id}")
                
                # 检查用户ID是否匹配
                if expected_user_id:
                    if current_user_id and current_user_id == expected_user_id:
                        result['status'] = 'logged_in_matched'
                        result['message'] = f"已登录且ID匹配（用户ID: {current_user_id}）"
                        print(f"[登录检查] ✓ 已登录且用户ID匹配")
                    else:
                        result['status'] = 'logged_in_mismatched'
                        result['message'] = f"已登录但ID不匹配（期望: {expected_user_id}, 实际: {current_user_id or '未知'}）"
                        print(f"[登录检查] ✗ 已登录但用户ID不匹配")
                        print(f"  期望用户ID: {expected_user_id}")
                        print(f"  实际用户ID: {current_user_id or '未知'}")
                else:
                    # 没有期望的用户ID，无法判断是否匹配
                    if current_user_id:
                        result['status'] = 'logged_in_matched'
                        result['message'] = f"已登录（用户ID: {current_user_id}，无期望ID可比对）"
                        print(f"[登录检查] ✓ 已登录（无期望用户ID可比对）")
                    else:
                        result['status'] = 'logged_in_matched'
                        result['message'] = "已登录（无法获取用户ID）"
                        print(f"[登录检查] ⚠️ 已登录但无法获取用户ID")
                
                return result
            
            else:
                # 其他未知状态
                result['status'] = 'error'
                result['message'] = f"未知页面状态: {current_state.value}"
                print(f"[登录检查] ❌ 未知页面状态: {current_state.value}")
                return result
            
        except Exception as e:
            result['status'] = 'error'
            result['message'] = f"检查失败: {str(e)}"
            print(f"[登录检查] ❌ 检查失败: {e}")
            return result
    
    async def read_balance_data(
        self,
        device_id: str
    ) -> Dict[str, any]:
        """只读取余额、积分、抵扣券、优惠券等动态数据
        
        前提：已经在个人页（已登录状态）
        
        Args:
            device_id: 设备ID
            
        Returns:
            dict: 动态数据
                - success: bool, 是否成功
                - message: str, 消息
                - balance: float, 余额（可能为None）
                - points: int, 积分（可能为None）
                - vouchers: float, 抵扣券（可能为None）
                - coupons: int, 优惠券（可能为None）
        """
        result = {
            'success': False,
            'message': '',
            'balance': None,
            'points': None,
            'vouchers': None,
            'coupons': None
        }
        
        try:
            print(f"[余额数据] 读取动态数据...")
            
            # 调用ProfileReader的_get_dynamic_data_only方法
            # 这个方法只读取余额、积分、抵扣券、优惠券，不读取昵称和用户ID
            dynamic_data = await self.profile_reader._get_dynamic_data_only(device_id)
            
            result['balance'] = dynamic_data.get('balance')
            result['points'] = dynamic_data.get('points')
            result['vouchers'] = dynamic_data.get('vouchers')
            result['coupons'] = dynamic_data.get('coupons')
            
            # 至少要能获取到余额，才认为成功
            if result['balance'] is not None:
                result['success'] = True
                result['message'] = "读取成功"
                print(f"[余额数据] ✓ 读取成功")
            else:
                result['message'] = "无法获取余额信息"
                print(f"[余额数据] ⚠️ 无法获取余额信息")
            
            return result
            
        except Exception as e:
            result['message'] = f"读取失败: {str(e)}"
            print(f"[余额数据] ❌ 读取失败: {e}")
            return result
    
    async def read_profile_data(
        self,
        device_id: str
    ) -> Dict[str, any]:
        """读取个人资料（保留原有功能）
        
        从首页导航到个人页，读取资料后停留在个人页
        
        Args:
            device_id: 设备ID
            
        Returns:
            dict: 个人资料
                - success: bool, 是否成功
                - message: str, 消息
                - balance: float, 余额（可能为None）
                - points: int, 积分（可能为None）
                - vouchers: int, 抵扣券（可能为None）
        """
        result = {
            'success': False,
            'message': '',
            'balance': None,
            'points': None,
            'vouchers': None
        }
        
        try:
            # 1. 确保在首页
            print(f"[资料] 确保在首页...")
            current_page = await self.detector.detect_page(device_id, use_ocr=False, use_dl=True)
            
            if not current_page or current_page.state != PageState.HOME:
                print(f"[资料] 当前不在首页: {current_page.state.value if current_page else 'Unknown'}")
                # 尝试导航到首页
                nav_success = await self.navigator.navigate_to_home(device_id)
                if not nav_success:
                    result['message'] = "无法导航到首页"
                    return result
            
            print(f"[资料] ✓ 已在首页")
            
            # 2. 导航到个人页
            print(f"[资料] 导航到个人页...")
            nav_success = await self.navigator.navigate_to_profile(device_id)
            
            if not nav_success:
                result['message'] = "无法导航到个人页"
                return result
            
            print(f"[资料] ✓ 已到达个人页")
            
            # 3. 读取个人信息
            print(f"[资料] 读取个人信息...")
            profile_data = await self.profile_reader.get_full_profile(device_id)
            
            if not profile_data:
                result['message'] = "无法读取个人信息"
                return result
            
            # 4. 提取数据
            balance = profile_data.get('balance')
            points = profile_data.get('points')
            vouchers = profile_data.get('vouchers')
            
            # 至少要能获取到余额，才认为成功
            if balance is not None:
                result['success'] = True
                result['balance'] = balance
                result['points'] = points
                result['vouchers'] = vouchers
                
                print(f"[资料] ✓ 个人信息:")
                print(f"  - 余额: {balance:.2f} 元")
                
                if points is not None:
                    print(f"  - 积分: {points} 积分")
                else:
                    print(f"  - 积分: 无法获取")
                
                if vouchers is not None:
                    print(f"  - 抵扣券: {vouchers} 张")
                else:
                    print(f"  - 抵扣券: 无法获取")
                
                result['message'] = "读取成功"
            else:
                result['message'] = "无法获取余额信息"
                print(f"[资料] ❌ 无法获取余额信息")
            
            # 5. 停留在个人页（不返回首页）
            print(f"[资料] ✓ 读取完成，停留在个人页")
            
            return result
            
        except Exception as e:
            result['message'] = f"读取失败: {str(e)}"
            print(f"[资料] ❌ 读取失败: {e}")
            return result
    
    async def get_balance_only(self, device_id: str) -> Optional[float]:
        """只读取余额（快速方法）
        
        注意：读取后停留在个人页
        
        Args:
            device_id: 设备ID
            
        Returns:
            float: 余额，失败返回None
        """
        result = await self.read_profile_data(device_id)
        return result.get('balance')
    
    async def get_points_only(self, device_id: str) -> Optional[int]:
        """只读取积分（快速方法）
        
        注意：读取后停留在个人页
        
        Args:
            device_id: 设备ID
            
        Returns:
            int: 积分，失败返回None
        """
        result = await self.read_profile_data(device_id)
        return result.get('points')
    
    async def login(
        self,
        device_id: str,
        phone: str,
        password: str
    ) -> Dict[str, any]:
        """执行登录（从个人未登录页到个人已登录页）
        
        前提：当前在个人未登录页
        
        Args:
            device_id: 设备ID
            phone: 手机号
            password: 密码
            
        Returns:
            dict: 登录结果
                - success: bool, 是否成功
                - message: str, 消息
        """
        result = {
            'success': False,
            'message': ''
        }
        
        try:
            print(f"[登录] 执行登录...")
            success, message = await self.login_handler.login_from_profile(
                device_id,
                phone,
                password
            )
            
            result['success'] = success
            result['message'] = message
            
            if success:
                print(f"[登录] ✓ {message}")
            else:
                print(f"[登录] ✗ {message}")
            
            return result
            
        except Exception as e:
            result['message'] = f"登录失败: {str(e)}"
            print(f"[登录] ❌ 登录失败: {e}")
            return result
    
    async def logout(
        self,
        device_id: str
    ) -> Dict[str, any]:
        """执行登出（从个人已登录页到个人未登录页）
        
        前提：当前在个人已登录页
        
        Args:
            device_id: 设备ID
            
        Returns:
            dict: 登出结果
                - success: bool, 是否成功
                - message: str, 消息
        """
        result = {
            'success': False,
            'message': ''
        }
        
        try:
            print(f"[登出] 执行登出...")
            success, message = await self.login_handler.logout_from_profile(device_id)
            
            result['success'] = success
            result['message'] = message
            
            if success:
                print(f"[登出] ✓ {message}")
            else:
                print(f"[登出] ✗ {message}")
            
            return result
            
        except Exception as e:
            result['message'] = f"登出失败: {str(e)}"
            print(f"[登出] ❌ 登出失败: {e}")
            return result
