"""
工作流程控制器 - 整合所有模块的完整流程控制
Workflow Controller - Complete workflow control integrating all modules
"""

import asyncio
from typing import Optional, Dict, Callable, Union
from datetime import datetime
from .adb_bridge import ADBBridge
from .page_detector_hybrid import PageDetectorHybrid
from .app_launcher import AppLauncher
from .profile_data_reader import LoginStatusChecker
from .navigator import Navigator
from .account_switcher import AccountSwitcher
from .account_cache import get_account_cache


class WorkflowController:
    """工作流程控制器
    
    负责整合所有模块，执行完整的工作流程：
    1. 启动应用到首页
    2. 检查登录状态
    3. 根据状态执行相应操作：
       - 未登录 → 登录 → 读取资料
       - 已登录且ID匹配 → 读取资料
       - 已登录但ID不匹配 → 切换账号 → 读取资料
    """
    
    def __init__(
        self,
        adb: ADBBridge,
        detector: Union['PageDetectorHybrid', 'PageDetectorIntegrated'],
        app_launcher: AppLauncher,
        login_checker: LoginStatusChecker,
        navigator: Navigator,
        account_switcher: AccountSwitcher,
        log_callback: Optional[Callable[[str], None]] = None
    ):
        """初始化工作流程控制器
        
        Args:
            adb: ADB桥接对象
            detector: 页面检测器（整合检测器或混合检测器）
            app_launcher: 应用启动器
            login_checker: 登录状态检查器
            navigator: 导航器
            account_switcher: 账号切换器
            log_callback: 日志回调函数（可选）
        """
        self.adb = adb
        self.detector = detector
        self.app_launcher = app_launcher
        self.login_checker = login_checker
        self.navigator = navigator
        self.account_switcher = account_switcher
        self.log_callback = log_callback
        self.cache = get_account_cache()
    
    def _log(self, message: str):
        """输出日志"""
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)
    
    async def run_complete_workflow(
        self,
        device_id: str,
        package_name: str,
        activity_name: str,
        phone: str,
        password: str
    ) -> Dict[str, any]:
        """执行完整的工作流程
        
        流程：
        1. 启动应用到首页
        2. 检查登录状态
        3. 根据状态执行操作：
           - 未登录 → 登录 → 读取资料
           - 已登录且ID匹配 → 读取资料
           - 已登录但ID不匹配 → 切换账号 → 读取资料
        
        Args:
            device_id: 设备ID
            package_name: 应用包名
            activity_name: 启动Activity
            phone: 手机号
            password: 密码
            
        Returns:
            dict: 工作流程结果
                - success: bool, 是否成功
                - message: str, 消息
                - data: dict, 数据（余额、积分等）
                - user_id: str, 用户ID
                - nickname: str, 昵称
        """
        result = {
            'success': False,
            'message': '',
            'data': {},
            'user_id': None,
            'nickname': None,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # 获取期望的用户ID
            expected_user_id = self.cache.get_user_id(phone)
            
            self._log("=" * 60)
            self._log("开始执行完整工作流程")
            self._log("=" * 60)
            self._log(f"账号: {phone}")
            if expected_user_id:
                self._log(f"期望用户ID: {expected_user_id}")
            
            # ===== 第一步：启动应用到首页 =====
            self._log("\n[第一步] 启动应用到首页...")
            
            launch_success = await self.app_launcher.launch_to_home(
                device_id,
                package_name,
                activity_name
            )
            
            if not launch_success:
                result['message'] = "启动应用失败"
                self._log(f"✗ {result['message']}")
                return result
            
            self._log("[第一步] ✓ 应用已启动到首页")
            
            # ===== 第二步：检查登录状态 =====
            self._log("\n[第二步] 检查登录状态...")
            
            status_result = await self.login_checker.check_login_status(
                device_id,
                phone,
                expected_user_id
            )
            
            if status_result['status'] == 'error':
                result['message'] = f"检查登录状态失败: {status_result['message']}"
                self._log(f"✗ {result['message']}")
                return result
            
            self._log(f"[第二步] 当前状态: {status_result['status']}")
            self._log(f"[第二步] {status_result['message']}")
            
            # ===== 第三步：根据状态执行操作 =====
            self._log("\n[第三步] 根据状态执行操作...")
            
            if status_result['status'] == 'not_logged_in':
                # 场景1：未登录 → 登录 → 读取资料
                self._log("[场景1] 未登录，执行登录...")
                
                login_result = await self.login_checker.login(
                    device_id,
                    phone,
                    password
                )
                
                if not login_result['success']:
                    result['message'] = f"登录失败: {login_result['message']}"
                    self._log(f"✗ {result['message']}")
                    return result
                
                self._log("✓ 登录成功")
                
                # 登录成功后读取资料
                profile_result = await self._read_profile_data(device_id, phone)
                
                if profile_result['success']:
                    result['success'] = True
                    result['message'] = "登录成功并读取资料"
                    result['data'] = profile_result['data']
                    result['user_id'] = profile_result['user_id']
                    result['nickname'] = profile_result['nickname']
                else:
                    result['message'] = f"登录成功但读取资料失败: {profile_result['message']}"
                    self._log(f"⚠️ {result['message']}")
            
            elif status_result['status'] == 'logged_in_matched':
                # 场景2：已登录且ID匹配 → 直接读取资料
                self._log("[场景2] 已登录且ID匹配，直接读取资料...")
                self._log(f"  昵称: {status_result['current_nickname']}")
                self._log(f"  用户ID: {status_result['current_user_id']}")
                
                # 直接读取资料
                profile_result = await self._read_profile_data(device_id, phone)
                
                if profile_result['success']:
                    result['success'] = True
                    result['message'] = "ID匹配，读取资料成功"
                    result['data'] = profile_result['data']
                    result['user_id'] = status_result['current_user_id']
                    result['nickname'] = status_result['current_nickname']
                else:
                    result['message'] = f"读取资料失败: {profile_result['message']}"
                    self._log(f"✗ {result['message']}")
            
            elif status_result['status'] == 'logged_in_mismatched':
                # 场景3：已登录但ID不匹配 → 切换账号 → 读取资料
                self._log("[场景3] 已登录但ID不匹配，执行账号切换...")
                self._log(f"  当前用户ID: {status_result['current_user_id']}")
                self._log(f"  期望用户ID: {expected_user_id}")
                
                # 执行账号切换
                switch_success, switch_message = await self.account_switcher.switch_account(
                    device_id,
                    package_name,
                    activity_name,
                    phone,
                    password,
                    current_user_id=status_result['current_user_id'],
                    expected_user_id=expected_user_id
                )
                
                if not switch_success:
                    result['message'] = f"账号切换失败: {switch_message}"
                    self._log(f"✗ {result['message']}")
                    return result
                
                self._log("✓ 账号切换成功")
                
                # 切换成功后读取资料
                profile_result = await self._read_profile_data(device_id, phone)
                
                if profile_result['success']:
                    result['success'] = True
                    result['message'] = "账号切换成功并读取资料"
                    result['data'] = profile_result['data']
                    result['user_id'] = profile_result['user_id']
                    result['nickname'] = profile_result['nickname']
                else:
                    result['message'] = f"账号切换成功但读取资料失败: {profile_result['message']}"
                    self._log(f"⚠️ {result['message']}")
            
            else:
                result['message'] = f"未知状态: {status_result['status']}"
                self._log(f"✗ {result['message']}")
            
            # ===== 完成 =====
            if result['success']:
                self._log("\n" + "=" * 60)
                self._log("工作流程完成")
                self._log("=" * 60)
                self._log(f"✓ {result['message']}")
                self._log(f"\n账号信息:")
                self._log(f"  手机号: {phone}")
                if result['nickname']:
                    self._log(f"  昵称: {result['nickname']}")
                if result['user_id']:
                    self._log(f"  用户ID: {result['user_id']}")
                
                if result['data']:
                    self._log(f"\n账户数据:")
                    if result['data'].get('balance') is not None:
                        self._log(f"  余额: {result['data']['balance']:.2f} 元")
                    if result['data'].get('points') is not None:
                        self._log(f"  积分: {result['data']['points']} 积分")
                    if result['data'].get('vouchers') is not None:
                        self._log(f"  抵扣券: {result['data']['vouchers']} 张")
                    if result['data'].get('coupons') is not None:
                        self._log(f"  优惠券: {result['data']['coupons']} 张")
            
            return result
            
        except Exception as e:
            result['message'] = f"工作流程异常: {str(e)}"
            self._log(f"\n❌ {result['message']}")
            import traceback
            traceback.print_exc()
            return result
    
    async def _read_profile_data(
        self,
        device_id: str,
        phone: str
    ) -> Dict[str, any]:
        """读取个人资料（内部方法）
        
        Args:
            device_id: 设备ID
            phone: 手机号
            
        Returns:
            dict: 读取结果
                - success: bool, 是否成功
                - message: str, 消息
                - data: dict, 数据
                - user_id: str, 用户ID
                - nickname: str, 昵称
        """
        result = {
            'success': False,
            'message': '',
            'data': {},
            'user_id': None,
            'nickname': None
        }
        
        try:
            self._log("\n读取个人资料...")
            
            # 读取动态数据（余额、积分、抵扣券、优惠券）
            balance_result = await self.login_checker.read_balance_data(device_id)
            
            if not balance_result['success']:
                result['message'] = balance_result['message']
                self._log(f"✗ 读取失败: {result['message']}")
                return result
            
            # 从缓存获取昵称和用户ID
            nickname = self.cache.get_nickname(phone)
            user_id = self.cache.get_user_id(phone)
            
            result['success'] = True
            result['message'] = "读取成功"
            result['data'] = {
                'balance': balance_result.get('balance'),
                'points': balance_result.get('points'),
                'vouchers': balance_result.get('vouchers'),
                'coupons': balance_result.get('coupons')
            }
            result['user_id'] = user_id
            result['nickname'] = nickname
            
            self._log("✓ 资料读取成功")
            
            return result
            
        except Exception as e:
            result['message'] = f"读取异常: {str(e)}"
            self._log(f"✗ {result['message']}")
            return result
