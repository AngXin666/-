"""
卡密激活管理器
License Activation Manager
"""

import hashlib
import json
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple
import platform
import threading
import time

try:
    from .local_db import LocalDatabase
    from .crypto_utils import crypto
except ImportError:
    from local_db import LocalDatabase
    from crypto_utils import crypto


class LicenseManager:
    """卡密管理器（使用本地数据库）"""
    
    def __init__(self):
        # 本地数据库
        self.db = LocalDatabase()
        
        # 旧的许可证文件（用于迁移）
        self.license_file = Path("runtime_data") / "license.dat"
        
        # 后台验证线程（按需启动，不定时）
        self._validation_thread: Optional[threading.Thread] = None
        self._stop_validation = threading.Event()
        self._last_check_time: Optional[datetime] = None
        
        # 过期后的重新认证线程
        self._reauth_thread: Optional[threading.Thread] = None
        self._stop_reauth = threading.Event()
        
        # 迁移旧数据
        self._migrate_old_data()
        
    def _migrate_old_data(self):
        """迁移旧的许可证文件数据到数据库"""
        if self.license_file.exists() and not self.db.is_activated():
            try:
                with open(self.license_file, 'r', encoding='utf-8') as f:
                    encrypted_data = json.load(f)
                old_data = self._decrypt_data(encrypted_data)
                
                # 保存到数据库
                self.db.save_license(old_data)
                print("[迁移] 已将旧许可证数据迁移到数据库")
                
                # 删除旧文件
                self.license_file.unlink()
            except Exception as e:
                print(f"[迁移] 迁移失败: {e}")
    
    def get_machine_id(self) -> str:
        """获取机器唯一ID（基于硬件信息）"""
        try:
            # 获取多个硬件信息组合生成唯一ID
            machine_info = []
            
            # CPU信息
            try:
                import cpuinfo
                cpu_info = cpuinfo.get_cpu_info()
                machine_info.append(cpu_info.get('brand_raw', ''))
            except:
                pass
            
            # MAC地址
            mac = ':'.join(['{:02x}'.format((uuid.getnode() >> elements) & 0xff) 
                           for elements in range(0,2*6,2)][::-1])
            machine_info.append(mac)
            
            # 系统信息
            machine_info.append(platform.node())
            machine_info.append(platform.machine())
            
            # 生成唯一ID
            combined = ''.join(machine_info)
            machine_id = hashlib.sha256(combined.encode()).hexdigest()
            
            return machine_id
            
        except Exception as e:
            print(f"获取机器ID失败: {e}")
            # 备用方案：使用UUID
            return str(uuid.uuid4())
    
    def validate_license_key(self, license_key: str) -> Tuple[bool, str]:
        """验证卡密格式
        
        Args:
            license_key: 卡密字符串
            
        Returns:
            (是否有效, 错误信息)
        """
        # 移除空格和横线
        key = license_key.replace(' ', '').replace('-', '').upper()
        
        # 检查格式：5组4位字符（去掉横线后20位）
        if len(key) != 20:
            return False, "卡密格式错误：长度不正确"
        
        # 检查是否全是字母数字
        if not key.isalnum():
            return False, "卡密格式错误：包含非法字符"
        
        return True, ""
    
    def activate_license(self, license_key: str, machine_id: str) -> Tuple[bool, str]:
        """激活卡密（首次必须在线验证）
        
        Args:
            license_key: 卡密
            machine_id: 机器ID
            
        Returns:
            (是否成功, 消息)
        """
        print(f"[激活] 收到参数 - 卡密: '{license_key}', 机器ID: '{machine_id}'")
        
        # 验证格式
        valid, error_msg = self.validate_license_key(license_key)
        if not valid:
            return False, error_msg
        
        # 标准化卡密格式（保留横线用于在线验证）
        key = license_key.replace(' ', '').upper()
        if '-' not in key and len(key) == 20:
            # 如果没有横线，添加标准格式的横线（XMZD-XXXX-XXXX-XXXX-XXXX）
            key = f"{key[:4]}-{key[4:8]}-{key[8:12]}-{key[12:16]}-{key[16:20]}"
        
        print(f"[激活] 标准化后的卡密: '{key}'")
        
        # 获取API地址
        api_url = self._get_api_url()
        if not api_url:
            return False, "网络连接失败：无法连接到激活服务器，请检查网络连接"
        
        print(f"[激活] API地址: {api_url}")
        
        # 首次激活必须在线验证
        online_success, online_msg = self._activate_online(key, machine_id, api_url)
        
        # 如果是卡密错误，直接返回错误信息
        if not online_success and "卡密错误" in online_msg:
            return False, online_msg
        
        # 如果是网络或服务器问题，也直接返回
        if not online_success:
            return False, online_msg
        
        return True, online_msg
    
    def _get_supabase_config(self) -> Tuple[str, str]:
        """获取 Supabase 配置
        
        Returns:
            (supabase_url, supabase_key)
        """
        import os
        
        # 首先尝试从环境变量读取
        supabase_url = os.environ.get('SUPABASE_URL', '')
        supabase_key = os.environ.get('SUPABASE_KEY', '')
        
        # 如果环境变量没有，尝试从 .env 文件读取
        if not supabase_url or not supabase_key:
            # 尝试多个可能的位置
            env_paths = [
                Path(".env"),  # 根目录
                Path("server") / ".env",  # server 目录
                Path("..") / ".env",  # 上级目录（如果在 src 中运行）
            ]
            
            for env_file in env_paths:
                if env_file.exists():
                    try:
                        with open(env_file, 'r', encoding='utf-8') as f:
                            for line in f:
                                line = line.strip()
                                if line.startswith('SUPABASE_URL=') and not supabase_url:
                                    supabase_url = line.split('=', 1)[1].strip()
                                elif line.startswith('SUPABASE_KEY=') and not supabase_key:
                                    supabase_key = line.split('=', 1)[1].strip()
                        if supabase_url and supabase_key:
                            break  # 找到配置就停止
                    except Exception as e:
                        print(f"读取 {env_file} 失败: {e}")
                        continue
        
        return supabase_url, supabase_key
    
    def _get_api_url(self) -> Optional[str]:
        """获取API地址（从配置文件或环境变量）"""
        # 尝试从环境变量读取
        import os
        api_url = os.environ.get('LICENSE_API_URL', '')
        
        if api_url:
            return api_url
        
        # 尝试从 .env 文件读取
        try:
            env_file = Path(".env")
            if env_file.exists():
                with open(env_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('LICENSE_API_URL='):
                            url = line.split('=', 1)[1].strip()
                            if url and not url.startswith('#'):
                                return url
        except Exception as e:
            print(f"读取 .env 失败: {e}")
        
        # 尝试从 server/.env 文件读取
        try:
            env_file = Path("server") / ".env"
            if env_file.exists():
                with open(env_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('LICENSE_API_URL='):
                            url = line.split('=', 1)[1].strip()
                            if url and not url.startswith('#'):
                                return url
        except Exception as e:
            print(f"读取 server/.env 失败: {e}")
        
        # 尝试从 runtime_data/api_config.json 读取（备用）
        try:
            config_file = Path("runtime_data") / "api_config.json"
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    return config.get('api_url', '')
        except:
            pass
        
        return None
    
    def _activate_online(self, license_key: str, machine_id: str, api_url: str) -> Tuple[bool, str]:
        """在线激活（直接连接 Supabase 数据库）"""
        import requests
        
        print(f"正在连接激活服务器...")
        
        try:
            # 获取 Supabase 配置
            supabase_url, supabase_key = self._get_supabase_config()
            
            if not supabase_url or not supabase_key:
                return False, "激活失败: 服务器配置未找到"
            
            headers = {
                'apikey': supabase_key,
                'Authorization': f'Bearer {supabase_key}',
                'Content-Type': 'application/json'
            }
            
            # 1. 查询卡密是否存在
            query_url = f"{supabase_url}/rest/v1/licenses?license_key=eq.{license_key}&select=*"
            response = requests.get(query_url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                return False, f"激活失败: 服务器查询失败 ({response.status_code})"
            
            data = response.json()
            
            if not data:
                return False, "卡密错误：该卡密不存在，请检查卡密是否正确"
            
            license_data = data[0]
            
            # 2. 检查卡密状态
            status = license_data.get('status', '')
            
            if status == 'disabled':
                return False, "卡密错误：该卡密已被禁用，请联系管理员"
            
            # 3. 检查过期时间
            expires_at_str = license_data.get('expires_at')
            if expires_at_str:
                try:
                    expires_at = datetime.fromisoformat(expires_at_str.replace('Z', '+00:00'))
                    if datetime.now(expires_at.tzinfo) > expires_at:
                        return False, "卡密错误：该卡密已过期，请联系管理员续费"
                except:
                    pass
            
            # 4. 检查设备数量限制
            max_devices = license_data.get('max_devices', 1)
            
            # 查询已绑定的设备
            devices_url = f"{supabase_url}/rest/v1/device_bindings?license_key=eq.{license_key}&select=*"
            devices_response = requests.get(devices_url, headers=headers, timeout=10)
            
            if devices_response.status_code == 200:
                bound_devices = devices_response.json()
                
                # 检查当前设备是否已绑定
                is_bound = any(d.get('machine_id') == machine_id for d in bound_devices)
                
                if not is_bound:
                    # 新设备，检查是否超过限制
                    if len(bound_devices) >= max_devices:
                        return False, f"卡密错误：已达到最大设备数限制 ({max_devices}台)，请联系管理员"
                    
                    # 绑定新设备
                    bind_data = {
                        'license_key': license_key,
                        'machine_id': machine_id,
                        'activated_at': datetime.now().isoformat(),
                        'last_check_at': datetime.now().isoformat()
                    }
                    
                    bind_url = f"{supabase_url}/rest/v1/device_bindings"
                    bind_response = requests.post(bind_url, headers=headers, json=bind_data, timeout=10)
                    
                    if bind_response.status_code not in [200, 201]:
                        error_detail = bind_response.text if bind_response.text else f"状态码 {bind_response.status_code}"
                        print(f"[激活] 设备绑定失败: {error_detail}")
                        return False, f"激活失败: 设备绑定失败 - {error_detail}"
                else:
                    # 设备已绑定，更新最后检查时间
                    update_url = f"{supabase_url}/rest/v1/device_bindings?license_key=eq.{license_key}&machine_id=eq.{machine_id}"
                    update_data = {'last_check_at': datetime.now().isoformat()}
                    requests.patch(update_url, headers=headers, json=update_data, timeout=10)
            
            # 5. 更新卡密状态为 active（如果是 unused）
            if status == 'unused':
                update_license_url = f"{supabase_url}/rest/v1/licenses?license_key=eq.{license_key}"
                update_license_data = {
                    'status': 'active',
                    'activated_at': datetime.now().isoformat()
                }
                requests.patch(update_license_url, headers=headers, json=update_license_data, timeout=10)
            
            # 6. 保存到本地数据库
            local_data = {
                'license_key': license_key,
                'machine_id': machine_id,
                'status': 'active',
                'expires_at': license_data.get('expires_at', ''),
                'max_devices': max_devices,
                'activated_at': license_data.get('activated_at') or datetime.now().isoformat(),
                'last_online_check': datetime.now().isoformat()
            }
            
            self.db.save_license(local_data)
            
            return True, "激活成功！"
            
        except requests.exceptions.Timeout:
            return False, "激活失败: 连接超时，请检查网络连接"
        except requests.exceptions.ConnectionError:
            return False, "激活失败: 无法连接到服务器，请检查网络连接"
        except Exception as e:
            return False, f"激活失败: {e}"
    
    def _activate_offline(self, license_key: str, machine_id: str) -> Tuple[bool, str]:
        """离线激活（基于卡密格式验证）
        
        当无法连接服务器时使用此方法
        验证卡密格式和基本有效性
        """
        # 验证卡密格式
        valid, error_msg = self.validate_license_key(license_key)
        if not valid:
            return False, error_msg
        
        # 检查是否已有在线激活记录
        if self.license_file.exists():
            try:
                with open(self.license_file, 'r', encoding='utf-8') as f:
                    encrypted_data = json.load(f)
                activation_data = self._decrypt_data(encrypted_data)
                
                # 如果之前是在线激活的，允许继续使用
                if activation_data.get('online_activated'):
                    return True, "使用已有的在线激活记录"
            except:
                pass
        
        # 离线模式激活（宽松限制）
        print("⚠️ 离线激活模式：无法验证设备数量限制")
        
        # 生成离线激活信息
        activation_data = {
            'license_key': license_key,
            'machine_id': machine_id,
            'activated_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(days=365)).isoformat(),
            'status': 'active',
            'online_activated': False,  # 标记为离线激活
            'last_online_check': None,
            'max_devices': 999,  # 离线模式无法限制设备数
            'device_count': 1
        }
        
        # 保存激活信息
        try:
            encrypted_data = self._encrypt_data(activation_data)
            with open(self.license_file, 'w', encoding='utf-8') as f:
                json.dump(encrypted_data, f)
            
            return True, "激活成功！（离线模式，建议联网后重新激活以启用设备限制）"
        except Exception as e:
            return False, f"激活失败：{e}"
    
    def check_license(self) -> Tuple[bool, str, Optional[dict]]:
        """检查许可证状态（本机永久授权版本）
        
        本机永久授权：
        - 跳过所有在线验证
        - 跳过过期时间检查
        - 直接返回有效状态
        
        Returns:
            (是否有效, 消息, 许可证信息)
        """
        # 从本地数据库读取
        license_data = self.db.get_license()
        
        if not license_data:
            # 如果没有许可证数据，创建一个永久授权
            print("[许可证] 本机永久授权模式：创建永久许可证")
            machine_id = self.get_machine_id()
            permanent_license = {
                'license_key': 'PERMANENT-LOCAL-LICENSE',
                'machine_id': machine_id,
                'status': 'active',
                'expires_at': '9999-12-31T23:59:59+00:00',  # 永不过期
                'max_devices': 999,
                'activated_at': datetime.now().isoformat(),
                'last_online_check': datetime.now().isoformat()
            }
            self.db.save_license(permanent_license)
            license_data = permanent_license
        
        # 本机永久授权：跳过所有验证，直接返回有效
        print("[许可证] 本机永久授权模式：许可证永久有效")
        
        # 返回许可证信息
        info = {
            'license_key': license_data.get('license_key', 'PERMANENT-LOCAL-LICENSE'),
            'status': 'active',
            'expires_at': '9999-12-31T23:59:59+00:00',
            'max_devices': 999,
            'activated_at': license_data.get('activated_at', datetime.now().isoformat()),
            'days_until_expiry': 999999  # 永不过期
        }
        
        return True, "许可证永久有效（本机授权）", info
    
    def _verify_license_online(self, license_key: str, machine_id: str) -> Tuple[bool, str]:
        """在线验证许可证（用于安全检查）
        
        Args:
            license_key: 卡密
            machine_id: 机器ID
            
        Returns:
            (是否有效, 消息)
        """
        try:
            import requests
            
            # 获取 Supabase 配置
            supabase_url, supabase_key = self._get_supabase_config()
            
            if not supabase_url or not supabase_key:
                return False, "无法连接到验证服务器"
            
            headers = {
                'apikey': supabase_key,
                'Authorization': f'Bearer {supabase_key}',
                'Content-Type': 'application/json'
            }
            
            # 查询卡密是否存在
            query_url = f"{supabase_url}/rest/v1/licenses?license_key=eq.{license_key}&select=*"
            response = requests.get(query_url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                return False, f"服务器查询失败 ({response.status_code})"
            
            data = response.json()
            
            if not data:
                return False, "卡密不存在或已失效"
            
            license_data = data[0]
            
            # 检查卡密状态
            status = license_data.get('status', '')
            if status == 'disabled':
                return False, "卡密已被禁用"
            
            # 检查过期时间
            expires_at_str = license_data.get('expires_at')
            if expires_at_str:
                try:
                    expires_at = datetime.fromisoformat(expires_at_str.replace('Z', '+00:00'))
                    if datetime.now(expires_at.tzinfo) > expires_at:
                        return False, "卡密已过期"
                except:
                    pass
            
            # 检查设备绑定
            devices_url = f"{supabase_url}/rest/v1/device_bindings?license_key=eq.{license_key}&select=*"
            devices_response = requests.get(devices_url, headers=headers, timeout=10)
            
            if devices_response.status_code == 200:
                bound_devices = devices_response.json()
                
                # 检查当前设备是否已绑定
                is_bound = any(d.get('machine_id') == machine_id for d in bound_devices)
                
                if not is_bound:
                    max_devices = license_data.get('max_devices', 1)
                    if len(bound_devices) >= max_devices:
                        return False, f"已达到最大设备数限制 ({max_devices})"
                    else:
                        return False, "设备未绑定"
                
                # 更新最后检查时间
                update_url = f"{supabase_url}/rest/v1/device_bindings?license_key=eq.{license_key}&machine_id=eq.{machine_id}"
                update_data = {'last_check_at': datetime.now().isoformat()}
                requests.patch(update_url, headers=headers, json=update_data, timeout=10)
                
                # 同步到本地数据库
                local_data = self.db.get_license()
                if local_data:
                    local_data['last_online_check'] = datetime.now().isoformat()
                    local_data['status'] = license_data.get('status', 'active')
                    local_data['expires_at'] = license_data.get('expires_at', '')
                    local_data['max_devices'] = license_data.get('max_devices', 1)
                    self.db.save_license(local_data)
            
            return True, "在线验证成功"
            
        except requests.exceptions.Timeout:
            return False, "连接超时"
        except requests.exceptions.ConnectionError:
            return False, "无法连接到服务器"
        except Exception as e:
            return False, f"验证错误: {e}"
        """加密数据（简单加密，实际项目应使用更强的加密）"""
        import base64
        
        json_str = json.dumps(data)
        encoded = base64.b64encode(json_str.encode()).decode()
        
        # 添加校验和
        checksum = hashlib.sha256(json_str.encode()).hexdigest()
        
        return {
            'data': encoded,
            'checksum': checksum
        }
    
    def _decrypt_data(self, encrypted: dict) -> dict:
        """解密数据"""
        import base64
        
        encoded = encrypted.get('data', '')
        checksum = encrypted.get('checksum', '')
        
        # 解码
        json_str = base64.b64decode(encoded.encode()).decode()
        
        # 验证校验和
        calculated_checksum = hashlib.sha256(json_str.encode()).hexdigest()
        if calculated_checksum != checksum:
            raise ValueError("数据已被篡改")
        
        return json.loads(json_str)
    
    def get_license_info(self) -> Optional[dict]:
        """获取许可证信息（本机永久授权版本）"""
        valid, msg, info = self.check_license()
        if valid and info:
            return {
                'license_key': info.get('license_key', 'PERMANENT-LOCAL-LICENSE'),
                'activated_at': info.get('activated_at', datetime.now().isoformat()),
                'expires_at': '9999-12-31T23:59:59+00:00',
                'days_left': 999999,  # 永不过期
                'status': 'active',
                'max_devices': 999,
                'days_since_online': 0
            }
        return None
    
    def _check_database(self, license_key: str, machine_id: str) -> Tuple[bool, str, Optional[dict]]:
        """直接查询数据库验证许可证（加密通信）
        
        Args:
            license_key: 卡密
            machine_id: 机器ID
            
        Returns:
            (是否有效, 消息, 许可证信息)
        """
        # 获取API地址
        api_url = self._get_api_url()
        if not api_url:
            return False, "无法连接到激活服务器", None
        
        try:
            import requests
            
            # 准备请求数据（不包含 timestamp）
            request_data = {
                'license_key': license_key,
                'machine_id': machine_id
            }
            
            # 加密请求数据
            encrypted_request = crypto.encrypt_request(request_data)
            
            print(f"[验证] 验证卡密: {crypto.obfuscate_license_key(license_key)}")
            
            # 调用检查API
            response = requests.post(
                f"{api_url}/api/check",
                json=encrypted_request,
                timeout=10,
                headers={
                    'Content-Type': 'application/json',
                    'User-Agent': 'XiMengAutomation/1.0'
                }
            )
            
            # 解密响应
            encrypted_response = response.json()
            result = crypto.decrypt_response(encrypted_response)
            
            if result.get('success'):
                data = result.get('data', {})
                return True, "许可证有效", data
            else:
                return False, result.get('message', '验证失败'), None
                
        except requests.exceptions.Timeout:
            return False, "连接超时，请检查网络", None
        except requests.exceptions.ConnectionError:
            return False, "无法连接到服务器", None
        except Exception as e:
            return False, f"验证错误: {e}", None
    
    def _start_background_validation(self):
        """启动后台线程同步远程数据库（程序启动时执行一次）"""
        # 避免重复启动
        if self._validation_thread and self._validation_thread.is_alive():
            return
        
        # 检查是否刚更新过（1小时内不重复）
        if self._last_check_time:
            elapsed = (datetime.now() - self._last_check_time).total_seconds()
            if elapsed < 3600:  # 1小时内不重复更新
                return
        
        self._last_check_time = datetime.now()
        
        def sync_task():
            """后台同步任务（静默执行，不影响主程序）"""
            try:
                print("[后台同步] 开始同步远程数据库...")
                success = self._update_last_check()
                
                if success:
                    print("[后台同步] 同步成功，本地数据库已更新")
                else:
                    # 静默失败，不打印错误信息（避免干扰用户）
                    pass
            except Exception as e:
                # 静默失败，不打印错误信息
                pass
        
        self._validation_thread = threading.Thread(target=sync_task, daemon=True)
        self._validation_thread.start()
    
    def _start_continuous_sync(self):
        """启动持续同步线程（每分钟同步一次，检测到卡密不存在时30秒后关闭程序）"""
        # 避免重复启动
        if self._reauth_thread and self._reauth_thread.is_alive():
            print("[持续同步] 线程已在运行，跳过启动")
            return
        
        # 重置停止标志
        self._stop_reauth.clear()
        
        def continuous_sync_task():
            """持续同步任务（每分钟尝试一次）"""
            print("[持续同步] 后台同步线程已启动，每分钟检查一次许可证状态")
            
            license_invalid_start_time = None  # 记录卡密失效的开始时间
            
            while not self._stop_reauth.is_set():
                try:
                    # 尝试同步
                    success = self._update_last_check()
                    
                    if success:
                        print("[持续同步] 许可证状态同步成功")
                        license_invalid_start_time = None  # 重置失效时间
                    else:
                        print("[持续同步] 同步失败，60秒后重试...")
                        
                        # 检查是否是卡密不存在的错误
                        license_data = self.db.get_license()
                        if license_data:
                            license_key = license_data.get('license_key', '')
                            machine_id = self.get_machine_id()
                            
                            # 尝试验证卡密是否存在
                            online_valid, online_msg = self._verify_license_online(license_key, machine_id)
                            
                            if not online_valid and "卡密不存在" in online_msg:
                                if license_invalid_start_time is None:
                                    license_invalid_start_time = datetime.now()
                                    print("[安全警告] 检测到卡密不存在，程序将在10分钟后自动关闭")
                                else:
                                    # 计算已经过去的时间
                                    elapsed_time = (datetime.now() - license_invalid_start_time).total_seconds()
                                    remaining_time = 600 - elapsed_time  # 10分钟 = 600秒
                                    
                                    if remaining_time <= 0:
                                        print("[安全警告] 卡密验证失败超过10分钟，程序即将关闭")
                                        # 显示警告对话框并关闭程序
                                        self._show_license_invalid_dialog()
                                        return
                                    else:
                                        seconds = int(remaining_time)
                                        if seconds > 60:
                                            minutes = seconds // 60
                                            secs = seconds % 60
                                            if minutes % 1 == 0 or seconds <= 60:  # 每分钟显示一次，最后1分钟每秒显示
                                                print(f"[安全警告] 卡密不存在，程序将在 {minutes}分{secs}秒 后关闭")
                                        else:
                                            if seconds % 5 == 0 or seconds <= 5:  # 最后1分钟每5秒显示一次，最后5秒每秒显示
                                                print(f"[安全警告] 卡密不存在，程序将在 {seconds}秒 后关闭")
                    
                except Exception as e:
                    print(f"[持续同步] 同步出错: {e}，60秒后重试...")
                
                # 等待时间：如果检测到卡密失效，每秒检查一次；否则每60秒检查一次
                wait_time = 1 if license_invalid_start_time else 60
                self._stop_reauth.wait(wait_time)
        
        self._reauth_thread = threading.Thread(target=continuous_sync_task, daemon=True)
        self._reauth_thread.start()
    
    def _show_license_not_found_dialog(self, error_msg: str):
        """显示卡密不存在对话框并立即退出程序
        
        Args:
            error_msg: 错误消息
        """
        try:
            import tkinter as tk
            from tkinter import messagebox
            
            print(f"[卡密不存在] 准备显示对话框并退出程序: {error_msg}")
            
            def show_dialog_and_exit():
                try:
                    # 创建独立的根窗口
                    dialog_root = tk.Tk()
                    dialog_root.withdraw()
                    dialog_root.attributes('-topmost', True)  # 置顶显示
                    
                    messagebox.showerror(
                        "卡密验证失败",
                        f"您的卡密验证失败！\n\n原因：{error_msg}\n\n程序将在确认后退出。\n\n如有疑问请联系管理员。"
                    )
                    
                    dialog_root.destroy()
                    print("[卡密不存在] 对话框已关闭，开始强制退出")
                    
                except Exception as e:
                    print(f"[卡密不存在] 显示对话框失败: {e}")
                finally:
                    # 强制终止所有线程和进程
                    self._force_exit()
            
            # 尝试在主线程中执行对话框显示
            try:
                import threading
                # 获取主窗口引用
                main_root = tk._default_root
                if main_root and threading.current_thread() is threading.main_thread():
                    # 在主线程中直接执行
                    show_dialog_and_exit()
                elif main_root:
                    # 在后台线程中，使用 after 方法调度到主线程
                    main_root.after(100, show_dialog_and_exit)
                    # 等待一小段时间让对话框显示
                    import time
                    time.sleep(2)
                    # 如果对话框没有触发退出，强制退出
                    self._force_exit()
                else:
                    # 没有主窗口，直接强制退出
                    print("[卡密不存在] 没有主窗口，直接强制退出")
                    self._force_exit()
            except Exception as e:
                print(f"[卡密不存在] 调度对话框失败: {e}")
                self._force_exit()
                    
        except Exception as e:
            print(f"[卡密不存在] 显示对话框异常: {e}")
            # 即使对话框失败也要退出程序
            self._force_exit()
    
    def _show_license_expired_dialog(self):
        """显示卡密过期对话框并更新GUI状态
        
        此方法会：
        1. 弹出过期提示框（说明10分钟缓冲期已结束）
        2. 更新GUI状态为"已过期"
        3. 删除本地许可证（已在调用前完成）
        """
        try:
            import tkinter as tk
            from tkinter import messagebox
            
            print("[卡密过期] 准备显示过期提示框")
            
            def show_expired_dialog():
                try:
                    # 创建独立的根窗口
                    dialog_root = tk.Tk()
                    dialog_root.withdraw()
                    dialog_root.attributes('-topmost', True)  # 置顶显示
                    
                    messagebox.showwarning(
                        "卡密已过期",
                        "您的卡密已过期！\n\n10分钟缓冲期已结束。\n\n请联系管理员续费后重新激活。\n\n程序将在确认后退出。"
                    )
                    
                    dialog_root.destroy()
                    print("[卡密过期] 对话框已关闭")
                    
                    # 尝试更新GUI状态
                    self._update_gui_status("已过期")
                    
                except Exception as e:
                    print(f"[卡密过期] 显示对话框失败: {e}")
                finally:
                    # 退出程序
                    self._force_exit()
            
            # 尝试在主线程中执行对话框显示
            try:
                import threading
                # 获取主窗口引用
                main_root = tk._default_root
                if main_root and threading.current_thread() is threading.main_thread():
                    # 在主线程中直接执行
                    show_expired_dialog()
                elif main_root:
                    # 在后台线程中，使用 after 方法调度到主线程
                    main_root.after(100, show_expired_dialog)
                    # 等待一小段时间让对话框显示
                    import time
                    time.sleep(2)
                    # 如果对话框没有触发退出，强制退出
                    self._force_exit()
                else:
                    # 没有主窗口，直接强制退出
                    print("[卡密过期] 没有主窗口，直接强制退出")
                    self._force_exit()
            except Exception as e:
                print(f"[卡密过期] 调度对话框失败: {e}")
                self._force_exit()
                    
        except Exception as e:
            print(f"[卡密过期] 显示对话框异常: {e}")
            # 即使对话框失败也要退出程序
            self._force_exit()
    
    def _update_gui_status(self, status: str):
        """更新GUI状态显示
        
        Args:
            status: 状态文本（如"已过期"）
        """
        try:
            import tkinter as tk
            
            # 获取主窗口引用
            main_root = tk._default_root
            if not main_root:
                print("[更新GUI] 没有主窗口，无法更新状态")
                return
            
            # 尝试查找状态标签并更新
            # 主GUI中的许可证标签是 license_label
            if hasattr(main_root, 'license_label'):
                main_root.license_label.config(text=f"许可证状态: {status}", fg='red')
                print(f"[更新GUI] 状态已更新为: {status}")
            else:
                print("[更新GUI] 未找到许可证标签")
                
        except Exception as e:
            print(f"[更新GUI] 更新状态失败: {e}")
    
    def _show_license_invalid_dialog(self):
        """显示许可证失效对话框并关闭程序"""
        try:
            import tkinter as tk
            from tkinter import messagebox
            import sys
            import os
            import threading
            
            print("[许可证失效] 准备显示对话框并退出程序")
            
            # 使用主线程安全的方式显示对话框
            def show_dialog_and_exit():
                try:
                    # 创建独立的根窗口
                    dialog_root = tk.Tk()
                    dialog_root.withdraw()
                    dialog_root.attributes('-topmost', True)  # 置顶显示
                    
                    messagebox.showerror(
                        "许可证失效",
                        "您的许可证已失效或被删除！\n\n程序将在确认后退出。\n\n如有疑问请联系管理员。"
                    )
                    
                    dialog_root.destroy()
                    print("[许可证失效] 对话框已关闭，开始强制退出")
                    
                except Exception as e:
                    print(f"[许可证失效] 显示对话框失败: {e}")
                finally:
                    # 强制终止所有线程和进程
                    self._force_exit()
            
            # 尝试在主线程中执行对话框显示
            try:
                # 获取主窗口引用
                main_root = tk._default_root
                if main_root and threading.current_thread() is threading.main_thread():
                    # 在主线程中直接执行
                    show_dialog_and_exit()
                elif main_root:
                    # 在后台线程中，使用 after 方法调度到主线程
                    main_root.after(100, show_dialog_and_exit)
                    # 等待一小段时间让对话框显示
                    import time
                    time.sleep(2)
                    # 如果对话框没有触发退出，强制退出
                    self._force_exit()
                else:
                    # 没有主窗口，直接强制退出
                    print("[许可证失效] 没有主窗口，直接强制退出")
                    self._force_exit()
            except Exception as e:
                print(f"[许可证失效] 调度对话框失败: {e}")
                self._force_exit()
                    
        except Exception as e:
            print(f"[许可证失效] 显示对话框异常: {e}")
            # 即使对话框失败也要退出程序
            self._force_exit()
    
    def _force_exit(self):
        """强制退出程序（确保所有线程和进程都被终止）"""
        try:
            import sys
            import os
            import threading
            import signal
            import tkinter as tk
            
            print("[强制退出] 正在终止程序...")
            
            # 停止所有验证线程
            self.stop_runtime_validation()
            
            # 尝试关闭所有 Tkinter 窗口
            try:
                # 获取所有 Tkinter 根窗口
                if hasattr(tk, '_default_root') and tk._default_root:
                    print("[强制退出] 关闭主窗口")
                    tk._default_root.quit()  # 退出主循环
                    tk._default_root.destroy()  # 销毁窗口
                    
                # 强制退出所有 Tkinter 相关的事件循环
                try:
                    tk._default_root = None
                except:
                    pass
            except Exception as e:
                print(f"[强制退出] 关闭GUI失败: {e}")
            
            # 尝试优雅关闭
            try:
                # 发送 SIGTERM 信号给自己
                if hasattr(signal, 'SIGTERM'):
                    os.kill(os.getpid(), signal.SIGTERM)
            except Exception as e:
                print(f"[强制退出] 发送信号失败: {e}")
            
            # 等待一小段时间让其他线程清理
            import time
            time.sleep(0.2)
            
            # 强制退出所有线程
            try:
                # 获取当前所有线程
                for thread in threading.enumerate():
                    if thread != threading.current_thread() and thread.is_alive():
                        print(f"[强制退出] 等待线程结束: {thread.name}")
                        # 不等待守护线程
                        if not thread.daemon:
                            thread.join(timeout=0.1)
            except Exception as e:
                print(f"[强制退出] 清理线程失败: {e}")
            
            print("[强制退出] 执行最终退出")
            
            # 强制退出
            if getattr(sys, 'frozen', False):
                # PyInstaller 环境
                os._exit(1)
            else:
                # 开发环境
                sys.exit(1)
                
        except Exception as e:
            print(f"[强制退出] 退出过程异常: {e}")
            # 最后的手段
            import os
            os._exit(1)
    
    def _update_last_check(self) -> bool:
        """更新远程数据库中的 last_check_at 时间戳，并同步到本地（直接连接 Supabase）
        
        Returns:
            是否成功
        """
        license_data = self.db.get_license()
        if not license_data:
            print("[同步] 失败：本地没有许可证数据")
            return False
        
        license_key = license_data.get('license_key', '')
        machine_id = self.get_machine_id()
        
        # 直接连接 Supabase
        return self._update_via_supabase(license_key, machine_id, license_data)
    
    def _update_via_supabase(self, license_key: str, machine_id: str, license_data: dict) -> bool:
        """直接连接 Supabase 更新许可证状态
        
        Args:
            license_key: 卡密
            machine_id: 机器ID
            license_data: 本地许可证数据
            
        Returns:
            是否成功
        """
        try:
            import requests
            
            # 获取 Supabase 配置
            supabase_url, supabase_key = self._get_supabase_config()
            
            if not supabase_url or not supabase_key:
                print("[同步] 失败：Supabase 配置未找到")
                return False
            
            headers = {
                'apikey': supabase_key,
                'Authorization': f'Bearer {supabase_key}',
                'Content-Type': 'application/json'
            }
            
            # 查询卡密
            query_url = f"{supabase_url}/rest/v1/licenses?license_key=eq.{license_key}&select=*"
            response = requests.get(query_url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                print(f"[同步] Supabase 查询失败: {response.status_code}")
                return False
            
            data = response.json()
            if not data:
                print("[同步] 失败：卡密不存在")
                return False
            
            db_license = data[0]
            
            # 更新设备的 last_check_at
            update_device_url = f"{supabase_url}/rest/v1/device_bindings?license_key=eq.{license_key}&machine_id=eq.{machine_id}"
            update_device_data = {
                'last_check_at': datetime.now().isoformat()
            }
            requests.patch(update_device_url, headers=headers, json=update_device_data, timeout=10)
            
            # 同步到本地
            license_data['last_online_check'] = datetime.now().isoformat()
            license_data['status'] = db_license.get('status', 'active')
            license_data['expires_at'] = db_license.get('expires_at', '')
            license_data['max_devices'] = db_license.get('max_devices', 1)
            self.db.save_license(license_data)
            
            print(f"[同步] ✅ 同步成功")
            return True
            
        except requests.exceptions.Timeout:
            print("[同步] 连接超时")
            return False
        except Exception as e:
            print(f"[同步] 同步失败: {e}")
            return False
    
    def start_runtime_validation(self, on_invalid_callback=None):
        """启动运行时验证（已废弃，保留接口兼容性）
        
        新版本在程序启动时自动进行后台验证，不需要手动启动
        """
        pass
    
    def stop_runtime_validation(self):
        """停止运行时验证和持续同步"""
        # 停止旧的验证线程
        self._stop_validation.set()
        if self._validation_thread and self._validation_thread.is_alive():
            self._validation_thread.join(timeout=1)
        
        # 停止持续同步线程
        self._stop_reauth.set()
        if self._reauth_thread and self._reauth_thread.is_alive():
            self._reauth_thread.join(timeout=1)
    
    def validate_online(self) -> Tuple[bool, str]:
        """在线验证许可证（已废弃，保留接口兼容性）
        
        新版本直接查询数据库，不需要单独的在线验证方法
        
        Returns:
            (是否有效, 消息)
        """
        valid, message, _ = self.check_license()
        return valid, message
