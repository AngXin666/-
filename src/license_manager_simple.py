"""
简化的卡密激活管理器
License Activation Manager (Simplified)

重构原则：
1. 不调用任何外部命令（避免 wmic.exe 等）
2. 使用最简单的硬件标识方法
3. 本地存储，简化验证逻辑
4. 使用 Supabase 数据库进行在线验证
"""

import hashlib
import json
import uuid
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple


class SimpleLicenseManager:
    """简化的卡密管理器"""
    
    def __init__(self):
        # 许可证文件路径（统一使用绝对路径）
        import sys
        
        # 确定许可证文件的存储位置
        if getattr(sys, 'frozen', False):
            # 打包后的 EXE，使用 EXE 同目录
            base_dir = Path(sys.executable).parent
        else:
            # 开发环境，使用项目根目录
            base_dir = Path(__file__).parent.parent
        
        self.license_file = base_dir / "runtime_data" / "license_simple.json"
        self.license_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 加载 Supabase 配置
        self.supabase_url = None
        self.supabase_key = None
        self._load_supabase_config()
    
    def _load_supabase_config(self):
        """从 .env 文件加载 Supabase 配置（支持多路径）"""
        import sys
        
        # 配置文件查找路径（按优先级）
        search_paths = []
        
        # 1. EXE 同目录（打包后）
        if getattr(sys, 'frozen', False):
            search_paths.append(Path(sys.executable).parent / ".env")
        
        # 2. 用户数据目录
        search_paths.append(Path("runtime_data") / ".env")
        
        # 3. 项目根目录（开发环境）
        search_paths.append(Path(".env"))
        
        env_file = None
        for path in search_paths:
            if path.exists():
                env_file = path
                print(f"[配置] 找到配置文件: {path.absolute()}")
                break
        
        if not env_file:
            print(f"[错误] 未找到配置文件，已搜索以下位置:")
            for path in search_paths:
                print(f"  - {path.absolute()}")
            print(f"[提示] 请将 .env 文件放在以下任一位置:")
            print(f"  1. EXE 文件同目录")
            print(f"  2. runtime_data 目录")
            print(f"  3. 项目根目录")
            return
        
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        if key == 'SUPABASE_URL':
                            self.supabase_url = value
                        elif key == 'SUPABASE_KEY':
                            self.supabase_key = value
            
            if self.supabase_url and self.supabase_key:
                print(f"[配置] Supabase 配置加载成功")
            else:
                print(f"[警告] 配置文件不完整")
        except Exception as e:
            print(f"[错误] 读取配置文件失败: {e}")
    
    def get_machine_id(self) -> str:
        """获取机器唯一ID（最简单安全的方式）
        
        使用方法：
        1. MAC 地址（uuid.getnode() - 纯 Python，不调用外部命令）
        2. 环境变量（Windows 特有，不调用外部命令）
        3. 组合生成 SHA256 哈希
        """
        try:
            parts = []
            
            # 1. MAC 地址（最可靠）
            mac = uuid.getnode()
            parts.append(str(mac))
            
            # 2. 计算机名（环境变量）
            comp_name = os.environ.get('COMPUTERNAME', 'UNKNOWN')
            parts.append(comp_name)
            
            # 3. 用户名（环境变量）
            username = os.environ.get('USERNAME', 'UNKNOWN')
            parts.append(username)
            
            # 生成哈希
            combined = '-'.join(parts)
            machine_id = hashlib.sha256(combined.encode()).hexdigest()
            
            return machine_id
            
        except Exception as e:
            print(f"[错误] 获取机器ID失败: {e}")
            # 备用：只用 MAC 地址
            return hashlib.sha256(str(uuid.getnode()).encode()).hexdigest()
    
    def save_license(self, license_key: str, machine_id: str, expires_days: int = 365) -> bool:
        """保存许可证信息
        
        Args:
            license_key: 卡密
            machine_id: 机器ID
            expires_days: 有效天数
            
        Returns:
            是否成功
        """
        try:
            from datetime import timedelta
            
            license_data = {
                'license_key': license_key,
                'machine_id': machine_id,
                'activated_at': datetime.now().isoformat(),
                'expires_at': (datetime.now() + timedelta(days=expires_days)).isoformat(),
                'status': 'active'
            }
            
            with open(self.license_file, 'w', encoding='utf-8') as f:
                json.dump(license_data, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            print(f"[错误] 保存许可证失败: {e}")
            return False
    
    def check_license(self) -> Tuple[bool, str]:
        """检查许可证状态（本机永久授权版本）
        
        本机永久授权：
        - 跳过所有在线验证
        - 跳过过期时间检查
        - 直接返回有效状态
        
        Returns:
            (是否有效, 消息)
        """
        try:
            # 检查文件是否存在
            if not self.license_file.exists():
                # 如果没有许可证文件，创建一个永久授权
                print("[许可证] 本机永久授权模式：创建永久许可证")
                machine_id = self.get_machine_id()
                
                # 创建永久授权数据
                license_data = {
                    'license_key': 'PERMANENT-LOCAL-LICENSE',
                    'machine_id': machine_id,
                    'status': 'active',
                    'expires_at': '9999-12-31T23:59:59',  # 永不过期
                    'activated_at': datetime.now().isoformat()
                }
                
                # 保存到文件
                self.license_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.license_file, 'w', encoding='utf-8') as f:
                    json.dump(license_data, f, indent=2)
            
            # 本机永久授权：跳过所有验证，直接返回有效
            print("[许可证] 本机永久授权模式：许可证永久有效")
            return True, "许可证永久有效（本机授权）"
            
        except Exception as e:
            print(f"[错误] 检查许可证失败: {e}")
            return False, f"验证错误: {e}"
    
    def activate_license(self, license_key: str) -> Tuple[bool, str]:
        """激活许可证（在线验证版本）
        
        Args:
            license_key: 卡密
            
        Returns:
            (是否成功, 消息)
        """
        try:
            # 验证卡密格式
            key = license_key.replace(' ', '').replace('-', '').upper()
            if len(key) != 20 or not key.isalnum():
                return False, "卡密格式错误"
            
            # 获取机器ID
            machine_id = self.get_machine_id()
            
            # 在线验证卡密
            print(f"[信息] 正在验证卡密: {license_key}")
            success, message, expires_days = self._verify_license_online(license_key, machine_id)
            
            if not success:
                return False, message
            
            # 保存许可证
            if self.save_license(license_key, machine_id, expires_days=expires_days):
                return True, f"激活成功！有效期 {expires_days} 天"
            else:
                return False, "保存许可证失败"
                
        except Exception as e:
            print(f"[错误] 激活失败: {e}")
            return False, f"激活错误: {e}"
    
    def _verify_license_online(self, license_key: str, machine_id: str) -> Tuple[bool, str, int]:
        """在线验证卡密（使用 Supabase 数据库）
        
        Args:
            license_key: 卡密
            machine_id: 机器ID
            
        Returns:
            (是否成功, 消息, 有效天数)
        """
        try:
            import requests
            
            if not self.supabase_url or not self.supabase_key:
                print(f"[错误] Supabase 配置未加载")
                error_msg = (
                    "配置错误：未找到数据库配置\n\n"
                    "请确保 .env 文件存在于以下任一位置：\n"
                    "1. EXE 文件同目录\n"
                    "2. runtime_data 目录\n"
                    "3. 项目根目录\n\n"
                    "配置文件应包含：\n"
                    "SUPABASE_URL=你的URL\n"
                    "SUPABASE_KEY=你的KEY"
                )
                return False, error_msg, 0
            
            # 1. 查询卡密是否存在
            print(f"[信息] 连接数据库验证...")
            headers = {
                'apikey': self.supabase_key,
                'Authorization': f'Bearer {self.supabase_key}',
                'Content-Type': 'application/json'
            }
            
            # 查询卡密
            url = f"{self.supabase_url}/rest/v1/licenses?license_key=eq.{license_key}"
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                print(f"[错误] 数据库查询失败: {response.status_code}")
                return False, f"数据库查询失败: {response.status_code}", 0
            
            licenses = response.json()
            
            if not licenses or len(licenses) == 0:
                print(f"[错误] 卡密不存在")
                return False, "卡密不存在或已失效", 0
            
            license_data = licenses[0]
            
            # 2. 检查卡密状态
            status = license_data.get('status')
            if status == 'disabled':
                print(f"[错误] 卡密已被禁用")
                return False, "卡密已被禁用", 0
            
            # 3. 检查过期时间
            expires_at_str = license_data.get('expires_at')
            if expires_at_str:
                try:
                    expires_at = datetime.fromisoformat(expires_at_str.replace('Z', '+00:00'))
                    if datetime.now(expires_at.tzinfo) > expires_at:
                        print(f"[错误] 卡密已过期")
                        return False, "卡密已过期", 0
                    
                    # 计算剩余天数
                    days_left = (expires_at - datetime.now(expires_at.tzinfo)).days
                except Exception as e:
                    print(f"[警告] 解析过期时间失败: {e}")
                    days_left = 365
            else:
                days_left = 365
            
            # 4. 检查设备绑定
            max_devices = license_data.get('max_devices', 1)
            
            # 查询已绑定的设备
            bindings_url = f"{self.supabase_url}/rest/v1/device_bindings?license_key=eq.{license_key}"
            bindings_response = requests.get(bindings_url, headers=headers, timeout=10)
            
            if bindings_response.status_code == 200:
                bindings = bindings_response.json()
                
                # 检查当前设备是否已绑定
                is_bound = False
                for binding in bindings:
                    if binding.get('machine_id') == machine_id:
                        is_bound = True
                        print(f"[信息] 当前设备已绑定，允许重新激活")
                        break
                
                # 如果未绑定，检查是否超过设备限制
                if not is_bound:
                    if len(bindings) >= max_devices:
                        print(f"[错误] 设备数量已达上限 ({len(bindings)}/{max_devices})")
                        return False, f"设备数量已达上限 ({len(bindings)}/{max_devices})", 0
                    
                    # 绑定新设备
                    print(f"[信息] 绑定新设备...")
                    bind_data = {
                        'license_key': license_key,
                        'machine_id': machine_id,
                        'activated_at': datetime.now().isoformat()
                    }
                    
                    bind_url = f"{self.supabase_url}/rest/v1/device_bindings"
                    bind_response = requests.post(bind_url, headers=headers, json=bind_data, timeout=10)
                    
                    print(f"[调试] 绑定响应状态码: {bind_response.status_code}")
                    print(f"[调试] 绑定响应内容: {bind_response.text}")
                    
                    if bind_response.status_code not in [200, 201]:
                        print(f"[错误] 绑定设备失败: {bind_response.status_code}")
                        error_detail = bind_response.text if bind_response.text else "未知错误"
                        return False, f"绑定设备失败: {error_detail}", 0
                else:
                    # 设备已绑定，允许重新激活（比如用户删除了本地许可证文件）
                    print(f"[信息] 设备已绑定此卡密，允许重新激活")
            
            # 5. 更新卡密状态为已激活
            if status == 'unused':
                print(f"[信息] 更新卡密状态为已激活...")
                update_data = {
                    'status': 'active',
                    'activated_at': datetime.now().isoformat()
                }
                
                update_url = f"{self.supabase_url}/rest/v1/licenses?license_key=eq.{license_key}"
                update_response = requests.patch(update_url, headers=headers, json=update_data, timeout=10)
                
                if update_response.status_code not in [200, 204]:
                    print(f"[警告] 更新卡密状态失败: {update_response.status_code}")
            
            print(f"[信息] 验证成功，有效期 {days_left} 天")
            return True, f"激活成功！有效期 {days_left} 天", days_left
            
        except requests.exceptions.Timeout:
            print(f"[错误] 连接数据库超时")
            return False, "连接数据库超时，请检查网络", 0
        except requests.exceptions.ConnectionError as e:
            print(f"[错误] 无法连接到数据库: {e}")
            return False, "无法连接到数据库，请检查网络", 0
        except Exception as e:
            import traceback
            print(f"[错误] 在线验证失败: {e}")
            print(f"[错误] 详细信息: {traceback.format_exc()}")
            return False, f"验证错误: {e}", 0
    
    def delete_license(self) -> bool:
        """删除许可证"""
        try:
            if self.license_file.exists():
                self.license_file.unlink()
            return True
        except Exception as e:
            print(f"[错误] 删除许可证失败: {e}")
            return False
    
    def get_license_info(self) -> Optional[dict]:
        """获取许可证信息（本机永久授权版本）
        
        Returns:
            许可证信息字典，如果未激活则返回 None
        """
        try:
            if not self.license_file.exists():
                # 如果没有许可证文件，创建一个永久授权
                print("[许可证] 本机永久授权模式：创建永久许可证")
                machine_id = self.get_machine_id()
                
                # 创建永久授权数据
                license_data = {
                    'license_key': 'PERMANENT-LOCAL-LICENSE',
                    'machine_id': machine_id,
                    'status': 'active',
                    'expires_at': '9999-12-31T23:59:59',  # 永不过期
                    'activated_at': datetime.now().isoformat()
                }
                
                # 保存到文件
                self.license_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.license_file, 'w', encoding='utf-8') as f:
                    json.dump(license_data, f, indent=2)
            else:
                # 读取现有许可证
                with open(self.license_file, 'r', encoding='utf-8') as f:
                    license_data = json.load(f)
            
            # 本机永久授权：返回永久有效的信息
            return {
                'license_key': license_data.get('license_key', 'PERMANENT-LOCAL-LICENSE'),
                'activated_at': license_data.get('activated_at', datetime.now().isoformat()),
                'expires_at': '9999-12-31T23:59:59',  # 永不过期
                'days_left': 999999,  # 永不过期
                'status': 'active'
            }
        except Exception as e:
            print(f"[错误] 获取许可证信息失败: {e}")
            return None
