"""
加密工具模块
用于保护客户端与服务器之间的通信
使用 AES-256-GCM 加密
"""

import base64
import hashlib
import json
import os
from typing import Dict, Any, Tuple
from datetime import datetime
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Protocol.KDF import PBKDF2


class CryptoUtils:
    """加密工具类（AES-256-GCM）"""
    
    # 固定盐值（用于派生密钥，实际部署时应该使用环境变量）
    SALT = b'XiMengAutomation2026SecretSalt'
    
    @staticmethod
    def generate_aes_key(password: str, salt: bytes = None) -> bytes:
        """生成 AES-256 密钥
        
        Args:
            password: 密码
            salt: 盐值
            
        Returns:
            32字节密钥
        """
        if salt is None:
            salt = CryptoUtils.SALT
        
        # 使用 PBKDF2 派生密钥
        key = PBKDF2(password, salt, dkLen=32, count=100000)
        return key
    
    @staticmethod
    def encrypt_aes_gcm(data: bytes, key: bytes) -> Dict[str, str]:
        """使用 AES-256-GCM 加密数据
        
        Args:
            data: 原始数据
            key: 32字节密钥
            
        Returns:
            包含密文、nonce、tag 的字典
        """
        # 创建 AES-GCM 加密器
        cipher = AES.new(key, AES.MODE_GCM)
        
        # 加密数据
        ciphertext, tag = cipher.encrypt_and_digest(data)
        
        return {
            'ciphertext': base64.b64encode(ciphertext).decode(),
            'nonce': base64.b64encode(cipher.nonce).decode(),
            'tag': base64.b64encode(tag).decode()
        }
    
    @staticmethod
    def decrypt_aes_gcm(encrypted: Dict[str, str], key: bytes) -> bytes:
        """使用 AES-256-GCM 解密数据
        
        Args:
            encrypted: 包含密文、nonce、tag 的字典
            key: 32字节密钥
            
        Returns:
            解密后的数据
        """
        # 解码
        ciphertext = base64.b64decode(encrypted['ciphertext'])
        nonce = base64.b64decode(encrypted['nonce'])
        tag = base64.b64decode(encrypted['tag'])
        
        # 创建 AES-GCM 解密器
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
        
        # 解密并验证
        plaintext = cipher.decrypt_and_verify(ciphertext, tag)
        
        return plaintext
    
    @staticmethod
    def encrypt_request(data: Dict[str, Any]) -> Dict[str, Any]:
        """加密请求数据（AES-256-GCM）
        
        Args:
            data: 原始请求数据
            
        Returns:
            加密后的请求数据
        """
        try:
            # 生成随机密钥
            timestamp = str(int(datetime.now().timestamp() * 1000))
            random_password = base64.b64encode(get_random_bytes(16)).decode()  # 16字节足够
            
            # 派生 AES 密钥（使用完整密码）
            key = CryptoUtils.generate_aes_key(random_password + timestamp)
            
            # 将数据转为 JSON
            json_data = json.dumps(data, ensure_ascii=False)
            
            # AES-GCM 加密
            encrypted = CryptoUtils.encrypt_aes_gcm(json_data.encode(), key)
            
            # 生成签名
            signature = CryptoUtils.generate_request_signature(data, random_password)
            
            # 返回加密数据（传输完整密码）
            return {
                'encrypted_data': encrypted['ciphertext'],
                'nonce': encrypted['nonce'],
                'tag': encrypted['tag'],
                'key_hint': random_password,  # 传输完整密码
                'timestamp': timestamp,
                'signature': signature
            }
            
        except Exception as e:
            print(f"[加密] 加密失败: {e}")
            # 加密失败，返回原始数据（降级）
            return data
    
    @staticmethod
    def decrypt_response(encrypted_data: Dict[str, Any]) -> Dict[str, Any]:
        """解密服务器响应（AES-256-GCM）
        
        Args:
            encrypted_data: 加密的响应数据
            
        Returns:
            解密后的响应数据
        """
        try:
            # 检查是否是加密数据
            if 'encrypted_data' not in encrypted_data:
                return encrypted_data
            
            # 提取加密信息
            key_hint = encrypted_data.get('key_hint', '')
            timestamp = encrypted_data.get('timestamp', '')
            
            # 重建密钥
            key = CryptoUtils.generate_aes_key(key_hint + timestamp)
            
            # 解密
            encrypted = {
                'ciphertext': encrypted_data['encrypted_data'],
                'nonce': encrypted_data['nonce'],
                'tag': encrypted_data['tag']
            }
            
            decrypted = CryptoUtils.decrypt_aes_gcm(encrypted, key)
            
            # 解析 JSON
            result = json.loads(decrypted.decode())
            
            return result
            
        except Exception as e:
            print(f"[解密] 解密失败: {e}")
            # 解密失败，返回原始数据
            return encrypted_data
    
    @staticmethod
    def generate_request_signature(data: Dict[str, Any], secret: str = None) -> str:
        """生成请求签名（HMAC-SHA256）
        
        Args:
            data: 请求数据
            secret: 密钥
            
        Returns:
            签名字符串
        """
        # 按键排序
        sorted_keys = sorted(data.keys())
        
        # 拼接字符串
        sign_str = ""
        for key in sorted_keys:
            value = data[key]
            if value is not None:
                sign_str += f"{key}={value}&"
        
        # 添加时间戳盐
        timestamp = str(int(datetime.now().timestamp()))
        sign_str += f"timestamp={timestamp}"
        
        # 如果有密钥，添加到签名字符串
        if secret:
            sign_str += f"&secret={secret}"
        
        # HMAC-SHA256
        import hmac
        signature = hmac.new(
            CryptoUtils.SALT,
            sign_str.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    @staticmethod
    def encrypt_database_value(value: str, key: bytes = None) -> str:
        """加密数据库字段值
        
        Args:
            value: 原始值
            key: 加密密钥（可选）
            
        Returns:
            加密后的值（Base64）
        """
        try:
            if key is None:
                # 使用机器特征生成密钥
                key = CryptoUtils.generate_aes_key("LocalDBEncryption")
            
            # AES-GCM 加密
            encrypted = CryptoUtils.encrypt_aes_gcm(value.encode(), key)
            
            # 组合所有部分
            combined = json.dumps(encrypted)
            
            # Base64 编码
            return base64.b64encode(combined.encode()).decode()
            
        except Exception as e:
            print(f"[数据库加密] 失败: {e}")
            return value
    
    @staticmethod
    def decrypt_database_value(encrypted_value: str, key: bytes = None) -> str:
        """解密数据库字段值
        
        Args:
            encrypted_value: 加密的值
            key: 解密密钥（可选）
            
        Returns:
            解密后的值
        """
        try:
            if key is None:
                # 使用机器特征生成密钥
                key = CryptoUtils.generate_aes_key("LocalDBEncryption")
            
            # Base64 解码
            combined = base64.b64decode(encrypted_value.encode()).decode()
            
            # 解析 JSON
            encrypted = json.loads(combined)
            
            # AES-GCM 解密
            decrypted = CryptoUtils.decrypt_aes_gcm(encrypted, key)
            
            return decrypted.decode()
            
        except Exception as e:
            print(f"[数据库解密] 失败: {e}")
            return encrypted_value
    
    @staticmethod
    def hash_machine_id(machine_id: str) -> str:
        """对机器ID进行哈希（防止泄露真实硬件信息）
        
        Args:
            machine_id: 原始机器ID
            
        Returns:
            哈希后的机器ID
        """
        # 使用 SHA256 哈希
        hashed = hashlib.sha256(machine_id.encode()).hexdigest()
        return hashed
    
    @staticmethod
    def obfuscate_license_key(license_key: str) -> str:
        """混淆卡密（用于日志输出，防止泄露）
        
        Args:
            license_key: 原始卡密
            
        Returns:
            混淆后的卡密
        """
        if len(license_key) <= 8:
            return "****"
        
        # 只显示前4位和后4位
        return f"{license_key[:4]}****{license_key[-4:]}"
    
    @staticmethod
    def encrypt_file_content(data: bytes, password: str = "TemplateBackup2026") -> bytes:
        """加密文件内容
        
        Args:
            data: 原始文件数据
            password: 加密密码
            
        Returns:
            加密后的数据
        """
        try:
            # 生成密钥
            key = CryptoUtils.generate_aes_key(password)
            
            # AES-GCM 加密
            encrypted = CryptoUtils.encrypt_aes_gcm(data, key)
            
            # 组合所有部分为 JSON
            combined = json.dumps(encrypted)
            
            # 返回字节数据
            return combined.encode()
            
        except Exception as e:
            print(f"[文件加密] 失败: {e}")
            raise
    
    @staticmethod
    def decrypt_file_content(encrypted_data: bytes, password: str = "TemplateBackup2026") -> bytes:
        """解密文件内容
        
        Args:
            encrypted_data: 加密的文件数据
            password: 解密密码
            
        Returns:
            解密后的数据
        """
        try:
            # 生成密钥
            key = CryptoUtils.generate_aes_key(password)
            
            # 解析 JSON
            encrypted = json.loads(encrypted_data.decode())
            
            # AES-GCM 解密
            decrypted = CryptoUtils.decrypt_aes_gcm(encrypted, key)
            
            return decrypted
            
        except Exception as e:
            print(f"[文件解密] 失败: {e}")
            raise
    
    @staticmethod
    def get_machine_id() -> str:
        """获取机器唯一ID（基于硬件信息）
        
        Returns:
            机器ID（SHA256哈希）
        """
        try:
            import platform
            import uuid
            
            machine_info = []
            
            # 1. CPU信息
            try:
                import cpuinfo
                cpu_info = cpuinfo.get_cpu_info()
                machine_info.append(cpu_info.get('brand_raw', ''))
            except:
                machine_info.append(platform.processor())
            
            # 2. 主板UUID（最稳定的标识）
            try:
                if platform.system() == 'Windows':
                    import subprocess
                    result = subprocess.check_output('wmic csproduct get uuid', shell=True)
                    uuid_str = result.decode().split('\n')[1].strip()
                    machine_info.append(uuid_str)
                else:
                    machine_info.append(str(uuid.getnode()))
            except:
                machine_info.append(str(uuid.getnode()))
            
            # 3. 系统信息
            machine_info.append(platform.system())
            machine_info.append(platform.machine())
            
            # 生成唯一ID
            combined = ''.join(machine_info)
            machine_id = hashlib.sha256(combined.encode()).hexdigest()
            
            return machine_id
            
        except Exception as e:
            print(f"[机器ID] 获取失败: {e}")
            # 降级方案：使用MAC地址
            return hashlib.sha256(str(uuid.getnode()).encode()).hexdigest()
    
    @staticmethod
    def encrypt_with_machine_binding(data: bytes) -> bytes:
        """使用机器绑定加密数据
        
        数据只能在当前机器上解密，复制到其他机器无法解密
        
        Args:
            data: 原始数据
            
        Returns:
            加密后的数据
        """
        try:
            # 获取机器ID作为密钥
            machine_id = CryptoUtils.get_machine_id()
            key = CryptoUtils.generate_aes_key(machine_id)
            
            # AES-GCM 加密
            encrypted = CryptoUtils.encrypt_aes_gcm(data, key)
            
            # 添加标识头（用于识别这是机器绑定加密的数据）
            result = {
                'version': '1.0',
                'type': 'machine_binding',
                'data': encrypted
            }
            
            # 转为JSON并编码
            return json.dumps(result).encode()
            
        except Exception as e:
            print(f"[机器绑定加密] 失败: {e}")
            raise
    
    @staticmethod
    def decrypt_with_machine_binding(encrypted_data: bytes) -> bytes:
        """解密机器绑定的数据
        
        只能在加密时的机器上解密
        
        Args:
            encrypted_data: 加密的数据
            
        Returns:
            解密后的数据
            
        Raises:
            ValueError: 如果在不同机器上解密
        """
        try:
            # 解析JSON
            result = json.loads(encrypted_data.decode())
            
            # 检查类型
            if result.get('type') != 'machine_binding':
                raise ValueError("不是机器绑定加密的数据")
            
            # 获取当前机器ID
            machine_id = CryptoUtils.get_machine_id()
            key = CryptoUtils.generate_aes_key(machine_id)
            
            # 解密
            encrypted = result['data']
            decrypted = CryptoUtils.decrypt_aes_gcm(encrypted, key)
            
            return decrypted
            
        except Exception as e:
            # 解密失败，可能是在不同机器上
            raise ValueError(f"解密失败：数据可能是在其他机器上加密的，或数据已损坏。错误: {e}")


# 全局加密工具实例
crypto = CryptoUtils()
