"""
加密账号文件管理器
Encrypted Accounts File Manager

使用机器绑定加密保护账号文件，防止账号信息泄露
"""

import json
from pathlib import Path
from typing import List, Tuple, Optional


class EncryptedAccountsFile:
    """加密账号文件管理器（带机器绑定加密）
    
    功能：
    - 账号文件使用机器ID加密
    - 只能在当前机器上解密
    - 自动升级旧版本未加密文件
    - 兼容旧版本明文文件
    """
    
    def __init__(self, accounts_file: str):
        """初始化加密账号文件管理器
        
        Args:
            accounts_file: 账号文件路径（不含 .enc 后缀）
        """
        self.accounts_file = Path(accounts_file)
        self.encrypted_file = Path(str(accounts_file) + '.enc')
        
        # 导入加密工具
        try:
            from .crypto_utils import CryptoUtils
        except ImportError:
            try:
                from crypto_utils import CryptoUtils
            except ImportError:
                from src.crypto_utils import CryptoUtils
        self.crypto = CryptoUtils()
    
    def read_accounts(self) -> List[Tuple[str, str]]:
        """读取账号列表（自动解密）
        
        Returns:
            账号列表 [(phone, password), ...]
        """
        # 优先读取加密文件
        if self.encrypted_file.exists():
            try:
                with open(self.encrypted_file, 'rb') as f:
                    encrypted_data = f.read()
                
                # 解密
                plain_data = self.crypto.decrypt_with_machine_binding(encrypted_data)
                
                # 解析JSON
                accounts_data = json.loads(plain_data.decode('utf-8'))
                
                # 转换为列表
                accounts = []
                for item in accounts_data:
                    phone = item.get('phone', '')
                    password = item.get('password', '')
                    if phone and password:
                        accounts.append((phone, password))
                
                return accounts
                
            except ValueError as e:
                # 解密失败（可能是在其他机器上）
                print(f"  [账号文件] 解密失败: {e}")
                print(f"  [账号文件] 提示：账号文件可能是在其他机器上创建的，将尝试读取明文文件")
                # 尝试读取明文文件
                return self._read_plain_file()
            except Exception as e:
                print(f"  [账号文件] 加载加密文件失败: {e}")
                # 尝试读取明文文件
                return self._read_plain_file()
        
        # 兼容旧版本：读取明文文件
        return self._read_plain_file()
    
    def _read_plain_file(self) -> List[Tuple[str, str]]:
        """读取明文账号文件（兼容旧版本）
        
        Returns:
            账号列表 [(phone, password), ...]
        """
        if not self.accounts_file.exists():
            return []
        
        accounts = []
        try:
            with open(self.accounts_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    if '----' in line:
                        parts = line.split('----', 1)
                        phone = parts[0].strip()
                        password = parts[1].strip() if len(parts) > 1 else ""
                        
                        if phone and password:
                            accounts.append((phone, password))
        except Exception as e:
            print(f"  [账号文件] 读取明文文件失败: {e}")
        
        return accounts
    
    def write_accounts(self, accounts: List[Tuple[str, str]]) -> bool:
        """写入账号列表（自动加密）
        
        Args:
            accounts: 账号列表 [(phone, password), ...]
            
        Returns:
            是否成功
        """
        try:
            # 转换为JSON格式
            accounts_data = []
            for phone, password in accounts:
                accounts_data.append({
                    'phone': phone,
                    'password': password
                })
            
            json_data = json.dumps(accounts_data, ensure_ascii=False, indent=2)
            
            # 加密
            encrypted_data = self.crypto.encrypt_with_machine_binding(json_data.encode('utf-8'))
            
            # 确保目录存在
            self.encrypted_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存加密文件
            with open(self.encrypted_file, 'wb') as f:
                f.write(encrypted_data)
            
            # 删除旧的明文文件（如果存在）
            if self.accounts_file.exists():
                self.accounts_file.unlink()
            
            return True
            
        except Exception as e:
            print(f"  [账号文件] 保存失败: {e}")
            return False
    
    def append_accounts(self, new_accounts: List[Tuple[str, str]]) -> bool:
        """追加账号到文件（自动加密）
        
        Args:
            new_accounts: 新账号列表 [(phone, password), ...]
            
        Returns:
            是否成功
        """
        # 读取现有账号
        existing_accounts = self.read_accounts()
        
        # 合并账号（去重）
        existing_phones = {phone for phone, _ in existing_accounts}
        for phone, password in new_accounts:
            if phone not in existing_phones:
                existing_accounts.append((phone, password))
                existing_phones.add(phone)
        
        # 写入所有账号
        return self.write_accounts(existing_accounts)
    
    def delete_accounts(self, phones_to_delete: List[str]) -> bool:
        """删除指定账号（自动加密）
        
        Args:
            phones_to_delete: 要删除的手机号列表
            
        Returns:
            是否成功
        """
        # 读取现有账号
        existing_accounts = self.read_accounts()
        
        # 过滤掉要删除的账号
        phones_set = set(phones_to_delete)
        remaining_accounts = [(phone, password) for phone, password in existing_accounts 
                             if phone not in phones_set]
        
        # 写入剩余账号
        return self.write_accounts(remaining_accounts)
    
    def clear_accounts(self) -> bool:
        """清空所有账号
        
        Returns:
            是否成功
        """
        return self.write_accounts([])
    
    def upgrade_to_encrypted(self) -> bool:
        """升级明文文件为加密文件
        
        Returns:
            是否成功
        """
        if not self.accounts_file.exists():
            print(f"  [账号文件] 明文文件不存在，无需升级")
            return False
        
        if self.encrypted_file.exists():
            print(f"  [账号文件] 加密文件已存在，无需升级")
            return False
        
        # 读取明文文件
        accounts = self._read_plain_file()
        
        if not accounts:
            print(f"  [账号文件] 明文文件为空，无需升级")
            return False
        
        # 写入加密文件
        if self.write_accounts(accounts):
            print(f"  [账号文件] ✓ 已升级为加密文件（{len(accounts)} 个账号）")
            return True
        else:
            print(f"  [账号文件] ✗ 升级失败")
            return False
    
    def has_encrypted_file(self) -> bool:
        """检查是否有加密文件
        
        Returns:
            是否有加密文件
        """
        return self.encrypted_file.exists()
    
    def has_plain_file(self) -> bool:
        """检查是否有明文文件
        
        Returns:
            是否有明文文件
        """
        return self.accounts_file.exists()


def migrate_accounts_file(accounts_file: str) -> bool:
    """迁移账号文件到加密格式
    
    Args:
        accounts_file: 账号文件路径
        
    Returns:
        是否成功
    """
    print("=" * 60)
    print("账号文件加密迁移")
    print("=" * 60)
    
    encrypted_file = EncryptedAccountsFile(accounts_file)
    
    # 检查文件状态
    has_plain = encrypted_file.has_plain_file()
    has_encrypted = encrypted_file.has_encrypted_file()
    
    print(f"\n文件状态:")
    print(f"  明文文件: {'存在' if has_plain else '不存在'}")
    print(f"  加密文件: {'存在' if has_encrypted else '不存在'}")
    
    if has_encrypted:
        print(f"\n✓ 账号文件已加密，无需迁移")
        return True
    
    if not has_plain:
        print(f"\n⚠️ 没有找到账号文件")
        return False
    
    # 执行升级
    print(f"\n正在升级账号文件...")
    
    if encrypted_file.upgrade_to_encrypted():
        print(f"\n✓ 迁移成功！")
        print(f"\n重要提示:")
        print(f"  1. 账号文件已加密，只能在当前机器使用")
        print(f"  2. 如果更换机器，需要重新导入账号")
        print(f"  3. 明文文件已自动删除")
        return True
    else:
        print(f"\n✗ 迁移失败")
        return False


if __name__ == '__main__':
    # 测试迁移
    import sys
    if len(sys.argv) > 1:
        accounts_file = sys.argv[1]
    else:
        accounts_file = "data/账号详情.xlsx"  # 默认账号文件
    
    migrate_accounts_file(accounts_file)
