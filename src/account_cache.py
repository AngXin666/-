"""
账号信息缓存模块 - 避免重复 OCR 识别
Account Cache Module - Avoid Repeated OCR Recognition

v2.0.7 更新：
- 添加机器绑定加密，防止缓存文件被复制到其他机器使用
- 缓存文件使用机器ID加密，只能在当前机器上解密
"""

import json
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime


class AccountCache:
    """账号信息缓存管理器（带机器绑定加密）
    
    功能：
    - 缓存已识别的用户信息（昵称、用户ID）
    - 避免重复 OCR 识别，提升性能
    - 持久化存储到本地文件（加密）
    - 机器绑定：缓存只能在当前机器使用
    """
    
    def __init__(self, cache_file: str = ".account_cache.json"):
        """初始化缓存管理器
        
        Args:
            cache_file: 缓存文件路径（不含 .enc 后缀）
        """
        self.cache_file = Path(cache_file)
        self.encrypted_cache_file = Path(str(cache_file) + '.enc')
        self._cache: Dict[str, Dict] = {}
        
        # 导入加密工具
        try:
            from .crypto_utils import CryptoUtils
        except ImportError:
            try:
                from crypto_utils import CryptoUtils
            except ImportError:
                from src.crypto_utils import CryptoUtils
        self.crypto = CryptoUtils()
        
        self._load_cache()
    
    def _load_cache(self):
        """从文件加载缓存（自动解密）"""
        # 优先加载加密文件
        if self.encrypted_cache_file.exists():
            try:
                with open(self.encrypted_cache_file, 'rb') as f:
                    encrypted_data = f.read()
                
                # 解密
                plain_data = self.crypto.decrypt_with_machine_binding(encrypted_data)
                self._cache = json.loads(plain_data.decode('utf-8'))
                # 静默加载成功
                return
            except ValueError as e:
                # 解密失败（可能是在其他机器上）
                print(f"  [缓存] 解密失败: {e}")
                print(f"  [缓存] 提示：缓存可能是在其他机器上创建的，将创建新缓存")
                self._cache = {}
                return
            except Exception as e:
                # 其他错误
                print(f"  [缓存] 加载加密缓存失败: {e}")
                self._cache = {}
                return
        
        # 兼容旧版本：加载未加密的文件
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self._cache = json.load(f)
                # 静默加载成功
                # 下次保存时会自动加密
                return
            except Exception as e:
                # 静默记录错误
                self._cache = {}
        else:
            self._cache = {}
    
    def _save_cache(self):
        """保存缓存到文件（自动加密）"""
        try:
            # 转为JSON
            json_data = json.dumps(self._cache, ensure_ascii=False, indent=2)
            
            # 加密
            encrypted_data = self.crypto.encrypt_with_machine_binding(json_data.encode('utf-8'))
            
            # 保存加密文件
            with open(self.encrypted_cache_file, 'wb') as f:
                f.write(encrypted_data)
            
            # 删除旧的未加密文件（如果存在）
            if self.cache_file.exists():
                self.cache_file.unlink()
                
        except Exception as e:
            print(f"  [缓存] 保存缓存失败: {e}")
    
    def get(self, phone: str) -> Optional[Dict]:
        """获取账号缓存信息
        
        Args:
            phone: 手机号
            
        Returns:
            dict: 缓存的账号信息，包含：
                - nickname: 昵称
                - user_id: 用户ID
                - last_updated: 最后更新时间
            未找到返回 None
        """
        return self._cache.get(phone)
    
    def set(self, phone: str, nickname: Optional[str] = None, user_id: Optional[str] = None):
        """设置账号缓存信息
        
        Args:
            phone: 手机号
            nickname: 昵称（可选）
            user_id: 用户ID（可选）
        """
        if phone not in self._cache:
            self._cache[phone] = {}
        
        # 更新非空字段
        if nickname is not None:
            self._cache[phone]['nickname'] = nickname
        if user_id is not None:
            self._cache[phone]['user_id'] = user_id
        
        # 更新时间戳
        self._cache[phone]['last_updated'] = datetime.now().isoformat()
        
        # 保存到文件
        self._save_cache()
    
    def has_complete_info(self, phone: str) -> bool:
        """检查账号是否有完整的缓存信息
        
        Args:
            phone: 手机号
            
        Returns:
            bool: 是否有完整信息（昵称和用户ID都存在）
        """
        cache = self.get(phone)
        if not cache:
            return False
        
        return cache.get('nickname') is not None and cache.get('user_id') is not None
    
    def get_nickname(self, phone: str) -> Optional[str]:
        """获取缓存的昵称
        
        Args:
            phone: 手机号
            
        Returns:
            str: 昵称，未找到返回 None
        """
        cache = self.get(phone)
        return cache.get('nickname') if cache else None
    
    def get_user_id(self, phone: str) -> Optional[str]:
        """获取缓存的用户ID
        
        Args:
            phone: 手机号
            
        Returns:
            str: 用户ID，未找到返回 None
        """
        cache = self.get(phone)
        return cache.get('user_id') if cache else None
    
    def clear(self, phone: Optional[str] = None):
        """清空缓存
        
        Args:
            phone: 手机号（可选），如果提供则只清空该账号的缓存，否则清空所有
        """
        if phone:
            if phone in self._cache:
                del self._cache[phone]
                self._save_cache()
                print(f"  [缓存] 已清空账号 {phone} 的缓存")
        else:
            self._cache = {}
            self._save_cache()
            print(f"  [缓存] 已清空所有缓存")
    
    def get_stats(self) -> Dict:
        """获取缓存统计信息
        
        Returns:
            dict: 统计信息
                - total: 总缓存数
                - complete: 完整信息数（昵称和ID都有）
                - partial: 部分信息数（只有昵称或ID）
        """
        total = len(self._cache)
        complete = sum(1 for cache in self._cache.values() 
                      if cache.get('nickname') and cache.get('user_id'))
        partial = total - complete
        
        return {
            'total': total,
            'complete': complete,
            'partial': partial
        }
    
    def print_stats(self):
        """打印缓存统计信息"""
        stats = self.get_stats()
        print(f"\n  [缓存统计]")
        print(f"  - 总缓存数: {stats['total']}")
        print(f"  - 完整信息: {stats['complete']}")
        print(f"  - 部分信息: {stats['partial']}")


# 全局单例实例
_account_cache = None


def get_account_cache(cache_file: str = ".account_cache.json") -> AccountCache:
    """获取全局账号缓存实例（单例模式）
    
    Args:
        cache_file: 缓存文件路径
        
    Returns:
        AccountCache: 全局单例实例
    """
    global _account_cache
    if _account_cache is None:
        _account_cache = AccountCache(cache_file)
    return _account_cache
