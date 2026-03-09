"""
登录缓存管理器
用于保存和恢复应用的登录缓存文件，实现快速登录

v2.0.7 更新：
- 添加机器绑定加密，防止缓存文件被复制到其他机器使用
- 缓存文件使用机器ID加密，只能在当前机器上解密
"""

import asyncio
import os
from pathlib import Path
from typing import Optional
from datetime import datetime


class LoginCacheManager:
    """登录缓存管理器（带机器绑定加密）"""
    
    # 需要缓存的文件列表
    CACHE_FILES = [
        "shared_prefs/lcdpr.xml",
        "databases/DCStorage",
        "databases/DCStorage-shm",  # SQLite临时文件，可能不存在
        "databases/DCStorage-wal"   # SQLite临时文件，可能不存在
    ]
    
    # 必须存在的文件（用于验证缓存完整性）
    REQUIRED_FILES = [
        "shared_prefs/lcdpr.xml",
        "databases/DCStorage"
    ]
    
    def __init__(self, adb_bridge, cache_dir: str = "login_cache"):
        """初始化登录缓存管理器
        
        Args:
            adb_bridge: ADB 桥接器实例
            cache_dir: 本地缓存目录
        """
        self.adb = adb_bridge
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # 全局的手机号-用户ID映射文件
        self.phone_userid_map_file = self.cache_dir / "phone_userid_mapping.txt"
        
        # 导入加密工具
        try:
            from .crypto_utils import CryptoUtils
        except ImportError:
            from crypto_utils import CryptoUtils
        self.crypto = CryptoUtils()
    
    def _encrypt_cache_file(self, file_path: Path) -> bool:
        """加密缓存文件（机器绑定）
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否成功
        """
        try:
            # 检查文件是否存在
            if not file_path.exists():
                # 文件不存在，可能已被删除，静默跳过
                return False
            
            # 读取原始文件
            with open(file_path, 'rb') as f:
                plain_data = f.read()
            
            # 使用机器绑定加密
            encrypted_data = self.crypto.encrypt_with_machine_binding(plain_data)
            
            # 写入加密文件（添加 .enc 后缀）
            encrypted_file = Path(str(file_path) + '.enc')
            with open(encrypted_file, 'wb') as f:
                f.write(encrypted_data)
            
            # 删除原始文件
            file_path.unlink()
            
            return True
            
        except FileNotFoundError:
            # 文件不存在，静默跳过
            return False
        except Exception as e:
            print(f"  [加密] 加密文件失败 {file_path.name}: {e}")
            return False
    
    def _decrypt_cache_file(self, encrypted_file: Path) -> Optional[Path]:
        """解密缓存文件
        
        Args:
            encrypted_file: 加密文件路径（.enc）
            
        Returns:
            解密后的临时文件路径，失败返回 None
        """
        try:
            # 读取加密文件
            with open(encrypted_file, 'rb') as f:
                encrypted_data = f.read()
            
            # 解密
            plain_data = self.crypto.decrypt_with_machine_binding(encrypted_data)
            
            # 写入临时文件（去掉 .enc 后缀）
            temp_file = Path(str(encrypted_file)[:-4])  # 去掉 .enc
            with open(temp_file, 'wb') as f:
                f.write(plain_data)
            
            return temp_file
            
        except ValueError as e:
            # 解密失败（可能是在其他机器上）
            print(f"  [解密] 解密失败 {encrypted_file.name}: {e}")
            print(f"  [解密] 提示：缓存文件可能是在其他机器上创建的，无法在当前机器使用")
            return None
        except Exception as e:
            print(f"  [解密] 解密文件失败 {encrypted_file.name}: {e}")
            return None
    
    def _is_encrypted_file(self, file_path: Path) -> bool:
        """检查文件是否已加密
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否已加密
        """
        return file_path.suffix == '.enc'
    
    def _get_cache_file_path(self, account_cache_dir: Path, cache_file_name: str, encrypted: bool = True) -> Path:
        """获取缓存文件路径
        
        Args:
            account_cache_dir: 账号缓存目录
            cache_file_name: 缓存文件名（如 shared_prefs_lcdpr.xml）
            encrypted: 是否返回加密文件路径
            
        Returns:
            文件路径
        """
        base_path = account_cache_dir / cache_file_name
        if encrypted:
            return Path(str(base_path) + '.enc')
        return base_path
    
    def _save_phone_userid_mapping(self, phone: str, user_id: str):
        """保存手机号和用户ID的映射关系
        
        Args:
            phone: 手机号
            user_id: 用户ID
        """
        # 读取现有映射
        mappings = {}
        if self.phone_userid_map_file.exists():
            with open(self.phone_userid_map_file, "r", encoding="utf-8") as f:
                for line in f:
                    if '=' in line:
                        p, uid = line.strip().split('=', 1)
                        mappings[p] = uid
        
        # 更新映射
        mappings[phone] = user_id
        
        # 保存映射
        with open(self.phone_userid_map_file, "w", encoding="utf-8") as f:
            for p, uid in mappings.items():
                f.write(f"{p}={uid}\n")
    
    def _get_expected_user_id(self, phone: str) -> Optional[str]:
        """获取手机号对应的预期用户ID
        
        Args:
            phone: 手机号
            
        Returns:
            用户ID，如果不存在返回 None
        """
        if not self.phone_userid_map_file.exists():
            return None
        
        with open(self.phone_userid_map_file, "r", encoding="utf-8") as f:
            for line in f:
                if '=' in line:
                    p, uid = line.strip().split('=', 1)
                    if p == phone:
                        return uid
        
        return None
    
    def _get_account_cache_dir(self, phone: str, user_id: Optional[str] = None) -> Path:
        """获取账号的缓存目录
        
        Args:
            phone: 手机号
            user_id: 用户ID（可选），如果提供则使用 手机号_用户ID 作为目录名
            
        Returns:
            缓存目录路径
        """
        if user_id:
            # 使用 手机号_用户ID 作为目录名，避免同一手机号不同账号的缓存冲突
            return self.cache_dir / f"{phone}_{user_id}"
        else:
            # 兼容旧版本：只使用手机号
            return self.cache_dir / phone
    
    async def save_login_cache(self, device_id: str, phone: str, 
                               package_name: str = "com.ry.xmsc", user_id: str = None) -> bool:
        """保存登录缓存（直接加密保存）
        
        Args:
            device_id: 设备 ID
            phone: 手机号（用于标识缓存）
            package_name: 应用包名
            user_id: 用户ID（用于验证缓存是否匹配，强烈建议提供）
            
        Returns:
            是否成功
        """
        try:
            # 创建账号缓存目录（使用 手机号_用户ID 作为目录名）
            account_cache_dir = self._get_account_cache_dir(phone, user_id)
            account_cache_dir.mkdir(parents=True, exist_ok=True)  # 创建所有必需的父目录
            
            print(f"  [缓存] 保存到目录: {account_cache_dir}")
            
            data_path = f"/data/data/{package_name}"
            saved_count = 0
            
            # 保存每个文件
            for file_path in self.CACHE_FILES:
                source_path = f"{data_path}/{file_path}"
                
                # 检查文件是否存在
                result = await self.adb.shell(
                    device_id, 
                    f"su -c 'test -f {source_path} && echo EXISTS || echo NOT_EXISTS'"
                )
                
                if "EXISTS" not in result:
                    # SQLite临时文件可能不存在，这是正常的
                    if file_path not in self.REQUIRED_FILES:
                        continue
                    else:
                        print(f"  [缓存] ⚠️ 必需文件不存在: {file_path}")
                        continue
                
                # 拉取文件到临时位置
                cache_file_name = file_path.replace('/', '_')
                temp_plain_file = account_cache_dir / f"temp_{cache_file_name}"
                temp_path = f"/sdcard/temp_{cache_file_name}"
                
                try:
                    # 先复制到 sdcard（不需要 root）
                    await self.adb.shell(device_id, f"su -c 'cp {source_path} {temp_path}'")
                    
                    # 确保临时文件的父目录存在
                    temp_plain_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # 从 sdcard 拉取到本地临时文件
                    await self.adb.pull(device_id, temp_path, str(temp_plain_file))
                    
                    # 删除 sdcard 上的临时文件
                    await self.adb.shell(device_id, f"rm {temp_path}")
                    
                    # 读取临时文件并加密
                    with open(temp_plain_file, 'rb') as f:
                        plain_data = f.read()
                    
                    # 使用机器绑定加密
                    encrypted_data = self.crypto.encrypt_with_machine_binding(plain_data)
                    
                    # 写入加密文件（.enc 后缀）
                    encrypted_file = account_cache_dir / f"{cache_file_name}.enc"
                    with open(encrypted_file, 'wb') as f:
                        f.write(encrypted_data)
                    
                    # 删除临时明文文件
                    temp_plain_file.unlink()
                    
                    print(f"  [缓存] ✓ 已保存并加密: {file_path}")
                    saved_count += 1
                    
                except Exception as e:
                    print(f"  [缓存] ✗ 保存失败 {file_path}: {e}")
                    # 清理临时文件
                    if temp_plain_file.exists():
                        try:
                            temp_plain_file.unlink()
                        except:
                            pass
            
            # 保存元数据（包含用户ID）
            metadata_file = account_cache_dir / "metadata.txt"
            with open(metadata_file, "w", encoding="utf-8") as f:
                f.write(f"phone={phone}\n")
                f.write(f"package={package_name}\n")
                f.write(f"saved_at={datetime.now().isoformat()}\n")
                f.write(f"files_count={saved_count}\n")
                if user_id:
                    f.write(f"user_id={user_id}\n")
            
            # 保存手机号-用户ID映射（用于后续验证）
            if user_id:
                self._save_phone_userid_mapping(phone, user_id)
            
            return saved_count > 0
            
        except Exception as e:
            print(f"保存登录缓存失败: {e}")
            return False
    
    async def restore_login_cache(self, device_id: str, phone: str,
                                  package_name: str = "com.ry.xmsc", user_id: Optional[str] = None) -> bool:
        """恢复登录缓存（直接读取加密文件，临时解密使用）
        
        Args:
            device_id: 设备 ID
            phone: 手机号（用于标识缓存）
            package_name: 应用包名
            user_id: 用户ID（可选），如果提供则优先使用 手机号_用户ID 查找缓存
            
        Returns:
            是否成功
        """
        try:
            # 优先尝试使用 手机号_用户ID 查找缓存
            account_cache_dir = None
            if user_id:
                account_cache_dir = self._get_account_cache_dir(phone, user_id)
                if not account_cache_dir.exists():
                    print(f"  [缓存] 未找到 {phone}_{user_id} 的缓存，尝试使用旧格式")
                    account_cache_dir = None
            
            # 如果没有user_id或新格式不存在，尝试旧格式（只用手机号）
            if account_cache_dir is None:
                account_cache_dir = self._get_account_cache_dir(phone)
                if not account_cache_dir.exists():
                    print(f"  [缓存] 未找到 {phone} 的缓存")
                    return False
            
            print(f"  [缓存] 从目录恢复: {account_cache_dir}")
            
            # 先停止应用
            await self.adb.shell(device_id, f"am force-stop {package_name}")
            await asyncio.sleep(0.5)  # 优化：减少等待时间从1秒到0.5秒
            
            data_path = f"/data/data/{package_name}"
            restored_count = 0
            
            # 优化：批量准备所有文件
            files_to_restore = []  # [(local_file, file_path, temp_decrypted_file), ...]
            
            # 第一步：解密所有文件（准备阶段）
            for file_path in self.CACHE_FILES:
                cache_file_name = file_path.replace('/', '_')
                
                # 优先使用加密文件（.enc）
                encrypted_file = self._get_cache_file_path(account_cache_dir, cache_file_name, encrypted=True)
                plain_file = self._get_cache_file_path(account_cache_dir, cache_file_name, encrypted=False)
                
                local_file = None
                temp_decrypted_file = None
                
                # 检查加密文件是否存在
                if encrypted_file.exists():
                    # 临时解密到内存
                    temp_decrypted_file = self._decrypt_cache_file(encrypted_file)
                    if temp_decrypted_file:
                        local_file = temp_decrypted_file
                    else:
                        print(f"  [缓存] ⚠️ 解密失败: {file_path}")
                        continue
                # 如果加密文件不存在，尝试未加密文件（兼容旧数据）
                elif plain_file.exists():
                    local_file = plain_file
                else:
                    # 文件不存在
                    if file_path not in self.REQUIRED_FILES:
                        continue
                    else:
                        print(f"  [缓存] ⚠️ 必需文件不存在: {file_path}")
                        continue
                
                files_to_restore.append((local_file, file_path, temp_decrypted_file))
            
            if not files_to_restore:
                print(f"  [缓存] 没有文件需要恢复")
                return False
            
            # 第二步：批量传输所有文件到 sdcard
            temp_files = []
            for local_file, file_path, _ in files_to_restore:
                temp_path = f"/sdcard/temp_{file_path.replace('/', '_')}"
                temp_files.append((temp_path, file_path))
                await self.adb.push(device_id, str(local_file), temp_path)
            
            # 第三步：批量复制到应用目录（一次性执行多个命令）
            # 先获取应用的 UID（只需要一次）
            uid_result = await self.adb.shell(device_id, f"su -c 'stat -c %u {data_path}'")
            uid = uid_result.strip()
            
            # 构建批量命令
            batch_commands = []
            for temp_path, file_path in temp_files:
                target_path = f"{data_path}/{file_path}"
                target_dir = target_path.rsplit('/', 1)[0]
                
                # 创建目录、删除旧文件、复制、设置权限和所有者
                batch_commands.append(f"mkdir -p {target_dir}")
                batch_commands.append(f"rm -f {target_path}")
                batch_commands.append(f"cp {temp_path} {target_path}")
                batch_commands.append(f"chmod 660 {target_path}")
                batch_commands.append(f"chown {uid}:{uid} {target_path}")
            
            # 一次性执行所有命令（用 && 连接）
            batch_command = " && ".join(batch_commands)
            await self.adb.shell(device_id, f"su -c '{batch_command}'")
            
            # 第四步：清理临时文件
            cleanup_commands = [f"rm {temp_path}" for temp_path, _ in temp_files]
            cleanup_command = " && ".join(cleanup_commands)
            await self.adb.shell(device_id, cleanup_command)
            
            # 清理临时解密文件
            for _, _, temp_decrypted_file in files_to_restore:
                if temp_decrypted_file and temp_decrypted_file.exists():
                    try:
                        temp_decrypted_file.unlink()
                    except:
                        pass
            
            restored_count = len(files_to_restore)
            print(f"  [缓存] ✓ 已恢复 {restored_count} 个文件")
            
            return restored_count > 0
            
        except Exception as e:
            print(f"恢复登录缓存失败: {e}")
            # 清理临时解密文件
            if 'files_to_restore' in locals():
                for _, _, temp_decrypted_file in files_to_restore:
                    if temp_decrypted_file and temp_decrypted_file.exists():
                        try:
                            temp_decrypted_file.unlink()
                        except:
                            pass
            return False
    
    def has_cache(self, phone: str, user_id: Optional[str] = None) -> bool:
        """检查是否有缓存（只检查加密文件）
        
        Args:
            phone: 手机号
            user_id: 用户ID（可选），如果提供则优先检查 手机号_用户ID 的缓存
            
        Returns:
            是否有缓存
        """
        # 优先检查新格式（手机号_用户ID）
        if user_id:
            account_cache_dir = self._get_account_cache_dir(phone, user_id)
            if account_cache_dir.exists():
                # 检查是否至少有一个必需的加密缓存文件
                for file_path in self.REQUIRED_FILES:
                    encrypted_file = self._get_cache_file_path(account_cache_dir, file_path.replace('/', '_'), encrypted=True)
                    if encrypted_file.exists():
                        return True
        
        # 检查旧格式（只用手机号）
        account_cache_dir = self._get_account_cache_dir(phone)
        if not account_cache_dir.exists():
            return False
        
        # 检查是否至少有一个必需的加密缓存文件
        for file_path in self.REQUIRED_FILES:
            encrypted_file = self._get_cache_file_path(account_cache_dir, file_path.replace('/', '_'), encrypted=True)
            if encrypted_file.exists():
                return True
        
        return False
    
    def get_cache_info(self, phone: str, user_id: Optional[str] = None) -> Optional[dict]:
        """获取缓存信息
        
        Args:
            phone: 手机号
            user_id: 用户ID（可选），如果提供则优先查找 手机号_用户ID 的缓存
            
        Returns:
            缓存信息字典，如果不存在返回 None
        """
        # 优先查找新格式
        if user_id:
            account_cache_dir = self._get_account_cache_dir(phone, user_id)
            metadata_file = account_cache_dir / "metadata.txt"
            if metadata_file.exists():
                info = {}
                with open(metadata_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if '=' in line:
                            key, value = line.strip().split('=', 1)
                            info[key] = value
                return info
        
        # 查找旧格式
        account_cache_dir = self._get_account_cache_dir(phone)
        metadata_file = account_cache_dir / "metadata.txt"
        
        if not metadata_file.exists():
            return None
        
        info = {}
        with open(metadata_file, "r", encoding="utf-8") as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    info[key] = value
        
        return info
    
    def delete_cache(self, phone: str, user_id: Optional[str] = None) -> bool:
        """删除缓存
        
        Args:
            phone: 手机号
            user_id: 用户ID（可选），如果提供则删除 手机号_用户ID 的缓存
            
        Returns:
            是否成功
        """
        try:
            # 如果提供了user_id，删除新格式的缓存
            if user_id:
                account_cache_dir = self._get_account_cache_dir(phone, user_id)
                if account_cache_dir.exists():
                    import shutil
                    shutil.rmtree(account_cache_dir)
                    print(f"  [缓存] 已删除 {phone}_{user_id} 的缓存")
                    return True
            
            # 删除旧格式的缓存
            account_cache_dir = self._get_account_cache_dir(phone)
            if account_cache_dir.exists():
                import shutil
                shutil.rmtree(account_cache_dir)
                print(f"  [缓存] 已删除 {phone} 的缓存")
            return True
        except Exception as e:
            print(f"删除缓存失败: {e}")
            return False

    async def clear_app_login_data(self, device_id: str, package_name: str = "com.ry.xmsc") -> bool:
        """清理应用中的登录数据（不清理整个应用数据）
        
        用于在没有缓存时，清理应用中旧账号的登录数据，
        避免启动应用后自动登录到错误的账号。
        
        优化：合并删除命令，从4次ADB通信减少到1次
        
        Args:
            device_id: 设备 ID
            package_name: 应用包名
            
        Returns:
            是否成功
        """
        try:
            data_path = f"/data/data/{package_name}"
            
            # 优化：合并所有删除命令为一个命令，减少ADB通信次数
            # 从4次 shell 调用 → 1次 shell 调用
            target_paths = [f"{data_path}/{file_path}" for file_path in self.CACHE_FILES]
            combined_cmd = f"su -c 'rm -f {' '.join(target_paths)}'"
            await self.adb.shell(device_id, combined_cmd)
            
            return True
            
        except Exception as e:
            print(f"清理应用登录数据失败: {e}")
            return False
    
    def decrypt_all_caches(self) -> int:
        """程序启动时解密所有缓存文件（多线程并行）
        
        Returns:
            解密的文件数量
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        decrypted_count = 0
        
        try:
            # 遍历所有账号缓存目录
            if not self.cache_dir.exists():
                return 0
            
            # 收集所有需要解密的文件
            files_to_decrypt = []
            
            for account_dir in self.cache_dir.iterdir():
                if not account_dir.is_dir():
                    continue
                
                # 跳过特殊文件
                if account_dir.name.startswith('.'):
                    continue
                
                # 收集该账号的所有加密文件
                for file_path in self.CACHE_FILES:
                    cache_file_name = file_path.replace('/', '_')
                    encrypted_file = self._get_cache_file_path(account_dir, cache_file_name, encrypted=True)
                    
                    if encrypted_file.exists():
                        files_to_decrypt.append((account_dir.name, cache_file_name, encrypted_file))
            
            if not files_to_decrypt:
                return 0
            
            # 使用线程池并行解密
            with ThreadPoolExecutor(max_workers=8) as executor:
                # 提交所有解密任务
                future_to_file = {
                    executor.submit(self._decrypt_cache_file, encrypted_file): (account_name, cache_name, encrypted_file)
                    for account_name, cache_name, encrypted_file in files_to_decrypt
                }
                
                # 等待所有任务完成
                for future in as_completed(future_to_file):
                    account_name, cache_name, encrypted_file = future_to_file[future]
                    try:
                        decrypted_file = future.result()
                        if decrypted_file:
                            # 删除加密文件
                            encrypted_file.unlink()
                            decrypted_count += 1
                            print(f"  [启动] ✓ 已解密: {account_name}/{cache_name}")
                        else:
                            print(f"  [启动] ✗ 解密失败: {account_name}/{cache_name}")
                    except Exception as e:
                        print(f"  [启动] ✗ 解密出错: {account_name}/{cache_name}: {e}")
            
            if decrypted_count > 0:
                print(f"  [启动] 共解密 {decrypted_count} 个缓存文件")
            
            return decrypted_count
            
        except Exception as e:
            print(f"  [启动] 解密缓存失败: {e}")
            return decrypted_count
    
    def encrypt_all_caches(self) -> int:
        """程序关闭时加密所有缓存文件（多线程并行）
        
        Returns:
            加密的文件数量
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        encrypted_count = 0
        
        try:
            # 遍历所有账号缓存目录
            if not self.cache_dir.exists():
                return 0
            
            # 收集所有需要加密的文件
            files_to_encrypt = []
            
            for account_dir in self.cache_dir.iterdir():
                if not account_dir.is_dir():
                    continue
                
                # 跳过特殊文件
                if account_dir.name.startswith('.'):
                    continue
                
                # 收集该账号的所有未加密文件
                for file_path in self.CACHE_FILES:
                    cache_file_name = file_path.replace('/', '_')
                    plain_file = self._get_cache_file_path(account_dir, cache_file_name, encrypted=False)
                    
                    # 只收集实际存在的文件
                    if plain_file.exists():
                        files_to_encrypt.append((account_dir.name, cache_file_name, plain_file))
            
            if not files_to_encrypt:
                return 0
            
            # 使用线程池并行加密
            with ThreadPoolExecutor(max_workers=8) as executor:
                # 提交所有加密任务
                future_to_file = {
                    executor.submit(self._encrypt_cache_file, plain_file): (account_name, cache_name, plain_file)
                    for account_name, cache_name, plain_file in files_to_encrypt
                }
                
                # 等待所有任务完成
                for future in as_completed(future_to_file):
                    account_name, cache_name, plain_file = future_to_file[future]
                    try:
                        success = future.result()
                        if success:
                            encrypted_count += 1
                            print(f"  [关闭] ✓ 已加密: {account_name}/{cache_name}")
                        # 如果失败但不报错，说明文件不存在，静默跳过
                    except Exception as e:
                        print(f"  [关闭] ✗ 加密出错: {account_name}/{cache_name}: {e}")
            
            if encrypted_count > 0:
                print(f"  [关闭] 共加密 {encrypted_count} 个缓存文件")
            
            return encrypted_count
            
        except Exception as e:
            print(f"  [关闭] 加密缓存失败: {e}")
            return encrypted_count
