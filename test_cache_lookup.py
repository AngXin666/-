"""
测试缓存查找功能
模拟 LoginCacheManager 的缓存查找逻辑
"""

from pathlib import Path
from typing import Optional

class TestCacheManager:
    """测试用的缓存管理器（模拟 LoginCacheManager 的逻辑）"""
    
    REQUIRED_FILES = [
        "shared_prefs/lcdpr.xml",
        "databases/DCStorage"
    ]
    
    def __init__(self, cache_dir: str = "login_cache"):
        self.cache_dir = Path(cache_dir)
        self.phone_userid_map_file = self.cache_dir / "phone_userid_mapping.txt"
    
    def _get_expected_user_id(self, phone: str) -> Optional[str]:
        """获取手机号对应的预期用户ID（模拟实际代码）"""
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
        """获取账号的缓存目录（模拟实际代码）"""
        if user_id:
            return self.cache_dir / f"{phone}_{user_id}"
        else:
            return self.cache_dir / phone
    
    def _get_cache_file_path(self, account_cache_dir: Path, cache_file_name: str, encrypted: bool = True) -> Path:
        """获取缓存文件路径（模拟实际代码）"""
        base_path = account_cache_dir / cache_file_name
        if encrypted:
            return Path(str(base_path) + '.enc')
        return base_path
    
    def has_cache(self, phone: str, user_id: Optional[str] = None) -> bool:
        """检查是否有缓存（模拟实际代码）"""
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


def main():
    print("=" * 80)
    print("测试缓存查找功能")
    print("=" * 80)
    
    # 创建测试管理器
    manager = TestCacheManager()
    
    # 1. 获取所有缓存目录
    print("\n【1. 扫描缓存目录】")
    cache_dir = Path("login_cache")
    
    cache_dirs = []
    for item in cache_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            if '_' in item.name:
                cache_dirs.append(item)
    
    print(f"✓ 找到 {len(cache_dirs)} 个缓存目录")
    
    # 2. 测试前10个账号的缓存查找
    print("\n【2. 测试缓存查找（前10个账号）】")
    
    test_accounts = []
    for cache_dir_item in cache_dirs[:10]:
        dir_name = cache_dir_item.name
        if '_' in dir_name:
            phone, user_id = dir_name.split('_', 1)
            test_accounts.append((phone, user_id, cache_dir_item))
    
    success_count = 0
    failed_count = 0
    
    for i, (phone, user_id, cache_dir_item) in enumerate(test_accounts):
        # 步骤1：从映射文件获取 expected_user_id
        expected_user_id = manager._get_expected_user_id(phone)
        
        # 步骤2：使用 expected_user_id 检查缓存
        has_cache = manager.has_cache(phone, expected_user_id)
        
        # 验证结果
        cache_exists = cache_dir_item.exists()
        
        status = "✓" if has_cache else "✗"
        result = "成功" if has_cache == cache_exists else "失败"
        
        print(f"  {i+1}. {phone}")
        print(f"     映射中的user_id: {expected_user_id}")
        print(f"     实际user_id: {user_id}")
        print(f"     has_cache(): {has_cache}")
        print(f"     缓存目录存在: {cache_exists}")
        print(f"     {status} 查找{result}")
        
        if has_cache == cache_exists:
            success_count += 1
        else:
            failed_count += 1
    
    print(f"\n测试结果：")
    print(f"  ✓ 成功: {success_count}/{len(test_accounts)}")
    print(f"  ✗ 失败: {failed_count}/{len(test_accounts)}")
    
    # 3. 随机测试20个账号
    print("\n【3. 随机测试20个账号】")
    
    import random
    random_accounts = random.sample(cache_dirs, min(20, len(cache_dirs)))
    
    random_success = 0
    random_failed = 0
    failed_list = []
    
    for cache_dir_item in random_accounts:
        dir_name = cache_dir_item.name
        if '_' in dir_name:
            phone, user_id = dir_name.split('_', 1)
            
            expected_user_id = manager._get_expected_user_id(phone)
            has_cache = manager.has_cache(phone, expected_user_id)
            cache_exists = cache_dir_item.exists()
            
            if has_cache == cache_exists:
                random_success += 1
            else:
                random_failed += 1
                failed_list.append((phone, expected_user_id, user_id))
    
    print(f"  ✓ 成功: {random_success}/{len(random_accounts)}")
    print(f"  ✗ 失败: {random_failed}/{len(random_accounts)}")
    
    if failed_list:
        print(f"\n  失败的账号：")
        for phone, expected_id, actual_id in failed_list:
            print(f"    - {phone}: 映射={expected_id}, 实际={actual_id}")
    
    # 4. 测试特定场景
    print("\n【4. 测试特定场景】")
    
    # 场景1：映射文件中有，缓存目录也有
    print("\n  场景1：映射文件中有记录，缓存目录存在")
    if test_accounts:
        phone, user_id, _ = test_accounts[0]
        expected_user_id = manager._get_expected_user_id(phone)
        has_cache = manager.has_cache(phone, expected_user_id)
        print(f"    账号: {phone}")
        print(f"    映射中的user_id: {expected_user_id}")
        print(f"    has_cache(): {has_cache}")
        print(f"    结果: {'✓ 正确' if has_cache else '✗ 错误'}")
    
    # 场景2：映射文件中没有，但缓存目录存在（不应该发生）
    print("\n  场景2：映射文件中没有记录，但缓存目录存在")
    test_phone = "99999999999"  # 不存在的手机号
    expected_user_id = manager._get_expected_user_id(test_phone)
    has_cache = manager.has_cache(test_phone, expected_user_id)
    print(f"    账号: {test_phone}")
    print(f"    映射中的user_id: {expected_user_id}")
    print(f"    has_cache(): {has_cache}")
    print(f"    结果: {'✓ 正确（应该返回False）' if not has_cache else '✗ 错误'}")
    
    # 5. 总结
    print("\n" + "=" * 80)
    print("【测试总结】")
    print("=" * 80)
    
    total_tests = len(test_accounts) + len(random_accounts) + 2
    total_success = success_count + random_success + (1 if test_accounts and manager.has_cache(test_accounts[0][0], manager._get_expected_user_id(test_accounts[0][0])) else 0) + (1 if not manager.has_cache("99999999999", None) else 0)
    
    print(f"总测试数: {total_tests}")
    print(f"成功: {total_success}")
    print(f"失败: {total_tests - total_success}")
    
    if total_success == total_tests:
        print("\n✅ 所有测试通过！缓存查找功能正常工作。")
    else:
        print(f"\n⚠️ 有 {total_tests - total_success} 个测试失败，缓存查找可能存在问题。")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
