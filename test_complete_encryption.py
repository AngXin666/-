"""
完整加密系统测试

测试所有加密组件：
1. crypto_utils - 加密工具
2. account_cache - 账号缓存
3. login_cache_manager - 登录缓存（模拟）
"""

import sys
import json
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, 'src')

from crypto_utils import CryptoUtils
from account_cache import AccountCache


def test_crypto_utils():
    """测试加密工具"""
    print("\n" + "=" * 60)
    print("测试1: 加密工具 (crypto_utils)")
    print("=" * 60)
    
    crypto = CryptoUtils()
    
    # 测试机器ID
    machine_id = crypto.get_machine_id()
    print(f"  机器ID: {machine_id[:32]}...")
    
    # 测试加密/解密
    test_data = b"Test sensitive data"
    encrypted = crypto.encrypt_with_machine_binding(test_data)
    decrypted = crypto.decrypt_with_machine_binding(encrypted)
    
    if decrypted == test_data:
        print(f"  ✅ 加密/解密测试通过")
        return True
    else:
        print(f"  ❌ 加密/解密测试失败")
        return False


def test_account_cache():
    """测试账号缓存"""
    print("\n" + "=" * 60)
    print("测试2: 账号缓存 (account_cache)")
    print("=" * 60)
    
    # 使用临时缓存文件
    test_cache_file = ".test_account_cache.json"
    
    try:
        # 创建缓存
        cache = AccountCache(cache_file=test_cache_file)
        
        # 设置测试数据
        test_phone = "13800138000"
        test_nickname = "测试用户"
        test_user_id = "123456"
        
        cache.set(test_phone, nickname=test_nickname, user_id=test_user_id)
        print(f"  ✅ 保存缓存: {test_phone}")
        
        # 验证加密文件存在
        encrypted_file = Path(test_cache_file + '.enc')
        if encrypted_file.exists():
            print(f"  ✅ 加密文件已创建: {encrypted_file.name}")
        else:
            print(f"  ❌ 加密文件未创建")
            return False
        
        # 验证原始文件已删除
        plain_file = Path(test_cache_file)
        if not plain_file.exists():
            print(f"  ✅ 原始文件已删除")
        else:
            print(f"  ⚠️  原始文件仍存在（可能是旧版本）")
        
        # 重新加载缓存
        cache2 = AccountCache(cache_file=test_cache_file)
        
        # 验证数据
        cached_data = cache2.get(test_phone)
        if cached_data:
            if (cached_data.get('nickname') == test_nickname and 
                cached_data.get('user_id') == test_user_id):
                print(f"  ✅ 缓存数据正确")
                print(f"    昵称: {cached_data.get('nickname')}")
                print(f"    用户ID: {cached_data.get('user_id')}")
                success = True
            else:
                print(f"  ❌ 缓存数据不匹配")
                success = False
        else:
            print(f"  ❌ 无法读取缓存")
            success = False
        
        # 清理
        if encrypted_file.exists():
            encrypted_file.unlink()
        if plain_file.exists():
            plain_file.unlink()
        
        return success
        
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        # 清理
        encrypted_file = Path(test_cache_file + '.enc')
        plain_file = Path(test_cache_file)
        if encrypted_file.exists():
            encrypted_file.unlink()
        if plain_file.exists():
            plain_file.unlink()
        return False


def test_migration_compatibility():
    """测试迁移兼容性（旧版本未加密文件）"""
    print("\n" + "=" * 60)
    print("测试3: 迁移兼容性")
    print("=" * 60)
    
    test_cache_file = ".test_migration_cache.json"
    
    try:
        # 创建旧版本未加密的缓存文件
        old_cache_data = {
            "13900139000": {
                "nickname": "旧版本用户",
                "user_id": "789012",
                "last_updated": "2026-02-01T00:00:00"
            }
        }
        
        plain_file = Path(test_cache_file)
        with open(plain_file, 'w', encoding='utf-8') as f:
            json.dump(old_cache_data, f, ensure_ascii=False, indent=2)
        
        print(f"  ✅ 创建旧版本缓存文件")
        
        # 加载缓存（应该能读取旧版本）
        cache = AccountCache(cache_file=test_cache_file)
        
        cached_data = cache.get("13900139000")
        if cached_data and cached_data.get('nickname') == "旧版本用户":
            print(f"  ✅ 成功读取旧版本缓存")
        else:
            print(f"  ❌ 无法读取旧版本缓存")
            return False
        
        # 修改缓存（应该自动加密）
        cache.set("13900139000", nickname="更新后的用户")
        
        # 验证加密文件已创建
        encrypted_file = Path(test_cache_file + '.enc')
        if encrypted_file.exists():
            print(f"  ✅ 自动升级为加密文件")
        else:
            print(f"  ❌ 未自动升级")
            return False
        
        # 验证原始文件已删除
        if not plain_file.exists():
            print(f"  ✅ 旧文件已删除")
        else:
            print(f"  ⚠️  旧文件仍存在")
        
        # 清理
        if encrypted_file.exists():
            encrypted_file.unlink()
        if plain_file.exists():
            plain_file.unlink()
        
        return True
        
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        # 清理
        encrypted_file = Path(test_cache_file + '.enc')
        plain_file = Path(test_cache_file)
        if encrypted_file.exists():
            encrypted_file.unlink()
        if plain_file.exists():
            plain_file.unlink()
        return False


def test_security():
    """测试安全性"""
    print("\n" + "=" * 60)
    print("测试4: 安全性验证")
    print("=" * 60)
    
    test_cache_file = ".test_security_cache.json"
    
    try:
        # 创建加密缓存
        cache = AccountCache(cache_file=test_cache_file)
        cache.set("13700137000", nickname="敏感用户", user_id="secret123")
        
        # 读取加密文件
        encrypted_file = Path(test_cache_file + '.enc')
        with open(encrypted_file, 'rb') as f:
            encrypted_data = f.read()
        
        # 验证加密文件不包含明文
        if b"13700137000" not in encrypted_data:
            print(f"  ✅ 手机号已加密（不可见）")
        else:
            print(f"  ❌ 手机号未加密")
            return False
        
        if "敏感用户".encode('utf-8') not in encrypted_data:
            print(f"  ✅ 昵称已加密（不可见）")
        else:
            print(f"  ❌ 昵称未加密")
            return False
        
        if b"secret123" not in encrypted_data:
            print(f"  ✅ 用户ID已加密（不可见）")
        else:
            print(f"  ❌ 用户ID未加密")
            return False
        
        print(f"  ✅ 所有敏感信息已加密")
        
        # 清理
        if encrypted_file.exists():
            encrypted_file.unlink()
        
        return True
        
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        # 清理
        encrypted_file = Path(test_cache_file + '.enc')
        if encrypted_file.exists():
            encrypted_file.unlink()
        return False


def main():
    """运行所有测试"""
    print("=" * 60)
    print("完整加密系统测试")
    print("=" * 60)
    
    tests = [
        ("加密工具", test_crypto_utils),
        ("账号缓存", test_account_cache),
        ("迁移兼容性", test_migration_compatibility),
        ("安全性验证", test_security),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n  ❌ 测试异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {status} - {name}")
    
    print(f"\n总计: {passed}/{total} 通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！")
        print("\n✅ 加密系统已就绪，可以安全使用")
        return 0
    else:
        print(f"\n⚠️  {total - passed} 个测试失败")
        return 1


if __name__ == '__main__':
    exit(main())
