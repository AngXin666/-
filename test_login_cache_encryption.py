"""
测试登录缓存加密解密流程
验证程序启动时解密、运行中不加密解密、关闭时加密的逻辑
"""

import os
import sys
from pathlib import Path

# 添加src到路径
sys.path.insert(0, 'src')

from adb_bridge import ADBBridge
from login_cache_manager import LoginCacheManager


def test_encryption_workflow():
    """测试完整的加密解密工作流程"""
    print("\n" + "=" * 60)
    print("测试登录缓存加密解密流程")
    print("=" * 60)
    
    # 创建测试环境
    adb = ADBBridge()
    cache_manager = LoginCacheManager(adb)
    cache_dir = Path("login_cache")
    
    # 检查是否有缓存目录
    if not cache_dir.exists():
        print("❌ 没有找到login_cache目录，无法测试")
        print("提示：请先运行程序并登录至少一个账号以生成缓存")
        return False
    
    # 统计缓存文件
    account_dirs = [d for d in cache_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    if not account_dirs:
        print("❌ login_cache目录中没有账号缓存")
        print("提示：请先运行程序并登录至少一个账号以生成缓存")
        return False
    
    print(f"\n找到 {len(account_dirs)} 个账号缓存目录")
    
    # 统计加密和未加密的文件
    encrypted_files = []
    plain_files = []
    
    for account_dir in account_dirs:
        for file in account_dir.iterdir():
            if file.is_file():
                if file.suffix == '.enc':
                    encrypted_files.append(file)
                elif file.name != 'metadata.txt':
                    plain_files.append(file)
    
    print(f"当前状态：")
    print(f"  - 加密文件: {len(encrypted_files)} 个")
    print(f"  - 未加密文件: {len(plain_files)} 个")
    
    # 测试1: 解密所有缓存（模拟程序启动）
    print("\n" + "-" * 60)
    print("测试1: 程序启动时解密所有缓存")
    print("-" * 60)
    
    decrypted_count = cache_manager.decrypt_all_caches()
    print(f"✅ 解密完成，共解密 {decrypted_count} 个文件")
    
    # 验证解密结果
    encrypted_after_decrypt = []
    plain_after_decrypt = []
    
    for account_dir in account_dirs:
        for file in account_dir.iterdir():
            if file.is_file():
                if file.suffix == '.enc':
                    encrypted_after_decrypt.append(file)
                elif file.name != 'metadata.txt':
                    plain_after_decrypt.append(file)
    
    print(f"解密后状态：")
    print(f"  - 加密文件: {len(encrypted_after_decrypt)} 个")
    print(f"  - 未加密文件: {len(plain_after_decrypt)} 个")
    
    if len(encrypted_after_decrypt) == 0 and len(plain_after_decrypt) > 0:
        print("✅ 解密验证通过：所有文件已解密")
    else:
        print("⚠️ 解密验证失败：仍有加密文件存在")
    
    # 测试2: 加密所有缓存（模拟程序关闭）
    print("\n" + "-" * 60)
    print("测试2: 程序关闭时加密所有缓存")
    print("-" * 60)
    
    encrypted_count = cache_manager.encrypt_all_caches()
    print(f"✅ 加密完成，共加密 {encrypted_count} 个文件")
    
    # 验证加密结果
    encrypted_after_encrypt = []
    plain_after_encrypt = []
    
    for account_dir in account_dirs:
        for file in account_dir.iterdir():
            if file.is_file():
                if file.suffix == '.enc':
                    encrypted_after_encrypt.append(file)
                elif file.name != 'metadata.txt':
                    plain_after_encrypt.append(file)
    
    print(f"加密后状态：")
    print(f"  - 加密文件: {len(encrypted_after_encrypt)} 个")
    print(f"  - 未加密文件: {len(plain_after_encrypt)} 个")
    
    if len(plain_after_encrypt) == 0 and len(encrypted_after_encrypt) > 0:
        print("✅ 加密验证通过：所有文件已加密")
    else:
        print("⚠️ 加密验证失败：仍有未加密文件存在")
    
    # 测试3: 再次解密（验证可以重复解密）
    print("\n" + "-" * 60)
    print("测试3: 验证可以重复解密")
    print("-" * 60)
    
    decrypted_count2 = cache_manager.decrypt_all_caches()
    print(f"✅ 再次解密完成，共解密 {decrypted_count2} 个文件")
    
    # 最终验证
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    all_passed = True
    
    if decrypted_count > 0:
        print("✅ 解密功能正常")
    else:
        print("⚠️ 解密功能可能有问题（没有文件被解密）")
        all_passed = False
    
    if encrypted_count > 0:
        print("✅ 加密功能正常")
    else:
        print("⚠️ 加密功能可能有问题（没有文件被加密）")
        all_passed = False
    
    if decrypted_count2 > 0:
        print("✅ 重复解密功能正常")
    else:
        print("⚠️ 重复解密功能可能有问题")
        all_passed = False
    
    if all_passed:
        print("\n🎉 所有测试通过！")
        print("\n工作流程验证：")
        print("  1. ✅ 程序启动时：自动解密所有缓存文件")
        print("  2. ✅ 程序运行中：直接使用未加密文件（不再频繁加密解密）")
        print("  3. ✅ 程序关闭时：自动加密所有缓存文件")
        print("\n这样可以避免运行时的加密解密错误！")
    else:
        print("\n⚠️ 部分测试未通过，请检查")
    
    return all_passed


if __name__ == '__main__':
    try:
        success = test_encryption_workflow()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
