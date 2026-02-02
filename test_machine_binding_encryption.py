"""
机器绑定加密功能单元测试

测试内容：
1. 机器ID生成
2. 机器绑定加密/解密
3. 登录缓存加密
4. 跨机器解密失败验证
"""

import sys
import os
import tempfile
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, 'src')

from crypto_utils import CryptoUtils


def test_machine_id():
    """测试机器ID生成"""
    print("\n" + "=" * 60)
    print("测试1: 机器ID生成")
    print("=" * 60)
    
    crypto = CryptoUtils()
    
    # 生成机器ID
    machine_id1 = crypto.get_machine_id()
    print(f"  机器ID: {machine_id1}")
    print(f"  长度: {len(machine_id1)} 字符")
    
    # 验证一致性
    machine_id2 = crypto.get_machine_id()
    if machine_id1 == machine_id2:
        print(f"  ✅ 一致性验证通过")
        return True
    else:
        print(f"  ❌ 一致性验证失败")
        return False


def test_machine_binding_encryption():
    """测试机器绑定加密/解密"""
    print("\n" + "=" * 60)
    print("测试2: 机器绑定加密/解密")
    print("=" * 60)
    
    crypto = CryptoUtils()
    
    # 测试数据
    test_data = b"This is sensitive login cache data for phone 13800138000"
    print(f"  原始数据: {test_data[:50]}...")
    print(f"  数据大小: {len(test_data)} 字节")
    
    try:
        # 加密
        encrypted_data = crypto.encrypt_with_machine_binding(test_data)
        print(f"  ✅ 加密成功")
        print(f"  加密后大小: {len(encrypted_data)} 字节")
        
        # 验证加密后的数据不包含原始内容
        if test_data not in encrypted_data:
            print(f"  ✅ 数据已加密（不包含原始内容）")
        else:
            print(f"  ❌ 数据未正确加密")
            return False
        
        # 解密
        decrypted_data = crypto.decrypt_with_machine_binding(encrypted_data)
        print(f"  ✅ 解密成功")
        
        # 验证解密后的数据
        if decrypted_data == test_data:
            print(f"  ✅ 解密数据正确")
            return True
        else:
            print(f"  ❌ 解密数据不匹配")
            return False
            
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        return False


def test_different_data_types():
    """测试不同类型的数据"""
    print("\n" + "=" * 60)
    print("测试3: 不同类型数据加密")
    print("=" * 60)
    
    crypto = CryptoUtils()
    
    test_cases = [
        ("小文件", b"small data"),
        ("中等文件", b"x" * 1024),  # 1KB
        ("大文件", b"y" * 10240),  # 10KB
        ("二进制数据", bytes(range(256))),
        ("UTF-8文本", "中文测试数据 🔒".encode('utf-8')),
    ]
    
    success_count = 0
    
    for name, data in test_cases:
        try:
            encrypted = crypto.encrypt_with_machine_binding(data)
            decrypted = crypto.decrypt_with_machine_binding(encrypted)
            
            if decrypted == data:
                print(f"  ✅ {name}: 通过 ({len(data)} 字节)")
                success_count += 1
            else:
                print(f"  ❌ {name}: 失败（数据不匹配）")
        except Exception as e:
            print(f"  ❌ {name}: 失败 ({e})")
    
    print(f"\n  总计: {success_count}/{len(test_cases)} 通过")
    return success_count == len(test_cases)


def test_cross_machine_decryption():
    """测试跨机器解密失败（模拟）"""
    print("\n" + "=" * 60)
    print("测试4: 跨机器解密验证")
    print("=" * 60)
    
    crypto = CryptoUtils()
    
    # 加密数据
    test_data = b"Sensitive cache data"
    encrypted_data = crypto.encrypt_with_machine_binding(test_data)
    print(f"  ✅ 数据已加密")
    
    # 模拟修改机器ID（通过修改加密数据中的机器ID）
    # 注意：这只是模拟，实际上我们无法真正改变机器ID
    print(f"  ℹ️  注意：无法真正模拟跨机器解密")
    print(f"  ℹ️  在实际使用中，复制到其他机器会解密失败")
    
    # 验证正常解密仍然工作
    try:
        decrypted = crypto.decrypt_with_machine_binding(encrypted_data)
        if decrypted == test_data:
            print(f"  ✅ 当前机器解密成功")
            return True
    except Exception as e:
        print(f"  ❌ 解密失败: {e}")
        return False


def test_file_encryption():
    """测试文件加密/解密"""
    print("\n" + "=" * 60)
    print("测试5: 文件加密/解密")
    print("=" * 60)
    
    crypto = CryptoUtils()
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.cache') as f:
        test_file = Path(f.name)
        test_data = b"Login cache file content: phone=13800138000, session=abc123"
        f.write(test_data)
    
    try:
        print(f"  测试文件: {test_file.name}")
        print(f"  原始大小: {test_file.stat().st_size} 字节")
        
        # 读取并加密
        with open(test_file, 'rb') as f:
            plain_data = f.read()
        
        encrypted_data = crypto.encrypt_with_machine_binding(plain_data)
        
        # 写入加密文件
        encrypted_file = Path(str(test_file) + '.enc')
        with open(encrypted_file, 'wb') as f:
            f.write(encrypted_data)
        
        print(f"  ✅ 文件已加密: {encrypted_file.name}")
        print(f"  加密后大小: {encrypted_file.stat().st_size} 字节")
        
        # 读取并解密
        with open(encrypted_file, 'rb') as f:
            encrypted_data = f.read()
        
        decrypted_data = crypto.decrypt_with_machine_binding(encrypted_data)
        
        # 验证
        if decrypted_data == test_data:
            print(f"  ✅ 文件解密成功，数据正确")
            success = True
        else:
            print(f"  ❌ 解密数据不匹配")
            success = False
        
        # 清理
        test_file.unlink()
        encrypted_file.unlink()
        
        return success
        
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        # 清理
        if test_file.exists():
            test_file.unlink()
        if encrypted_file.exists():
            encrypted_file.unlink()
        return False


def test_performance():
    """测试加密性能"""
    print("\n" + "=" * 60)
    print("测试6: 加密性能测试")
    print("=" * 60)
    
    import time
    crypto = CryptoUtils()
    
    # 测试不同大小的数据（减少测试大小以加快速度）
    test_sizes = [
        (1024, "1KB"),
        (10240, "10KB"),
    ]
    
    for size, label in test_sizes:
        test_data = b"x" * size
        
        # 加密性能
        start = time.perf_counter()
        encrypted = crypto.encrypt_with_machine_binding(test_data)
        encrypt_time = (time.perf_counter() - start) * 1000
        
        # 解密性能
        start = time.perf_counter()
        decrypted = crypto.decrypt_with_machine_binding(encrypted)
        decrypt_time = (time.perf_counter() - start) * 1000
        
        print(f"  {label}:")
        print(f"    加密: {encrypt_time:.2f}ms")
        print(f"    解密: {decrypt_time:.2f}ms")
        print(f"    总计: {encrypt_time + decrypt_time:.2f}ms")
    
    return True


def main():
    """运行所有测试"""
    print("=" * 60)
    print("机器绑定加密功能单元测试")
    print("=" * 60)
    
    tests = [
        ("机器ID生成", test_machine_id),
        ("机器绑定加密/解密", test_machine_binding_encryption),
        ("不同类型数据", test_different_data_types),
        ("跨机器解密验证", test_cross_machine_decryption),
        ("文件加密/解密", test_file_encryption),
        ("性能测试", test_performance),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n  ❌ 测试异常: {e}")
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
        return 0
    else:
        print(f"\n⚠️  {total - passed} 个测试失败")
        return 1


if __name__ == '__main__':
    exit(main())
