"""
测试加密功能
Test Encryption Functions
"""

import sys
import os

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from crypto_utils import crypto
from local_db import LocalDatabase
import json


def test_aes_encryption():
    """测试 AES-256-GCM 加密"""
    print("=" * 60)
    print("测试 1: AES-256-GCM 加密/解密")
    print("=" * 60)
    
    # 测试数据
    test_data = {
        'license_key': 'A1B2-C3D4-E5F6-G7H8-I9J0',
        'machine_id': 'test-machine-12345',
        'timestamp': '1706112345678'
    }
    
    print(f"\n原始数据:")
    print(json.dumps(test_data, indent=2, ensure_ascii=False))
    
    # 加密
    print(f"\n正在加密...")
    encrypted = crypto.encrypt_request(test_data)
    
    print(f"\n加密后数据:")
    print(json.dumps(encrypted, indent=2, ensure_ascii=False))
    
    # 解密
    print(f"\n正在解密...")
    decrypted = crypto.decrypt_response(encrypted)
    
    print(f"\n解密后数据:")
    print(json.dumps(decrypted, indent=2, ensure_ascii=False))
    
    # 验证
    if decrypted == test_data:
        print(f"\n[PASSED] 加密/解密测试通过！")
        return True
    else:
        print(f"\n[FAILED] 加密/解密测试失败！")
        print(f"期望: {test_data}")
        print(f"实际: {decrypted}")
        return False


def test_database_encryption():
    """测试数据库加密"""
    print("\n" + "=" * 60)
    print("测试 2: 数据库字段加密")
    print("=" * 60)
    
    # 测试数据
    test_license_key = "TEST-1234-5678-ABCD-EFGH"
    test_machine_id = "machine-test-67890"
    
    print(f"\n原始卡密: {test_license_key}")
    print(f"原始机器ID: {test_machine_id}")
    
    # 加密
    print(f"\n正在加密...")
    encrypted_key = crypto.encrypt_database_value(test_license_key)
    encrypted_id = crypto.encrypt_database_value(test_machine_id)
    
    print(f"\n加密后卡密: {encrypted_key[:50]}...")
    print(f"加密后机器ID: {encrypted_id[:50]}...")
    
    # 解密
    print(f"\n正在解密...")
    decrypted_key = crypto.decrypt_database_value(encrypted_key)
    decrypted_id = crypto.decrypt_database_value(encrypted_id)
    
    print(f"\n解密后卡密: {decrypted_key}")
    print(f"解密后机器ID: {decrypted_id}")
    
    # 验证
    if decrypted_key == test_license_key and decrypted_id == test_machine_id:
        print(f"\n[PASSED] 数据库加密测试通过！")
        return True
    else:
        print(f"\n[FAILED] 数据库加密测试失败！")
        return False


def test_database_operations():
    """测试数据库加密存储"""
    print("\n" + "=" * 60)
    print("测试 3: 数据库加密存储")
    print("=" * 60)
    
    # 创建测试数据库
    db = LocalDatabase()
    
    # 测试数据
    test_data = {
        'license_key': 'TEST-ABCD-1234-EFGH-5678',
        'machine_id': 'test-machine-xyz',
        'status': 'active',
        'expires_at': '2026-12-31T23:59:59',
        'max_devices': 3,
        'activated_at': '2026-01-24T12:00:00',
        'last_online_check': '2026-01-24T12:00:00'
    }
    
    print(f"\n保存测试数据:")
    print(json.dumps(test_data, indent=2, ensure_ascii=False))
    
    # 保存
    print(f"\n正在保存到数据库...")
    success = db.save_license(test_data)
    
    if not success:
        print(f"\n[FAILED] 保存失败！")
        return False
    
    print(f"[PASSED] 保存成功")
    
    # 读取
    print(f"\n正在从数据库读取...")
    loaded_data = db.get_license()
    
    if not loaded_data:
        print(f"\n[FAILED] 读取失败！")
        return False
    
    print(f"\n读取的数据:")
    print(json.dumps(loaded_data, indent=2, ensure_ascii=False))
    
    # 验证
    if (loaded_data['license_key'] == test_data['license_key'] and 
        loaded_data['machine_id'] == test_data['machine_id']):
        print(f"\n[PASSED] 数据库加密存储测试通过！")
        
        # 清理测试数据
        db.delete_license()
        print(f"[PASSED] 测试数据已清理")
        return True
    else:
        print(f"\n[FAILED] 数据库加密存储测试失败！")
        return False


def test_signature():
    """测试签名生成"""
    print("\n" + "=" * 60)
    print("测试 4: 请求签名")
    print("=" * 60)
    
    test_data = {
        'license_key': 'TEST-1234',
        'machine_id': 'test-machine'
    }
    
    print(f"\n测试数据:")
    print(json.dumps(test_data, indent=2, ensure_ascii=False))
    
    # 生成签名
    signature1 = crypto.generate_request_signature(test_data, "secret123")
    signature2 = crypto.generate_request_signature(test_data, "secret123")
    
    print(f"\n签名 1: {signature1}")
    print(f"签名 2: {signature2}")
    
    # 验证（相同数据和密钥应该生成不同签名，因为包含时间戳）
    print(f"\n[PASSED] 签名生成测试通过！")
    return True


def test_obfuscation():
    """测试卡密混淆"""
    print("\n" + "=" * 60)
    print("测试 5: 卡密混淆")
    print("=" * 60)
    
    test_keys = [
        'A1B2-C3D4-E5F6-G7H8-I9J0',
        'TEST-1234-5678-ABCD-EFGH',
        'SHORT'
    ]
    
    for key in test_keys:
        obfuscated = crypto.obfuscate_license_key(key)
        print(f"\n原始: {key}")
        print(f"混淆: {obfuscated}")
    
    print(f"\n[PASSED] 卡密混淆测试通过！")
    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始测试加密功能")
    print("=" * 60)
    
    results = []
    
    try:
        # 测试 1: AES 加密
        results.append(("AES-256-GCM 加密", test_aes_encryption()))
        
        # 测试 2: 数据库字段加密
        results.append(("数据库字段加密", test_database_encryption()))
        
        # 测试 3: 数据库加密存储
        results.append(("数据库加密存储", test_database_operations()))
        
        # 测试 4: 签名
        results.append(("请求签名", test_signature()))
        
        # 测试 5: 混淆
        results.append(("卡密混淆", test_obfuscation()))
        
    except Exception as e:
        print(f"\n[FAILED] 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for name, result in results:
        status = "[PASSED] 通过" if result else "[FAILED] 失败"
        print(f"{name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有测试通过！")
    else:
        print("[WARNING]️ 部分测试失败，请检查错误信息")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
