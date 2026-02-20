#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试打包后的加密功能
"""

import sys
import os

# 添加src到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_machine_id():
    """测试获取机器ID"""
    print("=" * 60)
    print("测试获取机器ID")
    print("=" * 60)
    
    try:
        from src.crypto_utils import CryptoUtils
        crypto = CryptoUtils()
        
        print("\n正在获取机器ID...")
        machine_id = crypto.get_machine_id()
        
        print(f"✓ 机器ID: {machine_id}")
        print(f"✓ 长度: {len(machine_id)}")
        
        return True
    except Exception as e:
        print(f"✗ 获取机器ID失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_encrypt_decrypt():
    """测试加密解密"""
    print("\n" + "=" * 60)
    print("测试加密解密")
    print("=" * 60)
    
    try:
        from src.crypto_utils import CryptoUtils
        crypto = CryptoUtils()
        
        test_data = "测试数据：手机号----密码----归属人"
        print(f"\n原始数据: {test_data}")
        
        # 加密
        print("\n正在加密...")
        encrypted = crypto.encrypt_with_machine_binding(test_data.encode('utf-8'))
        print(f"✓ 加密成功，长度: {len(encrypted)}")
        
        # 解密
        print("\n正在解密...")
        decrypted = crypto.decrypt_with_machine_binding(encrypted)
        decrypted_str = decrypted.decode('utf-8')
        print(f"✓ 解密成功: {decrypted_str}")
        
        # 验证
        if decrypted_str == test_data:
            print("\n✓ 加密解密测试通过")
            return True
        else:
            print("\n✗ 加密解密测试失败：数据不匹配")
            return False
            
    except Exception as e:
        print(f"\n✗ 加密解密测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_accounts_file():
    """测试账号文件加密"""
    print("\n" + "=" * 60)
    print("测试账号文件加密")
    print("=" * 60)
    
    try:
        from src.encrypted_accounts_file import EncryptedAccountsFile
        import tempfile
        import os
        
        # 创建临时文件
        temp_file = os.path.join(tempfile.gettempdir(), "test_accounts.txt")
        print(f"\n临时文件: {temp_file}")
        
        # 创建加密文件管理器
        encrypted_file = EncryptedAccountsFile(temp_file)
        
        # 测试数据
        test_accounts = [
            ("13800138000", "password123"),
            ("13900139000", "password456"),
        ]
        
        # 写入
        print("\n正在写入账号...")
        if encrypted_file.write_accounts(test_accounts):
            print("✓ 写入成功")
        else:
            print("✗ 写入失败")
            return False
        
        # 读取
        print("\n正在读取账号...")
        accounts = encrypted_file.read_accounts()
        print(f"✓ 读取成功，账号数量: {len(accounts)}")
        
        # 验证
        if accounts == test_accounts:
            print("✓ 账号数据匹配")
            
            # 清理
            if os.path.exists(temp_file + '.enc'):
                os.remove(temp_file + '.enc')
            
            return True
        else:
            print("✗ 账号数据不匹配")
            return False
            
    except Exception as e:
        print(f"\n✗ 账号文件测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("开始测试打包后的加密功能...\n")
    
    results = []
    
    # 测试1：获取机器ID
    results.append(("获取机器ID", test_machine_id()))
    
    # 测试2：加密解密
    results.append(("加密解密", test_encrypt_decrypt()))
    
    # 测试3：账号文件
    results.append(("账号文件", test_accounts_file()))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n✓ 所有测试通过！")
        sys.exit(0)
    else:
        print("\n✗ 部分测试失败")
        sys.exit(1)
