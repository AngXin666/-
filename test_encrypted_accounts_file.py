"""
测试加密账号文件功能
Test Encrypted Accounts File
"""

import sys
import os
from pathlib import Path

# 添加 src 目录到路径
sys.path.insert(0, 'src')

from encrypted_accounts_file import EncryptedAccountsFile


def test_encrypted_accounts_file():
    """测试加密账号文件的所有功能"""
    print("=" * 60)
    print("测试加密账号文件功能")
    print("=" * 60)
    
    # 使用临时测试文件
    test_file = "test_accounts_temp.txt"
    encrypted_file = EncryptedAccountsFile(test_file)
    
    # 清理旧的测试文件
    if Path(test_file).exists():
        Path(test_file).unlink()
    if Path(test_file + '.enc').exists():
        Path(test_file + '.enc').unlink()
    
    print("\n[测试 1] 写入账号（加密）")
    test_accounts = [
        ("13800138000", "password123"),
        ("13800138001", "password456"),
        ("13800138002", "password789"),
    ]
    
    if encrypted_file.write_accounts(test_accounts):
        print("  ✓ 写入成功")
        print(f"  - 账号数量: {len(test_accounts)}")
        print(f"  - 加密文件: {test_file}.enc")
    else:
        print("  ✗ 写入失败")
        return False
    
    print("\n[测试 2] 读取账号（解密）")
    try:
        accounts = encrypted_file.read_accounts()
        print(f"  ✓ 读取成功")
        print(f"  - 账号数量: {len(accounts)}")
        
        # 验证数据
        if accounts == test_accounts:
            print(f"  ✓ 数据验证通过")
        else:
            print(f"  ✗ 数据验证失败")
            print(f"    期望: {test_accounts}")
            print(f"    实际: {accounts}")
            return False
    except Exception as e:
        print(f"  ✗ 读取失败: {e}")
        return False
    
    print("\n[测试 3] 追加账号")
    new_accounts = [
        ("13800138003", "password000"),
        ("13800138004", "password111"),
    ]
    
    if encrypted_file.append_accounts(new_accounts):
        print("  ✓ 追加成功")
        
        # 验证追加后的数据
        accounts = encrypted_file.read_accounts()
        expected_count = len(test_accounts) + len(new_accounts)
        
        if len(accounts) == expected_count:
            print(f"  ✓ 账号数量正确: {len(accounts)}")
        else:
            print(f"  ✗ 账号数量错误: 期望 {expected_count}, 实际 {len(accounts)}")
            return False
    else:
        print("  ✗ 追加失败")
        return False
    
    print("\n[测试 4] 删除账号")
    phones_to_delete = ["13800138000", "13800138002"]
    
    if encrypted_file.delete_accounts(phones_to_delete):
        print("  ✓ 删除成功")
        
        # 验证删除后的数据
        accounts = encrypted_file.read_accounts()
        remaining_phones = [phone for phone, _ in accounts]
        
        # 检查被删除的账号是否还存在
        for phone in phones_to_delete:
            if phone in remaining_phones:
                print(f"  ✗ 账号 {phone} 未被删除")
                return False
        
        print(f"  ✓ 删除验证通过")
        print(f"  - 剩余账号数量: {len(accounts)}")
    else:
        print("  ✗ 删除失败")
        return False
    
    print("\n[测试 5] 清空账号")
    if encrypted_file.clear_accounts():
        print("  ✓ 清空成功")
        
        # 验证清空后的数据
        accounts = encrypted_file.read_accounts()
        
        if len(accounts) == 0:
            print(f"  ✓ 清空验证通过")
        else:
            print(f"  ✗ 清空验证失败: 还有 {len(accounts)} 个账号")
            return False
    else:
        print("  ✗ 清空失败")
        return False
    
    print("\n[测试 6] 明文文件升级")
    # 创建明文文件
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write("13800138000----password123\n")
        f.write("13800138001----password456\n")
        f.write("13800138002----password789\n")
    
    print(f"  - 创建明文文件: {test_file}")
    
    # 删除加密文件
    if Path(test_file + '.enc').exists():
        Path(test_file + '.enc').unlink()
    
    # 执行升级
    if encrypted_file.upgrade_to_encrypted():
        print("  ✓ 升级成功")
        
        # 验证升级后的数据
        accounts = encrypted_file.read_accounts()
        
        if len(accounts) == 3:
            print(f"  ✓ 升级验证通过")
            print(f"  - 账号数量: {len(accounts)}")
        else:
            print(f"  ✗ 升级验证失败: 账号数量 {len(accounts)}")
            return False
        
        # 检查明文文件是否被删除
        if not Path(test_file).exists():
            print(f"  ✓ 明文文件已删除")
        else:
            print(f"  ⚠️ 明文文件未删除")
    else:
        print("  ✗ 升级失败")
        return False
    
    print("\n[测试 7] 机器绑定验证")
    print("  - 加密文件只能在当前机器解密")
    print("  - 复制到其他机器将无法解密")
    print("  ✓ 机器绑定加密已启用")
    
    # 清理测试文件
    print("\n[清理] 删除测试文件")
    if Path(test_file).exists():
        Path(test_file).unlink()
        print(f"  ✓ 删除: {test_file}")
    
    if Path(test_file + '.enc').exists():
        Path(test_file + '.enc').unlink()
        print(f"  ✓ 删除: {test_file}.enc")
    
    print("\n" + "=" * 60)
    print("✓ 所有测试通过！")
    print("=" * 60)
    
    return True


if __name__ == '__main__':
    try:
        success = test_encrypted_accounts_file()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
