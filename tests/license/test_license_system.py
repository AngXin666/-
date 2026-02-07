"""
测试许可证系统
Test License System - 完整流程测试
"""

import sys
import os

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from license_manager import LicenseManager
from local_db import LocalDatabase
import time


def test_machine_id():
    """测试机器ID生成"""
    print("=" * 60)
    print("测试 1: 机器ID生成")
    print("=" * 60)
    
    manager = LicenseManager()
    
    # 生成机器ID
    machine_id = manager.get_machine_id()
    
    print(f"\n机器ID: {machine_id}")
    print(f"长度: {len(machine_id)} 字符")
    
    # 验证一致性
    machine_id2 = manager.get_machine_id()
    
    if machine_id == machine_id2:
        print(f"\n[PASSED] 机器ID生成测试通过（一致性验证）")
        return True
    else:
        print(f"\n[FAILED] 机器ID不一致！")
        return False


def test_license_format_validation():
    """测试卡密格式验证"""
    print("\n" + "=" * 60)
    print("测试 2: 卡密格式验证")
    print("=" * 60)
    
    manager = LicenseManager()
    
    test_cases = [
        ("A1B2-C3D4-E5F6-G7H8-I9J0", True, "正确格式"),
        ("TEST-1234-5678-ABCD-EFGH", True, "正确格式"),
        ("ABCD1234EFGH5678IJKL", True, "无横线格式"),
        ("A1B2-C3D4-E5F6", False, "长度不足"),
        ("A1B2-C3D4-E5F6-G7H8-I9J0-EXTRA", False, "长度过长"),
        ("A1B2-C3D4-E5F6-G7H8-I9J!", False, "包含非法字符"),
        ("", False, "空字符串"),
    ]
    
    all_passed = True
    
    for key, expected, desc in test_cases:
        valid, msg = manager.validate_license_key(key)
        status = "[PASSED]" if valid == expected else "[FAILED]"
        
        print(f"\n{status} {desc}")
        print(f"   卡密: {key if key else '(空)'}")
        print(f"   期望: {'有效' if expected else '无效'}")
        print(f"   实际: {'有效' if valid else '无效'}")
        if not valid:
            print(f"   消息: {msg}")
        
        if valid != expected:
            all_passed = False
    
    if all_passed:
        print(f"\n[PASSED] 卡密格式验证测试通过")
    else:
        print(f"\n[FAILED] 卡密格式验证测试失败")
    
    return all_passed


def test_local_database():
    """测试本地数据库操作"""
    print("\n" + "=" * 60)
    print("测试 3: 本地数据库操作")
    print("=" * 60)
    
    db = LocalDatabase()
    
    # 清理旧数据
    db.delete_license()
    
    # 测试数据
    test_data = {
        'license_key': 'TEST-ABCD-1234-EFGH-5678',
        'machine_id': 'test-machine-xyz-123',
        'status': 'active',
        'expires_at': '2026-12-31T23:59:59',
        'max_devices': 3,
        'activated_at': '2026-01-24T12:00:00',
        'last_online_check': '2026-01-24T12:00:00'
    }
    
    print(f"\n1. 保存许可证数据...")
    success = db.save_license(test_data)
    
    if not success:
        print(f"[FAILED] 保存失败")
        return False
    
    print(f"[PASSED] 保存成功")
    
    print(f"\n2. 读取许可证数据...")
    loaded = db.get_license()
    
    if not loaded:
        print(f"[FAILED] 读取失败")
        return False
    
    print(f"[PASSED] 读取成功")
    
    print(f"\n3. 验证数据完整性...")
    if (loaded['license_key'] == test_data['license_key'] and
        loaded['machine_id'] == test_data['machine_id'] and
        loaded['status'] == test_data['status']):
        print(f"[PASSED] 数据完整性验证通过")
    else:
        print(f"[FAILED] 数据不匹配")
        return False
    
    print(f"\n4. 更新检查时间...")
    db.update_last_check('local')
    db.update_last_check('online')
    print(f"[PASSED] 更新成功")
    
    print(f"\n5. 删除许可证...")
    success = db.delete_license()
    
    if not success:
        print(f"[FAILED] 删除失败")
        return False
    
    print(f"[PASSED] 删除成功")
    
    print(f"\n6. 验证删除...")
    loaded = db.get_license()
    
    if loaded is None:
        print(f"[PASSED] 删除验证通过")
    else:
        print(f"[FAILED] 数据仍然存在")
        return False
    
    print(f"\n[PASSED] 本地数据库操作测试通过")
    return True


def test_license_check():
    """测试许可证检查"""
    print("\n" + "=" * 60)
    print("测试 4: 许可证检查")
    print("=" * 60)
    
    manager = LicenseManager()
    db = LocalDatabase()
    
    # 清理旧数据
    db.delete_license()
    
    print(f"\n1. 检查未激活状态...")
    valid, msg, info = manager.check_license()
    
    if not valid and msg == "未激活":
        print(f"[PASSED] 未激活状态正确")
    else:
        print(f"[FAILED] 未激活状态错误: {msg}")
        return False
    
    print(f"\n2. 模拟激活...")
    machine_id = manager.get_machine_id()
    test_data = {
        'license_key': 'TEST-1234-5678-ABCD-EFGH',
        'machine_id': machine_id,
        'status': 'active',
        'expires_at': '2026-12-31T23:59:59',
        'max_devices': 1,
        'activated_at': '2026-01-24T12:00:00',
        'last_online_check': '2026-01-24T12:00:00'
    }
    
    db.save_license(test_data)
    print(f"[PASSED] 激活数据已保存")
    
    print(f"\n3. 检查激活状态...")
    valid, msg, info = manager.check_license()
    
    if valid:
        print(f"[PASSED] 激活状态正确: {msg}")
        print(f"   卡密: {info.get('license_key', 'N/A')}")
        print(f"   状态: {info.get('status', 'N/A')}")
        print(f"   设备限制: {info.get('max_devices', 'N/A')}")
    else:
        print(f"[FAILED] 激活状态错误: {msg}")
        db.delete_license()
        return False
    
    print(f"\n4. 测试机器ID不匹配...")
    test_data['machine_id'] = 'wrong-machine-id'
    db.save_license(test_data)
    
    valid, msg, info = manager.check_license()
    
    if not valid and "不匹配" in msg:
        print(f"[PASSED] 机器ID验证正确")
    else:
        print(f"[FAILED] 机器ID验证失败: {msg}")
        db.delete_license()
        return False
    
    print(f"\n5. 测试禁用状态...")
    test_data['machine_id'] = machine_id
    test_data['status'] = 'disabled'
    db.save_license(test_data)
    
    valid, msg, info = manager.check_license()
    
    if not valid and "禁用" in msg:
        print(f"[PASSED] 禁用状态验证正确")
    else:
        print(f"[FAILED] 禁用状态验证失败: {msg}")
        db.delete_license()
        return False
    
    print(f"\n6. 测试过期...")
    test_data['status'] = 'active'
    test_data['expires_at'] = '2020-01-01T00:00:00'
    db.save_license(test_data)
    
    valid, msg, info = manager.check_license()
    
    if not valid and "过期" in msg:
        print(f"[PASSED] 过期验证正确")
    else:
        print(f"[FAILED] 过期验证失败: {msg}")
        db.delete_license()
        return False
    
    # 清理
    db.delete_license()
    
    print(f"\n[PASSED] 许可证检查测试通过")
    return True


def test_license_info():
    """测试许可证信息获取"""
    print("\n" + "=" * 60)
    print("测试 5: 许可证信息获取")
    print("=" * 60)
    
    manager = LicenseManager()
    db = LocalDatabase()
    
    # 准备测试数据
    machine_id = manager.get_machine_id()
    test_data = {
        'license_key': 'INFO-TEST-1234-5678-ABCD',
        'machine_id': machine_id,
        'status': 'active',
        'expires_at': '2026-12-31T23:59:59',
        'max_devices': 5,
        'activated_at': '2026-01-24T12:00:00',
        'last_online_check': '2026-01-24T12:00:00'
    }
    
    db.save_license(test_data)
    
    print(f"\n获取许可证信息...")
    info = manager.get_license_info()
    
    if info:
        print(f"\n许可证信息:")
        print(f"  卡密: {info.get('license_key', 'N/A')}")
        print(f"  激活时间: {info.get('activated_at', 'N/A')}")
        print(f"  过期时间: {info.get('expires_at', 'N/A')}")
        print(f"  剩余天数: {info.get('days_left', 'N/A')}")
        print(f"  状态: {info.get('status', 'N/A')}")
        print(f"  设备限制: {info.get('max_devices', 'N/A')}")
        
        if info['license_key'] == test_data['license_key']:
            print(f"\n[PASSED] 许可证信息获取测试通过")
            db.delete_license()
            return True
        else:
            print(f"\n[FAILED] 许可证信息不匹配")
            db.delete_license()
            return False
    else:
        print(f"\n[FAILED] 无法获取许可证信息")
        db.delete_license()
        return False


def test_encryption_in_database():
    """测试数据库中的加密"""
    print("\n" + "=" * 60)
    print("测试 6: 数据库加密验证")
    print("=" * 60)
    
    import sqlite3
    from pathlib import Path
    
    db = LocalDatabase()
    
    # 保存测试数据
    test_data = {
        'license_key': 'ENCRYPT-TEST-1234-ABCD',
        'machine_id': 'encrypt-machine-test',
        'status': 'active',
        'expires_at': '2026-12-31T23:59:59',
        'max_devices': 1,
        'activated_at': '2026-01-24T12:00:00',
        'last_online_check': '2026-01-24T12:00:00'
    }
    
    db.save_license(test_data)
    
    # 直接读取数据库文件
    db_path = Path("runtime_data") / "license.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    cursor.execute("SELECT license_key_encrypted, machine_id_encrypted FROM license LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    
    if row:
        encrypted_key = row[0]
        encrypted_id = row[1]
        
        print(f"\n数据库中的加密数据:")
        print(f"  加密卡密: {encrypted_key[:50]}...")
        print(f"  加密机器ID: {encrypted_id[:50]}...")
        
        # 验证是否真的加密了（不应该包含原始文本）
        if test_data['license_key'] not in encrypted_key and test_data['machine_id'] not in encrypted_id:
            print(f"\n[PASSED] 数据库加密验证通过（数据已加密）")
            db.delete_license()
            return True
        else:
            print(f"\n[FAILED] 数据未加密！")
            db.delete_license()
            return False
    else:
        print(f"\n[FAILED] 无法读取数据库")
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始测试许可证系统")
    print("=" * 60)
    
    results = []
    
    try:
        # 测试 1: 机器ID
        results.append(("机器ID生成", test_machine_id()))
        
        # 测试 2: 卡密格式验证
        results.append(("卡密格式验证", test_license_format_validation()))
        
        # 测试 3: 本地数据库
        results.append(("本地数据库操作", test_local_database()))
        
        # 测试 4: 许可证检查
        results.append(("许可证检查", test_license_check()))
        
        # 测试 5: 许可证信息
        results.append(("许可证信息获取", test_license_info()))
        
        # 测试 6: 数据库加密
        results.append(("数据库加密验证", test_encryption_in_database()))
        
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
        print("🎉 所有测试通过！许可证系统工作正常")
    else:
        print("[WARNING]️ 部分测试失败，请检查错误信息")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
