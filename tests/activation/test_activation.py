"""
测试激活功能
Test Activation System
"""

import sys
import os

# 确保导入路径正确 - 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

def test_license_manager():
    """测试许可证管理器"""
    print("=" * 60)
    print("测试许可证管理器")
    print("=" * 60)
    
    from src.license_manager import LicenseManager
    
    manager = LicenseManager()
    
    # 1. 测试获取机器ID
    print("\n1. 获取机器ID:")
    machine_id = manager.get_machine_id()
    print(f"   机器ID: {machine_id}")
    
    # 2. 测试卡密格式验证
    print("\n2. 测试卡密格式验证:")
    test_keys = [
        "KIRO-TEST-1234-5678-ABCD",  # 正确格式
        "KIRO-XXXX-YYYY-ZZZZ-AAAA",  # 正确格式
        "KIRO-123",                   # 错误格式（太短）
        "TEST-1234-5678-ABCD-EFGH",  # 错误格式（前缀错误）
    ]
    
    for key in test_keys:
        valid, msg = manager.validate_license_key(key)
        status = "[OK] 有效" if valid else "[ERROR] 无效"
        print(f"   {key}: {status}")
        if not valid:
            print(f"      原因: {msg}")
    
    # 3. 检查当前许可证状态
    print("\n3. 检查当前许可证状态:")
    valid, message, info = manager.check_license()
    print(f"   状态: {'有效' if valid else '无效'}")
    print(f"   消息: {message}")
    if info:
        print(f"   详细信息:")
        print(f"      卡密: {info.get('license_key', 'N/A')}")
        print(f"      激活时间: {info.get('activated_at', 'N/A')}")
        print(f"      过期时间: {info.get('expires_at', 'N/A')}")
        print(f"      状态: {info.get('status', 'N/A')}")
        print(f"      设备限制: {info.get('max_devices', 1)} 台")
        print(f"      已绑定: {info.get('device_count', 0)} 台")
    
    # 4. 测试离线激活（使用测试卡密）
    print("\n4. 测试离线激活:")
    test_license = "KIRO-TEST-1234-5678-ABCD"
    print(f"   使用测试卡密: {test_license}")
    
    # 如果已经激活，先删除许可证文件
    if manager.license_file.exists():
        print("   检测到已有许可证，跳过激活测试")
    else:
        success, msg = manager.activate_license(test_license, machine_id)
        print(f"   激活结果: {'成功' if success else '失败'}")
        print(f"   消息: {msg}")
        
        if success:
            # 再次检查状态
            valid, message, info = manager.check_license()
            if valid and info:
                print(f"   验证成功:")
                print(f"      卡密: {info.get('license_key')}")
                print(f"      过期时间: {info.get('expires_at')}")
    
    print("\n" + "=" * 60)
    print("许可证管理器测试完成")
    print("=" * 60)


def test_removed_process_lock():
    """进程锁功能已移除"""
    print("\n" + "=" * 60)
    print("进程锁功能已移除（简化程序）")
    print("=" * 60)
    print("   [OK] 进程锁功能已移除，程序更简洁")
    print("\n" + "=" * 60)
    print("进程锁测试完成")
    print("=" * 60)


def test_activation_dialog():
    """测试激活对话框"""
    print("\n" + "=" * 60)
    print("测试激活对话框")
    print("=" * 60)
    
    try:
        import tkinter as tk
        from src.activation_dialog import ActivationDialog
        from src.license_manager import LicenseManager
        
        print("\n正在打开激活对话框...")
        print("请在对话框中测试以下功能：")
        print("  1. 查看设备ID")
        print("  2. 复制设备ID")
        print("  3. 输入测试卡密: KIRO-TEST-1234-5678-ABCD")
        print("  4. 点击激活按钮")
        print("\n注意：如果已经激活，对话框可能不会显示")
        
        # 检查是否已激活
        manager = LicenseManager()
        valid, message, info = manager.check_license()
        
        if valid:
            print(f"\n当前已激活: {message}")
            print("如需测试激活对话框，请先删除许可证文件:")
            print(f"  {manager.license_file}")
            
            response = input("\n是否删除现有许可证以测试激活对话框？(y/n): ")
            if response.lower() == 'y':
                try:
                    manager.license_file.unlink()
                    print("[OK] 已删除许可证文件")
                except Exception as e:
                    print(f"[ERROR] 删除失败: {e}")
                    return
            else:
                print("跳过激活对话框测试")
                return
        
        # 显示激活对话框
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        
        dialog = ActivationDialog(root)
        
        if dialog.result:
            print("\n[OK] 激活成功！")
            
            # 验证激活结果
            valid, message, info = manager.check_license()
            if valid and info:
                print(f"\n激活信息:")
                print(f"  卡密: {info.get('license_key')}")
                print(f"  设备限制: {info.get('max_devices', 1)} 台")
                print(f"  已绑定: {info.get('device_count', 1)} 台")
                print(f"  剩余天数: {info.get('days_left', 0)} 天")
        else:
            print("\n[ERROR] 激活取消或失败")
        
        root.destroy()
        
    except ImportError as e:
        print(f"[ERROR] 导入错误: {e}")
        print("请确保 tkinter 已安装")
    except Exception as e:
        print(f"[ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("激活对话框测试完成")
    print("=" * 60)


def main():
    """主函数"""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 18 + "激活系统测试工具" + " " * 18 + "║")
    print("╚" + "═" * 58 + "╝")
    
    while True:
        print("\n请选择测试项目:")
        print("  1. 测试许可证管理器")
        print("  2. 进程锁功能已移除")
        print("  3. 测试激活对话框（GUI）")
        print("  4. 运行所有测试")
        print("  0. 退出")
        
        choice = input("\n请输入选项 (0-4): ").strip()
        
        if choice == '1':
            test_license_manager()
        elif choice == '2':
            test_removed_process_lock()
        elif choice == '3':
            test_activation_dialog()
        elif choice == '4':
            test_license_manager()
            test_removed_process_lock()
            test_activation_dialog()
        elif choice == '0':
            print("\n退出测试")
            break
        else:
            print("\n[ERROR] 无效选项，请重新选择")
    
    print("\n测试完成！\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
    except Exception as e:
        print(f"\n[ERROR] 测试出错: {e}")
        import traceback
        traceback.print_exc()
