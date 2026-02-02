"""
测试激活对话框
直接测试激活对话框的显示和功能
"""

import sys
import os
from pathlib import Path

# 确保导入路径正确
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    """主函数"""
    print("=" * 60)
    print("激活对话框测试")
    print("=" * 60)
    
    try:
        from src.license_manager import LicenseManager
        from src.activation_dialog import ActivationDialog
        import tkinter as tk
        
        # 检查当前激活状态
        manager = LicenseManager()
        valid, message, info = manager.check_license()
        
        print(f"\n当前激活状态: {'已激活' if valid else '未激活'}")
        print(f"消息: {message}")
        
        if valid and info:
            print(f"\n当前许可证信息:")
            print(f"  卡密: {info.get('license_key', 'N/A')}")
            print(f"  设备限制: {info.get('max_devices', 1)} 台")
            print(f"  已绑定: {info.get('device_count', 1)} 台")
            print(f"  剩余天数: {info.get('days_left', 0)} 天")
            print(f"  过期时间: {info.get('expires_at', 'N/A')}")
            
            print("\n" + "=" * 60)
            choice = input("是否删除现有许可证以测试激活对话框？(y/n): ")
            
            if choice.lower() != 'y':
                print("\n保持现有激活状态，退出测试")
                return
            
            # 删除许可证文件
            try:
                if manager.license_file.exists():
                    manager.license_file.unlink()
                    print("[OK] 已删除许可证文件")
                else:
                    print("[OK] 许可证文件不存在")
            except Exception as e:
                print(f"[ERROR] 删除失败: {e}")
                return
        
        print("\n" + "=" * 60)
        print("准备显示激活对话框")
        print("=" * 60)
        print("\n测试说明:")
        print("  1. 对话框会显示您的设备ID（机器码）")
        print("  2. 可以点击复制按钮复制设备ID")
        print("  3. 输入测试卡密: KIRO-TEST-1234-5678-ABCD")
        print("  4. 点击'激活'按钮进行激活")
        print("  5. 激活成功后会显示设备授权信息")
        print("\n提示: 这是离线测试模式，不会连接服务器")
        print("\n按回车键打开激活对话框...")
        input()
        
        # 创建并显示激活对话框
        print("\n正在打开激活对话框...")
        
        dialog = ActivationDialog()
        
        # 检查激活结果
        if dialog.result:
            print("\n" + "=" * 60)
            print("[OK] 激活成功！")
            print("=" * 60)
            
            # 验证激活
            valid, message, info = manager.check_license()
            
            if valid and info:
                print(f"\n激活信息:")
                print(f"  卡密: {info.get('license_key')}")
                print(f"  激活时间: {info.get('activated_at')}")
                print(f"  过期时间: {info.get('expires_at')}")
                print(f"  设备限制: {info.get('max_devices', 1)} 台")
                print(f"  已绑定: {info.get('device_count', 1)} 台")
                print(f"  剩余天数: {info.get('days_left', 0)} 天")
                print(f"  状态: {info.get('status')}")
                
                print(f"\n许可证文件位置: {manager.license_file}")
            else:
                print(f"\n[ERROR] 激活验证失败: {message}")
        else:
            print("\n" + "=" * 60)
            print("[ERROR] 激活取消或失败")
            print("=" * 60)
        
    except ImportError as e:
        print(f"\n[ERROR] 导入错误: {e}")
        print("请确保所有依赖已安装")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n[ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
    print("\n按回车键退出...")
    input()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
    except Exception as e:
        print(f"\n[ERROR] 发生错误: {e}")
        import traceback
        traceback.print_exc()
        input("\n按回车键退出...")
