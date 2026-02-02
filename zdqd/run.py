#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
夜神模拟器自动化脚本 - 启动入口
Nox Emulator Automation Script - Run Entry Point
"""

import sys
import os
import traceback
import logging

# ============================================================
# Windows控制台UTF-8编码设置（彻底修复中文显示问题）
# ============================================================
if sys.platform == 'win32':
    import io
    import locale
    
    # 方法1: 设置环境变量（在Python启动前生效）
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    # 方法2: 设置控制台代码页为UTF-8（65001）
    try:
        # 静默执行，不显示输出
        os.system('chcp 65001 >nul 2>&1')
    except:
        pass
    
    # 方法3: 重新包装stdout和stderr，强制使用UTF-8
    try:
        # 检查是否已经是UTF-8编码
        if sys.stdout.encoding.lower() != 'utf-8':
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, 
                encoding='utf-8', 
                errors='replace',
                line_buffering=True  # 启用行缓冲，立即显示输出
            )
        if sys.stderr.encoding.lower() != 'utf-8':
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer, 
                encoding='utf-8', 
                errors='replace',
                line_buffering=True
            )
    except Exception as e:
        # 如果重新包装失败，至少记录错误
        print(f"警告: 无法设置UTF-8编码: {e}", file=sys.stderr)
    
    # 方法4: 设置默认编码（Python 3.7+）
    try:
        # 设置locale为UTF-8
        locale.setlocale(locale.LC_ALL, '')
    except:
        pass

# 隐藏第三方库的日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 隐藏TensorFlow日志
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('rapidocr').setLevel(logging.ERROR)
logging.getLogger('RapidOCR').setLevel(logging.ERROR)

def close_old_instances():
    """关闭旧的Python实例（避免多个实例同时运行）"""
    try:
        import psutil
        current_pid = os.getpid()
        script_name = os.path.basename(__file__)
        
        closed_count = 0
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                # 检查是否是Python进程
                if proc.info['name'] in ['python.exe', 'pythonw.exe']:
                    cmdline = proc.info['cmdline']
                    if cmdline and len(cmdline) > 1:
                        # 检查是否运行相同脚本
                        if script_name in ' '.join(cmdline) and proc.info['pid'] != current_pid:
                            print(f"发现旧实例 PID: {proc.info['pid']}, 正在关闭...")
                            proc.terminate()  # 优雅终止
                            try:
                                proc.wait(timeout=2)  # 等待2秒
                            except psutil.TimeoutExpired:
                                proc.kill()  # 强制终止
                            closed_count += 1
                            print(f"[OK] 已关闭旧实例 PID: {proc.info['pid']}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        if closed_count > 0:
            print(f"共关闭 {closed_count} 个旧实例")
        
    except ImportError:
        # 如果没有psutil库，使用taskkill命令（Windows）
        if sys.platform == 'win32':
            try:
                import subprocess
                script_name = os.path.basename(__file__)
                current_pid = os.getpid()
                
                # 查找运行相同脚本的进程
                cmd = f'wmic process where "name=\'python.exe\' and CommandLine like \'%{script_name}%\'" get ProcessId'
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
                
                # 解析进程ID
                closed_count = 0
                for line in result.stdout.split('\n'):
                    line = line.strip()
                    if line and line.isdigit():
                        pid = int(line)
                        if pid != current_pid:
                            try:
                                subprocess.run(f'taskkill /F /PID {pid}', shell=True, check=False, timeout=5)
                                print(f"[OK] 已关闭旧实例 PID: {pid}")
                                closed_count += 1
                            except:
                                pass
                
                if closed_count > 0:
                    print(f"共关闭 {closed_count} 个旧实例")
            except Exception as e:
                print(f"关闭旧实例失败: {e}")
    except Exception as e:
        print(f"关闭旧实例时出错: {e}")

def main():
    """主函数"""
    
    # 关闭旧实例（避免多个实例同时运行）
    close_old_instances()
    
    # 设置工作目录
    if getattr(sys, 'frozen', False):
        application_path = os.path.dirname(sys.executable)
    else:
        application_path = os.path.dirname(os.path.abspath(__file__))
    
    os.chdir(application_path)
    sys.path.insert(0, application_path)
    
    try:
        print("=" * 60)
        print("溪盟商城自动化助手 v2.0.6")
        print("=" * 60)
        
        # 使用简化版许可证管理器（避免调用外部命令，防止无限重启）
        from src.license_manager_simple import SimpleLicenseManager
        license_manager = SimpleLicenseManager()
        
        valid, message = license_manager.check_license()
        
        if not valid:
            # 未激活，显示激活对话框
            print(f"许可证无效: {message}")
            print("显示激活对话框...")
            
            try:
                from src.simple_activation_dialog import SimpleActivationDialog
                dialog = SimpleActivationDialog()
                
                if not dialog.result:
                    # 用户取消激活，退出程序
                    print("用户取消激活，程序退出")
                    return
                
                # 激活成功，继续启动
                print("激活成功，启动主程序...")
            except Exception as e:
                print(f"激活对话框错误: {e}")
                import traceback
                traceback.print_exc()
                return
        else:
            print(f"许可证有效：{message}")
        
        # ===== 初始化ADB连接 =====
        print("\n[启动] 正在初始化ADB连接...")
        from src.adb_bridge import ADBBridge
        from src.emulator_controller import EmulatorController
        
        # 从配置加载模拟器路径
        import yaml
        config_path = 'config.yaml'
        nox_path = None
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
                nox_path = config_data.get('nox_path', '')
        
        # 初始化模拟器控制器以获取ADB路径
        adb_path = None
        if nox_path:
            emulator_controller = EmulatorController(nox_path)
            adb_path = emulator_controller.get_adb_path()
            if adb_path:
                print(f"[OK] 找到ADB路径: {adb_path}")
            else:
                print("[WARNING] 未找到ADB路径，将使用系统PATH中的adb命令")
        else:
            print("[WARNING] 未配置模拟器路径，将使用系统PATH中的adb命令")
        
        # 创建ADB桥接器（传递ADB路径）
        adb = ADBBridge(adb_path)
        
        # ===== 启动GUI（模型将在GUI显示后后台加载）=====
        print("\n[启动] 正在启动用户界面...")
        print("[提示] 模型将在界面显示后后台加载，请稍候...")
        from src.gui import main as gui_main
        
        # 传递ADB实例给GUI，让GUI在显示后加载模型
        gui_main(adb_bridge=adb)
        
    except KeyboardInterrupt:
        print("程序被用户中断")
        # 清理模型资源
        try:
            from src.model_manager import ModelManager
            model_manager = ModelManager.get_instance()
            if model_manager.is_initialized():
                print("\n[退出] 正在清理模型资源...")
                model_manager.cleanup()
                print("[OK] 模型资源清理完成")
        except Exception as e:
            print(f"清理模型资源时出错: {e}")
        sys.exit(0)
    except Exception as e:
        # 捕获所有异常
        error_msg = f"程序运行错误: {e}\n{traceback.format_exc()}"
        print(error_msg)
        
        # 写入错误日志
        try:
            with open("error.log", "a", encoding="utf-8") as f:
                from datetime import datetime
                f.write(f"\n{'='*60}\n")
                f.write(f"时间: {datetime.now()}\n")
                f.write(error_msg)
                f.write(f"\n{'='*60}\n")
        except:
            pass
        
        # 显示错误对话框
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("错误", f"程序运行出错:\n{str(e)}\n\n详细信息已保存到 error.log")
            root.destroy()
        except:
            pass
        
        sys.exit(1)
    finally:
        # 清理模型资源
        try:
            from src.model_manager import ModelManager
            model_manager = ModelManager.get_instance()
            if model_manager.is_initialized():
                print("\n[退出] 正在清理模型资源...")
                model_manager.cleanup()
                print("[OK] 模型资源清理完成")
        except Exception as e:
            print(f"清理模型资源时出错: {e}")

if __name__ == "__main__":
    main()
