#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Nuitka打包脚本 - 溪盟商城自动化助手
Build Script using Nuitka (alternative to PyInstaller)
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

# 设置控制台UTF-8编码
if sys.platform == 'win32':
    import io
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except:
        pass

# 打包配置
APP_NAME = "溪盟商城自动化助手"
VERSION = "2.0.6"
OUTPUT_DIR = "D:/溪盟商城自动化助手_打包_Nuitka"

def clean_models_for_packaging():
    """打包前清理models文件夹，只保留必需的模型文件"""
    print("\n[预处理] 清理models文件夹...")
    
    models_dir = Path('models')
    if not models_dir.exists():
        print("  ⚠ models文件夹不存在，跳过清理")
        return
    
    # 需要保留的文件
    keep_files = {
        "page_classifier_pytorch_best.pth",
        "yolo26n.pt",
        "yolov8n.pt",
        "model_version.json",
        "page_classes.json",
        "page_yolo_mapping.json",
        "yolo_model_registry.json",
    }
    
    # 统计
    deleted_count = 0
    saved_size = 0
    
    # 删除训练文件夹
    for folder_name in ["runs", "yolo_runs"]:
        folder_path = models_dir / folder_name
        if folder_path.exists():
            folder_size = sum(f.stat().st_size for f in folder_path.rglob('*') if f.is_file())
            shutil.rmtree(folder_path)
            deleted_count += 1
            saved_size += folder_size
            print(f"  删除训练文件夹: {folder_name} (节省 {folder_size/1024/1024:.1f} MB)")
    
    # 删除不需要的文件
    for item in models_dir.iterdir():
        if item.is_file() and item.name not in keep_files:
            file_size = item.stat().st_size
            item.unlink()
            deleted_count += 1
            saved_size += file_size
    
    if deleted_count > 0:
        print(f"  ✓ 清理完成，删除 {deleted_count} 项，节省 {saved_size/1024/1024:.1f} MB")
    else:
        print(f"  ✓ 无需清理")

def main():
    """主函数"""
    print("="*60)
    print(f"  {APP_NAME} v{VERSION} - Nuitka打包脚本")
    print("="*60)
    
    # 检查Nuitka是否安装
    try:
        result = subprocess.run(['python', '-m', 'nuitka', '--version'], capture_output=True, text=True)
        print(f"\nNuitka版本: {result.stdout.strip()}")
    except FileNotFoundError:
        print("\n✗ 未安装Nuitka")
        print("请运行: pip install nuitka")
        return False
    
    print("\n[提示] Nuitka打包说明：")
    print("  1. 首次使用会自动下载MinGW64编译器（约500MB）")
    print("  2. 编译过程需要10-30分钟，请耐心等待")
    print("  3. Nuitka会将Python代码编译成机器码，性能更好")
    print("  4. 编译后的程序无法反编译，代码更安全\n")
    
    # 清理models文件夹
    clean_models_for_packaging()
    
    # 构建Nuitka命令
    cmd = [
        'python', '-m', 'nuitka',
        '--standalone',  # 独立模式，包含所有依赖
        '--windows-disable-console',  # 不显示控制台（GUI应用）
        '--enable-plugin=tk-inter',  # 启用tkinter插件
        '--enable-plugin=numpy',  # 启用numpy插件
        '--include-data-dir=config=config',  # 包含config文件夹
        '--include-data-dir=models=models',  # 包含models文件夹
        '--include-data-file=config.yaml=config.yaml',  # 包含配置文件
        '--include-data-file=model_config.json.example=model_config.json.example',
        '--include-data-file=transfer_config.json.example=transfer_config.json.example',
        '--include-data-file=.env.example=.env.example',
        '--output-dir=nuitka_build',  # 输出目录
        f'--output-filename={APP_NAME}.exe',  # 输出文件名
        '--assume-yes-for-downloads',  # 自动下载依赖
        '--show-progress',  # 显示进度
        '--show-memory',  # 显示内存使用
        '--jobs=4',  # 使用4个并行任务加速编译
        'run.py'  # 主入口文件
    ]
    
    print("\n[开始打包]")
    print(f"执行命令: {' '.join(cmd)}\n")
    print("="*60)
    
    try:
        # 运行Nuitka
        result = subprocess.run(cmd, check=True)
        print("="*60)
        print("\n✓ 编译完成！")
        
        # 查找生成的文件夹
        build_dir = Path('nuitka_build/run.dist')
        if build_dir.exists():
            print(f"\n生成的文件在: {build_dir}")
            
            # 复制额外文件
            print("\n[复制额外文件]")
            copy_additional_files(build_dir)
            
            # 移动到输出目录
            print(f"\n[移动到输出目录: {OUTPUT_DIR}]")
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            target_dir = Path(OUTPUT_DIR) / APP_NAME
            
            if target_dir.exists():
                print(f"  删除旧版本: {target_dir}")
                shutil.rmtree(target_dir)
            
            shutil.move(str(build_dir), str(target_dir))
            print(f"  ✓ 已移动到: {target_dir}")
            
            # 显示结果
            exe_path = target_dir / f'{APP_NAME}.exe'
            print(f"\n{'='*60}")
            print(f"  打包完成！")
            print(f"  输出位置: {target_dir}")
            print(f"  可执行文件: {exe_path}")
            print(f"{'='*60}\n")
            
            # 显示文件夹大小
            total_size = sum(f.stat().st_size for f in target_dir.rglob('*') if f.is_file())
            print(f"  总大小: {total_size/1024/1024/1024:.2f} GB")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print("="*60)
        print(f"\n✗ 打包失败: {e}")
        return False
    except Exception as e:
        print(f"\n✗ 打包过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def copy_additional_files(dist_dir):
    """复制额外需要的文件到dist目录"""
    
    # 需要复制的文件和文件夹
    items_to_copy = [
        ('README.md', 'README.md'),
        ('更新日志.md', '更新日志.md'),
        ('docs', 'docs'),  # 文档文件夹
    ]
    
    for src, dst in items_to_copy:
        if not os.path.exists(src):
            continue
        
        dst_path = dist_dir / dst
        
        try:
            if os.path.isdir(src):
                if dst_path.exists():
                    shutil.rmtree(dst_path)
                shutil.copytree(src, dst_path)
                print(f"  复制文件夹: {src} -> {dst}")
            else:
                shutil.copy2(src, dst_path)
                print(f"  复制文件: {src} -> {dst}")
        except Exception as e:
            print(f"  ⚠ 复制失败 {src}: {e}")
    
    # 创建运行时需要的空文件夹
    runtime_dirs = [
        'data',  # 账号文件目录
        'login_cache',
        'screenshots',
        'logs',
        'reports',
        'runtime_data',
        'checkin_screenshots',
        'no_checkin_screenshots',
    ]
    
    for dir_name in runtime_dirs:
        dir_path = dist_dir / dir_name
        os.makedirs(dir_path, exist_ok=True)
        print(f"  创建目录: {dir_name}")
    
    # 复制账号文件示例到data目录
    account_example_src = '账号文件示例.txt'
    if os.path.exists(account_example_src):
        account_example_dst = dist_dir / 'data' / '账号文件示例.txt'
        try:
            shutil.copy2(account_example_src, account_example_dst)
            print(f"  复制文件: {account_example_src} -> data/账号文件示例.txt")
        except Exception as e:
            print(f"  ⚠ 复制账号文件示例失败: {e}")
    
    # 创建使用说明
    readme_content = f"""
# {APP_NAME} v{VERSION}

## 使用说明

1. 首次运行前，请确保：
   - 已安装MuMu模拟器
   - 模拟器分辨率设置为 540x960（竖屏）
   - 已在模拟器中安装目标应用

2. 双击 `{APP_NAME}.exe` 启动程序

3. 首次启动需要输入激活码

4. 配置文件说明：
   - config.yaml: 主配置文件（模拟器路径等）
   - model_config.json: 模型配置文件
   - transfer_config.json: 转账配置文件

5. GPU加速说明：
   - 如果你的电脑有NVIDIA显卡，程序会自动使用GPU加速
   - 如果没有GPU或GPU不可用，程序会自动使用CPU模式
   - 无需额外配置

## 常见问题

Q: 程序无法启动？
A: 请检查是否有杀毒软件拦截，将程序添加到白名单

Q: 找不到模拟器？
A: 请在config.yaml中配置正确的模拟器路径

Q: 模型加载失败？
A: 请确保models文件夹完整，不要删除任何文件

## 技术支持

如有问题，请查看docs文件夹中的文档或联系技术支持。

---
版本: {VERSION}
打包方式: Nuitka编译
打包时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    readme_path = dist_dir / '使用说明.txt'
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"  创建文件: 使用说明.txt")

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
