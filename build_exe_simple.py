#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyInstaller打包脚本 - 溪盟商城自动化助手（简化版）
回归到最基本的配置，避免过度配置导致的问题
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
OUTPUT_DIR = "D:/溪盟商城自动化助手_打包"

def clean_build_dirs():
    """清理旧的构建目录"""
    print("\n[1/5] 清理旧的构建文件...")
    dirs_to_clean = ['build', 'dist', '__pycache__']
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            print(f"  删除: {dir_name}")
            shutil.rmtree(dir_name, ignore_errors=True)
    
    # 清理.spec文件
    spec_files = [f for f in os.listdir('.') if f.endswith('.spec')]
    for spec_file in spec_files:
        print(f"  删除: {spec_file}")
        os.remove(spec_file)
    
    print("  ✓ 清理完成")

def create_pyinstaller_command():
    """创建PyInstaller打包命令 - 简化版"""
    print("\n[2/5] 准备打包命令（简化版）...")
    
    # 基础命令 - 只保留最必要的配置
    cmd = [
        'pyinstaller',
        '--name', APP_NAME,
        '--windowed',
        '--onedir',
        '--clean',
        '--noconfirm',
        # 只添加subprocess的runtime hook
        '--runtime-hook', 'pyi_rth_subprocess.py',
        # multiprocessing支持
        '--hidden-import', 'multiprocessing',
        '--hidden-import', 'multiprocessing.spawn',
    ]
    
    # 添加数据文件
    data_items = [
        ('config', 'config'),
        ('models', 'models'),
        ('config.yaml', '.'),
        ('model_config.json.example', '.'),
        ('transfer_config.json.example', '.'),
        ('.env.example', '.'),
        ('check_packed_models.py', '.'),  # 添加诊断工具
    ]
    
    for src, dst in data_items:
        if os.path.exists(src):
            cmd.extend(['--add-data', f'{src}{os.pathsep}{dst}'])
            print(f"  添加数据: {src} -> {dst}")
    
    # 添加rapidocr的数据文件
    try:
        import rapidocr
        rapidocr_path = os.path.dirname(rapidocr.__file__)
        
        # 添加rapidocr的所有必需文件
        # 注意: PyInstaller的--add-data格式是 源路径;目标路径
        rapidocr_items = [
            (os.path.join(rapidocr_path, 'config.yaml'), 'rapidocr'),
            (os.path.join(rapidocr_path, 'models'), 'rapidocr/models'),
        ]
        
        for src, dst in rapidocr_items:
            if os.path.exists(src):
                cmd.extend(['--add-data', f'{src}{os.pathsep}{dst}'])
                if os.path.isdir(src):
                    print(f"  添加数据: rapidocr/{os.path.basename(src)}/ (目录)")
                else:
                    print(f"  添加数据: rapidocr/{os.path.basename(src)}")
            else:
                print(f"  ⚠ 找不到: {src}")
                
        # 同时使用collect-data收集所有rapidocr数据
        cmd.extend(['--collect-data', 'rapidocr'])
        print(f"  收集数据: rapidocr (所有数据文件)")
        
    except Exception as e:
        print(f"  ⚠ rapidocr数据文件添加失败: {e}")
    
    # 只添加必需的hidden imports - 不要过度指定asyncio
    hidden_imports = [
        'yaml',
        'cv2',
        'PIL',
        'numpy',
        'rapidocr',
        'cryptography',
        'psutil',
        'imagehash',
        'tkinter',
        'tkinter.ttk',
        'tkinter.messagebox',
        'tkinter.filedialog',
        # asyncio - 让PyInstaller自动处理，不要手动指定子模块
        'asyncio',
        'sqlite3',
        'omegaconf',
    ]
    
    for module in hidden_imports:
        cmd.extend(['--hidden-import', module])
    
    # 排除不需要的模块
    exclude_modules = [
        'matplotlib',
        'scipy',
        'IPython',
        'notebook',
        'pytest',
        'hypothesis',
    ]
    
    for module in exclude_modules:
        cmd.extend(['--exclude-module', module])
    
    # 主入口
    cmd.append('run.py')
    
    print("  ✓ 命令准备完成（简化配置）")
    return cmd

def run_pyinstaller(cmd):
    """运行PyInstaller打包"""
    print("\n[3/5] 开始打包...")
    print(f"  执行命令: {' '.join(cmd)}")
    print("\n" + "="*60)
    
    try:
        result = subprocess.run(cmd, check=True)
        print("="*60)
        print("  ✓ 打包完成")
        return True
    except subprocess.CalledProcessError as e:
        print("="*60)
        print(f"  ✗ 打包失败: {e}")
        return False

def copy_additional_files():
    """复制额外需要的文件到dist目录"""
    print("\n[4/5] 复制额外文件...")
    
    dist_dir = f'dist/{APP_NAME}'
    if not os.path.exists(dist_dir):
        print(f"  ✗ 找不到打包目录: {dist_dir}")
        return False
    
    # 修复文件结构 - 确保models文件夹在根目录
    print("  修复文件结构...")
    folders_to_fix = ['config', 'models']
    for folder in folders_to_fix:
        src = os.path.join(dist_dir, '_internal', folder)
        dst = os.path.join(dist_dir, folder)
        
        if os.path.exists(src):
            print(f"    复制 {folder} 到根目录...")
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"    ✓ {folder} 复制完成")
            
            # 验证模型文件
            if folder == 'models':
                required_files = [
                    'page_classifier_pytorch_best.pth',
                    'yolo26n.pt',
                    'yolov8n.pt',
                    'model_version.json',
                    'page_classes.json',
                    'page_yolo_mapping.json',
                    'yolo_model_registry.json'
                ]
                print(f"    验证模型文件...")
                for file_name in required_files:
                    file_path = os.path.join(dst, file_name)
                    if os.path.exists(file_path):
                        file_size = os.path.getsize(file_path) / 1024 / 1024
                        print(f"      ✓ {file_name} ({file_size:.1f}MB)")
                    else:
                        print(f"      ✗ 缺失: {file_name}")
        else:
            print(f"    ⚠ 警告：找不到 {src}")
    
    # 复制文档
    items_to_copy = [
        ('README.md', 'README.md'),
        ('更新日志.md', '更新日志.md'),
        ('docs', 'docs'),
    ]
    
    for src, dst in items_to_copy:
        if not os.path.exists(src):
            continue
        
        dst_path = os.path.join(dist_dir, dst)
        
        try:
            if os.path.isdir(src):
                if os.path.exists(dst_path):
                    shutil.rmtree(dst_path)
                shutil.copytree(src, dst_path)
                print(f"  复制文件夹: {src}")
            else:
                shutil.copy2(src, dst_path)
                print(f"  复制文件: {src}")
        except Exception as e:
            print(f"  ⚠ 复制失败 {src}: {e}")
    
    # 创建运行时目录
    runtime_dirs = [
        'data', 'login_cache', 'screenshots', 'logs', 'reports',
        'runtime_data', 'checkin_screenshots', 'no_checkin_screenshots',
    ]
    
    for dir_name in runtime_dirs:
        dir_path = os.path.join(dist_dir, dir_name)
        os.makedirs(dir_path, exist_ok=True)
    
    # 复制账号文件示例
    account_example_src = '账号文件示例.txt'
    if os.path.exists(account_example_src):
        account_example_dst = os.path.join(dist_dir, 'data', '账号文件示例.txt')
        shutil.copy2(account_example_src, account_example_dst)
    
    # 创建使用说明
    readme_content = f"""# {APP_NAME} v{VERSION}

## 使用说明

1. 双击 `{APP_NAME}.exe` 启动程序
2. 首次启动需要输入激活码
3. 确保MuMu模拟器已安装并设置为 540x960 分辨率

## 故障排查

如果程序启动时提示"模型文件缺失"或"模型加载失败":

1. 运行 `check_packed_models.exe` 检查模型文件是否完整
2. 确保 `models` 文件夹与程序在同一目录下
3. 检查 `models` 文件夹中是否包含以下文件:
   - page_classifier_pytorch_best.pth
   - yolo26n.pt
   - yolov8n.pt
   - model_version.json
   - page_classes.json
   - page_yolo_mapping.json
   - yolo_model_registry.json

## 配置文件

- config.yaml: 主配置文件
- model_config.json: 模型配置
- transfer_config.json: 转账配置

---
版本: {VERSION}
"""
    
    readme_path = os.path.join(dist_dir, '使用说明.txt')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print("  ✓ 额外文件复制完成")
    return True

def move_to_output_dir():
    """移动打包结果到输出目录"""
    print(f"\n[5/5] 移动到输出目录: {OUTPUT_DIR}")
    
    dist_dir = f'dist/{APP_NAME}'
    if not os.path.exists(dist_dir):
        print(f"  ✗ 找不到打包目录: {dist_dir}")
        return False
    
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        target_dir = os.path.join(OUTPUT_DIR, APP_NAME)
        
        # 备份用户数据（优先从旧版本，如果没有则从工作区）
        backup_data = {}
        
        # 尝试从旧版本备份
        if os.path.exists(target_dir):
            print(f"  备份旧版本用户数据...")
            
            # 备份账号文件
            data_dir = os.path.join(target_dir, 'data')
            if os.path.exists(data_dir):
                backup_data['data'] = []
                for file in os.listdir(data_dir):
                    if file.endswith('.enc') or file.endswith('.txt'):
                        src = os.path.join(data_dir, file)
                        if os.path.isfile(src):
                            with open(src, 'rb') as f:
                                backup_data['data'].append((file, f.read()))
                            print(f"    备份: data/{file}")
            
            # 备份登录缓存
            login_cache_dir = os.path.join(target_dir, 'login_cache')
            if os.path.exists(login_cache_dir):
                backup_data['login_cache'] = []
                for item in os.listdir(login_cache_dir):
                    item_path = os.path.join(login_cache_dir, item)
                    if os.path.isdir(item_path):
                        # 备份整个账号缓存目录
                        backup_data['login_cache'].append((item, item_path))
                        print(f"    备份: login_cache/{item}/")
                    elif os.path.isfile(item_path):
                        # 备份单个文件
                        with open(item_path, 'rb') as f:
                            backup_data['login_cache'].append((item, f.read()))
                        print(f"    备份: login_cache/{item}")
            
            print(f"  删除旧版本...")
            shutil.rmtree(target_dir)
        
        # 如果没有备份到数据，从工作区复制
        if 'data' not in backup_data or not backup_data['data']:
            print(f"  从工作区复制账号文件...")
            workspace_data_dir = 'data'
            if os.path.exists(workspace_data_dir):
                backup_data['data'] = []
                for file in os.listdir(workspace_data_dir):
                    if file.endswith('.enc') or file.endswith('.txt'):
                        src = os.path.join(workspace_data_dir, file)
                        if os.path.isfile(src):
                            with open(src, 'rb') as f:
                                backup_data['data'].append((file, f.read()))
                            print(f"    复制: data/{file}")
        
        # 复制账号缓存文件（.account_cache.json.enc）到根目录
        account_cache_file = '.account_cache.json.enc'
        if os.path.exists(account_cache_file):
            print(f"  从工作区复制账号缓存文件...")
            with open(account_cache_file, 'rb') as f:
                if 'root_files' not in backup_data:
                    backup_data['root_files'] = []
                backup_data['root_files'].append((account_cache_file, f.read()))
                print(f"    复制: {account_cache_file} (根目录)")
        
        if 'login_cache' not in backup_data or not backup_data['login_cache']:
            print(f"  从工作区复制登录缓存...")
            workspace_login_cache = 'login_cache'
            if os.path.exists(workspace_login_cache):
                backup_data['login_cache'] = []
                for item in os.listdir(workspace_login_cache):
                    item_path = os.path.join(workspace_login_cache, item)
                    if os.path.isdir(item_path):
                        # 复制整个账号缓存目录
                        backup_data['login_cache'].append((item, item_path))
                        print(f"    复制: login_cache/{item}/")
                    elif os.path.isfile(item_path):
                        # 复制单个文件
                        with open(item_path, 'rb') as f:
                            backup_data['login_cache'].append((item, f.read()))
                        print(f"    复制: login_cache/{item}")
        
        print(f"  移动文件...")
        shutil.move(dist_dir, target_dir)
        
        # 恢复用户数据
        if backup_data:
            print(f"  恢复用户数据...")
            
            # 恢复账号文件
            if 'data' in backup_data:
                data_dir = os.path.join(target_dir, 'data')
                os.makedirs(data_dir, exist_ok=True)
                for file_name, content in backup_data['data']:
                    dst = os.path.join(data_dir, file_name)
                    with open(dst, 'wb') as f:
                        f.write(content)
                    print(f"    恢复: data/{file_name}")
            
            # 恢复根目录文件（如 .account_cache.json.enc）
            if 'root_files' in backup_data:
                for file_name, content in backup_data['root_files']:
                    dst = os.path.join(target_dir, file_name)
                    with open(dst, 'wb') as f:
                        f.write(content)
                    print(f"    恢复: {file_name} (根目录)")
            
            # 恢复登录缓存
            if 'login_cache' in backup_data:
                login_cache_dir = os.path.join(target_dir, 'login_cache')
                os.makedirs(login_cache_dir, exist_ok=True)
                for item_name, content in backup_data['login_cache']:
                    # 判断是目录路径还是文件内容
                    if isinstance(content, str):
                        # content 是目录路径（字符串）
                        if os.path.isdir(content):
                            dst = os.path.join(login_cache_dir, item_name)
                            if os.path.exists(dst):
                                shutil.rmtree(dst)
                            shutil.copytree(content, dst)
                            print(f"    恢复: login_cache/{item_name}/")
                    elif isinstance(content, bytes):
                        # content 是文件内容（bytes）
                        dst = os.path.join(login_cache_dir, item_name)
                        with open(dst, 'wb') as f:
                            f.write(content)
                        print(f"    恢复: login_cache/{item_name}")
        
        print(f"\n{'='*60}")
        print(f"  ✓ 打包完成！")
        print(f"  输出位置: {target_dir}")
        print(f"{'='*60}\n")
        
        return True
    except Exception as e:
        print(f"  ✗ 移动失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("="*60)
    print(f"  {APP_NAME} v{VERSION} - 打包脚本（简化版）")
    print("="*60)
    
    try:
        clean_build_dirs()
        cmd = create_pyinstaller_command()
        
        if not run_pyinstaller(cmd):
            return False
        
        copy_additional_files()
        move_to_output_dir()
        
        # 打包诊断工具
        print("\n[额外] 打包诊断工具...")
        try:
            diag_cmd = [
                'pyinstaller',
                '--name', 'check_packed_models',
                '--onefile',
                '--console',
                '--clean',
                '--noconfirm',
                'check_packed_models.py'
            ]
            subprocess.run(diag_cmd, check=True, capture_output=True)
            
            # 复制诊断工具到输出目录
            diag_exe = 'dist/check_packed_models.exe'
            if os.path.exists(diag_exe):
                target_dir = os.path.join(OUTPUT_DIR, APP_NAME)
                if os.path.exists(target_dir):
                    shutil.copy2(diag_exe, os.path.join(target_dir, 'check_packed_models.exe'))
                    print("  ✓ 诊断工具已添加")
        except Exception as e:
            print(f"  ⚠ 诊断工具打包失败: {e}")
        
        print("\n✓ 打包完成！")
        print("\n提示:")
        print("  1. 如果程序启动失败,请先运行 check_packed_models.exe 检查模型文件")
        print("  2. 确保 models 文件夹与程序在同一目录")
        return True
        
    except KeyboardInterrupt:
        print("\n\n✗ 用户中断")
        return False
    except Exception as e:
        print(f"\n✗ 打包出错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
