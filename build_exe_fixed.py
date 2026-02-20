#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyInstaller打包脚本 - 溪盟商城自动化助手（修复版）
Build Script for Packaging with PyInstaller (Fixed)
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
ICON_PATH = None  # 如果有图标文件，设置路径

def clean_build_dirs():
    """清理旧的构建目录"""
    print("\n[1/6] 清理旧的构建文件...")
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
    
    print("  [OK] 清理完成")

def create_pyinstaller_command():
    """创建PyInstaller打包命令"""
    print("\n[2/6] 准备打包命令...")
    
    # 基础命令
    cmd = [
        'pyinstaller',
        '--name', APP_NAME,
        '--windowed',  # 不显示控制台窗口（GUI应用）
        '--onedir',    # 打包成文件夹（推荐，方便更新）
        '--clean',     # 清理临时文件
        '--noconfirm', # 不询问确认
        '--additional-hooks-dir', '.',  # 使用当前目录的hook文件
        # 添加runtime hook，在程序启动时自动隐藏所有subprocess的CMD窗口
        '--runtime-hook', 'pyi_rth_subprocess.py',
        # 添加 multiprocessing 支持（Windows 必需）
        '--hidden-import', 'multiprocessing',
        '--hidden-import', 'multiprocessing.spawn',
        '--hidden-import', 'multiprocessing.pool',
        '--collect-submodules', 'multiprocessing',
    ]
    
    # 添加图标（如果有）
    if ICON_PATH and os.path.exists(ICON_PATH):
        cmd.extend(['--icon', ICON_PATH])
    
    # 添加数据文件
    data_items = [
        ('config', 'config'),
        ('models', 'models'),
        ('config.yaml', '.'),
        ('model_config.json.example', '.'),
        ('transfer_config.json.example', '.'),
        ('.env.example', '.'),
    ]
    
    for src, dst in data_items:
        if os.path.exists(src):
            cmd.extend(['--add-data', f'{src}{os.pathsep}{dst}'])
            print(f"  添加数据: {src} -> {dst}")
    
    # 添加隐藏导入
    hidden_imports = [
        'yaml',
        'cv2',
        'PIL',
        'numpy',
        'rapidocr_onnxruntime',
        'cryptography',
        'psutil',
        'imagehash',
        'tkinter',
        'tkinter.ttk',
        'tkinter.messagebox',
        'tkinter.filedialog',
        'asyncio',
        'sqlite3',
        'omegaconf',
    ]
    
    for module in hidden_imports:
        cmd.extend(['--hidden-import', module])
    
    # 排除大型不需要的模块
    # 注意：不排除 ultralytics，因为运行时需要YOLO
    exclude_modules = [
        'matplotlib',
        'scipy',
        'pandas',
        'IPython',
        'notebook',
        'jupyter',
        'pytest',
        'hypothesis',
        '_pytest',
        'pytest_cov',
        'coverage',
        # 不排除 torch，因为 ultralytics 需要它
        # 'torch',
        # 'torch.cuda',
        # 'torch.nn',
        # 'torch.optim',
        'tensorflow',
        'tensorflow.python',
        'tensorrt',
        'tensorrt_bindings',
        'jedi',
        'parso',
        'setuptools',
        'distutils',
        'wheel',
        'pip',
        'sklearn',
        'scikit-learn',
        'wandb',
        'h5py',
        'pywt',
        'shapely',
        'torchvision',
        # 不排除 ultralytics，运行时需要
        # 'ultralytics',
    ]
    
    for module in exclude_modules:
        cmd.extend(['--exclude-module', module])
    
    # 添加主入口文件
    cmd.append('run.py')
    
    print("  [OK] 命令准备完成")
    return cmd

def run_pyinstaller(cmd):
    """运行PyInstaller打包"""
    print("\n[3/6] 开始打包...")
    print(f"  执行命令: {' '.join(cmd)}")
    print("\n" + "="*60)
    
    try:
        result = subprocess.run(cmd, check=True)
        print("="*60)
        print("  [OK] 打包完成")
        return True
    except subprocess.CalledProcessError as e:
        print("="*60)
        print(f"  [ERROR] 打包失败: {e}")
        return False

def clean_unnecessary_files():
    """清理打包后不需要的文件"""
    print("\n[4/6] 清理不需要的文件...")
    
    base_dir = f'dist/{APP_NAME}'
    internal_dir = f'{base_dir}/_internal'
    
    if not os.path.exists(base_dir):
        print(f"  [ERROR] 找不到打包目录: {base_dir}")
        return False
    
    total_deleted = 0
    total_size = 0
    
    # 1. 删除TensorFlow库
    print("  [1/8] 删除TensorFlow库...")
    tf_dir = os.path.join(internal_dir, 'tensorflow')
    if os.path.exists(tf_dir):
        size = sum(f.stat().st_size for f in Path(tf_dir).rglob('*') if f.is_file())
        shutil.rmtree(tf_dir)
        total_deleted += 1
        total_size += size
        print(f"    删除: tensorflow ({size / 1024 / 1024:.2f} MB)")
    
    # 2. 删除PyTorch CUDA库（保留nvrtc和nvJitLink）
    print("  [2/8] 删除PyTorch CUDA库（保留nvrtc和nvJitLink）...")
    cuda_patterns = [
        'torch_cuda.dll',
        'cublas*.dll', 'cublasLt*.dll', 'cufft*.dll',
        'cusparse*.dll', 'cusolver*.dll', 'cusolverMg*.dll', 'curand*.dll',
        'cudnn*.dll',
    ]
    
    if os.path.exists(internal_dir):
        for pattern in cuda_patterns:
            for file_path in Path(internal_dir).rglob(pattern):
                if file_path.is_file():
                    size = file_path.stat().st_size
                    file_path.unlink()
                    total_deleted += 1
                    total_size += size
                    if size > 10 * 1024 * 1024:
                        print(f"    删除: {file_path.name} ({size / 1024 / 1024:.2f} MB)")
    
    # 3. 删除训练图片
    print("  [3/8] 删除训练图片...")
    models_dir = f'{base_dir}/models'
    img_count = 0
    img_size = 0
    if os.path.exists(models_dir):
        for pattern in ['**/*.jpg', '**/*.png', '**/*.jpeg']:
            for file_path in Path(models_dir).rglob(pattern):
                if file_path.is_file():
                    try:
                        size = file_path.stat().st_size
                        file_path.unlink()
                        img_count += 1
                        img_size += size
                    except:
                        pass
    if img_count > 0:
        print(f"    删除训练图片: {img_count} 个文件, {img_size / 1024 / 1024:.2f} MB")
        total_deleted += img_count
        total_size += img_size
    
    # 4. 删除YOLO训练权重（last.pt）
    print("  [4/8] 删除YOLO训练权重（last.pt）...")
    last_pt_count = 0
    last_pt_size = 0
    for file_path in Path(base_dir).rglob('last.pt'):
        if file_path.is_file():
            size = file_path.stat().st_size
            file_path.unlink()
            last_pt_count += 1
            last_pt_size += size
    if last_pt_count > 0:
        print(f"    删除last.pt: {last_pt_count} 个文件, {last_pt_size / 1024 / 1024:.2f} MB")
        total_deleted += last_pt_count
        total_size += last_pt_size
    
    # 5. 删除备份文件
    print("  [5/8] 删除备份文件...")
    backup_count = 0
    backup_size = 0
    for pattern in ['**/*.backup*', '**/*_backup*']:
        for file_path in Path(base_dir).rglob(pattern.split('/')[-1]):
            if file_path.is_file():
                size = file_path.stat().st_size
                file_path.unlink()
                backup_count += 1
                backup_size += size
    if backup_count > 0:
        print(f"    删除备份文件: {backup_count} 个文件, {backup_size / 1024 / 1024:.2f} MB")
        total_deleted += backup_count
        total_size += backup_size
    
    # 6. 删除不需要的库
    print("  [6/8] 删除不需要的库...")
    unnecessary_libs = [
        'pandas', 'sklearn', 'shapely', 'h5py', 'pywt',
        'matplotlib', 'scipy', 'torchvision', 'torchaudio',
        # 注意：不删除 ultralytics，运行时需要用它加载YOLO模型
    ]
    
    if os.path.exists(internal_dir):
        for lib_name in unnecessary_libs:
            for lib_dir in Path(internal_dir).glob(lib_name):
                if lib_dir.is_dir():
                    size = sum(f.stat().st_size for f in lib_dir.rglob('*') if f.is_file())
                    shutil.rmtree(lib_dir)
                    total_deleted += 1
                    total_size += size
                    if size > 1024 * 1024:
                        print(f"    删除库: {lib_name} ({size / 1024 / 1024:.2f} MB)")
            
            for lib_dir in Path(internal_dir).glob(f'{lib_name}.libs'):
                if lib_dir.is_dir():
                    size = sum(f.stat().st_size for f in lib_dir.rglob('*') if f.is_file())
                    shutil.rmtree(lib_dir)
                    total_deleted += 1
                    total_size += size
    
    # 7. 清理torch目录
    print("  [7/8] 清理torch目录...")
    torch_dir = os.path.join(internal_dir, 'torch')
    if os.path.exists(torch_dir):
        torch_unnecessary = ['include', 'share', 'test', 'testing', '_inductor']
        for subdir in torch_unnecessary:
            subdir_path = os.path.join(torch_dir, subdir)
            if os.path.exists(subdir_path):
                size = sum(f.stat().st_size for f in Path(subdir_path).rglob('*') if f.is_file())
                shutil.rmtree(subdir_path)
                total_deleted += 1
                total_size += size
                if size > 1024 * 1024:
                    print(f"    删除torch子目录: {subdir} ({size / 1024 / 1024:.2f} MB)")
    
    # 8. 删除重复的models目录
    print("  [8/8] 删除重复的models目录...")
    internal_models = os.path.join(internal_dir, 'models')
    root_models = os.path.join(base_dir, 'models')
    if os.path.exists(internal_models) and os.path.exists(root_models):
        size = sum(f.stat().st_size for f in Path(internal_models).rglob('*') if f.is_file())
        shutil.rmtree(internal_models)
        total_deleted += 1
        total_size += size
        print(f"    删除_internal/models: {size / 1024 / 1024:.2f} MB")
    
    print(f"  [OK] 清理完成：删除 {total_deleted} 项，释放 {total_size / 1024 / 1024:.2f} MB ({total_size / 1024 / 1024 / 1024:.2f} GB)")
    return True

def copy_additional_files():
    """复制额外需要的文件到dist目录"""
    print("\n[5/6] 复制额外文件...")
    
    dist_dir = f'dist/{APP_NAME}'
    if not os.path.exists(dist_dir):
        print(f"  [ERROR] 找不到打包目录: {dist_dir}")
        return False
    
    # 修复文件结构
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
            print(f"    [OK] {folder} 复制完成")
        else:
            print(f"    [WARN] 找不到 {src}")
    
    # 复制文件和文件夹
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
                print(f"  复制文件夹: {src} -> {dst}")
            else:
                shutil.copy2(src, dst_path)
                print(f"  复制文件: {src} -> {dst}")
        except Exception as e:
            print(f"  [WARN] 复制失败 {src}: {e}")
    
    # 创建运行时需要的空文件夹
    runtime_dirs = [
        'data',
        'login_cache',
        'screenshots',
        'logs',
        'reports',
        'runtime_data',
        'checkin_screenshots',
        'no_checkin_screenshots',
    ]
    
    for dir_name in runtime_dirs:
        dir_path = os.path.join(dist_dir, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        print(f"  创建目录: {dir_name}")
    
    # 复制账号文件示例
    account_example_src = '账号文件示例.txt'
    if os.path.exists(account_example_src):
        account_example_dst = os.path.join(dist_dir, 'data', '账号文件示例.txt')
        try:
            shutil.copy2(account_example_src, account_example_dst)
            print(f"  复制文件: {account_example_src} -> data/账号文件示例.txt")
        except Exception as e:
            print(f"  [WARN] 复制账号文件示例失败: {e}")
    
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
打包时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    readme_path = os.path.join(dist_dir, '使用说明.txt')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"  创建文件: 使用说明.txt")
    
    print("  [OK] 额外文件复制完成")
    return True

def move_to_output_dir():
    """移动打包结果到输出目录"""
    print(f"\n[6/6] 移动到输出目录: {OUTPUT_DIR}")
    
    dist_dir = f'dist/{APP_NAME}'
    if not os.path.exists(dist_dir):
        print(f"  [ERROR] 找不到打包目录: {dist_dir}")
        return False
    
    try:
        # 如果目标目录已存在，先删除（跳过，避免文件占用问题）
        # if os.path.exists(OUTPUT_DIR):
        #     print(f"  删除旧版本: {OUTPUT_DIR}")
        #     shutil.rmtree(OUTPUT_DIR)
        
        # 创建输出目录
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # 移动dist目录下的所有内容到输出目录
        print(f"  移动内容: {dist_dir}/* -> {OUTPUT_DIR}")
        for item in os.listdir(dist_dir):
            src = os.path.join(dist_dir, item)
            dst = os.path.join(OUTPUT_DIR, item)
            shutil.move(src, dst)
            print(f"    移动: {item}")
        
        print(f"  [OK] 移动完成")
        print(f"\n打包完成！")
        print(f"输出目录: {OUTPUT_DIR}")
        print(f"EXE文件: {os.path.join(OUTPUT_DIR, APP_NAME + '.exe')}")
        
        # 显示目录大小
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(OUTPUT_DIR):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        
        print(f"总大小: {total_size / 1024 / 1024:.2f} MB ({total_size / 1024 / 1024 / 1024:.2f} GB)")
        
        return True
    except Exception as e:
        print(f"  [ERROR] 移动失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    try:
        print("="*60)
        print(f"开始打包 {APP_NAME} v{VERSION}")
        print("="*60)
        
        # 1. 清理旧的构建文件
        clean_build_dirs()
        
        # 2. 创建PyInstaller命令
        cmd = create_pyinstaller_command()
        
        # 3. 运行PyInstaller打包
        if not run_pyinstaller(cmd):
            print("\n打包失败！")
            return False
        
        # 4. 清理不需要的文件
        if not clean_unnecessary_files():
            print("\n清理文件失败！")
            return False
        
        # 5. 复制额外文件
        if not copy_additional_files():
            print("\n复制额外文件失败！")
            return False
        
        # 6. 移动到输出目录
        if not move_to_output_dir():
            print("\n移动到输出目录失败！")
            return False
        
        print("\n" + "="*60)
        print("打包完成！")
        print("="*60)
        return True
    except Exception as e:
        print(f"\n打包过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
