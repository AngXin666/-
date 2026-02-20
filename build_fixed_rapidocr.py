#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复后的打包脚本 - 正确处理rapidocr和models目录结构
"""

import os
import subprocess
import sys
import shutil
from pathlib import Path

def main():
    """主函数"""
    print("=" * 60)
    print("开始打包程序...")
    print("=" * 60)
    
    # 检查rapidocr安装
    print("\n[1] 检查rapidocr安装...")
    try:
        import rapidocr
        rapidocr_path = os.path.dirname(rapidocr.__file__)
        print(f"   ✓ rapidocr已安装: {rapidocr_path}")
        
        # 检查必需文件
        required_files = [
            'default_models.yaml',
            'config.yaml',
            'models/ch_PP-OCRv4_det_infer.onnx',
            'models/ch_PP-OCRv4_rec_infer.onnx',
        ]
        
        for file in required_files:
            file_path = os.path.join(rapidocr_path, file)
            if os.path.exists(file_path):
                print(f"   ✓ 找到: {file}")
            else:
                print(f"   ✗ 缺失: {file}")
                
    except ImportError:
        print("   ✗ rapidocr未安装")
        return False
    
    # 清理旧的打包文件
    print("\n[2] 清理旧的打包文件...")
    if os.path.exists("build"):
        print("   删除build目录...")
    if os.path.exists("dist"):
        print("   删除dist目录...")
    
    # 构建PyInstaller命令
    print("\n[3] 构建打包命令...")
    
    cmd = [
        "pyinstaller",
        "--name", "溪盟商城自动化助手",
        "--windowed",
        "--onedir",
        "--clean",
        "--noconfirm",
        
        # Runtime hooks
        "--runtime-hook", "pyi_rth_subprocess.py",
        
        # Additional hooks directory
        "--additional-hooks-dir", ".",
        
        # Hidden imports
        "--hidden-import", "multiprocessing",
        "--hidden-import", "yaml",
        "--hidden-import", "cv2",
        "--hidden-import", "PIL",
        "--hidden-import", "numpy",
        "--hidden-import", "rapidocr",  # 修复：使用rapidocr而不是rapidocr_onnxruntime
        "--hidden-import", "cryptography",
        "--hidden-import", "psutil",
        "--hidden-import", "tkinter",
        "--hidden-import", "asyncio",
        "--hidden-import", "sqlite3",
        
        # Data files
        "--add-data", "config;config",
        "--add-data", "models;models",
        "--add-data", "config.yaml;.",
        
        # Output directory
        "--distpath", "D:\\溪盟商城自动化助手_打包",
        
        # Entry point
        "run.py"
    ]
    
    print("   命令:")
    print("   " + " ".join(cmd))
    
    # 执行打包
    print("\n[4] 开始打包...")
    print("=" * 60)
    
    try:
        result = subprocess.run(cmd, check=True)
        print("=" * 60)
        print("\n✓ 打包完成！")
        
        # 后处理：移动models和config到exe同级目录
        print("\n[5] 后处理：调整目录结构...")
        output_dir = Path("D:\\溪盟商城自动化助手_打包\\溪盟商城自动化助手")
        internal_dir = output_dir / "_internal"
        
        # 移动models文件夹
        models_src = internal_dir / "models"
        models_dst = output_dir / "models"
        if models_src.exists():
            if models_dst.exists():
                print(f"   删除旧的models目录: {models_dst}")
                shutil.rmtree(models_dst)
            print(f"   移动models: {models_src} -> {models_dst}")
            shutil.move(str(models_src), str(models_dst))
            print("   ✓ models文件夹已移动到exe同级目录")
        else:
            print(f"   ⚠️ 未找到models目录: {models_src}")
        
        # 移动config文件夹
        config_src = internal_dir / "config"
        config_dst = output_dir / "config"
        if config_src.exists():
            if config_dst.exists():
                print(f"   删除旧的config目录: {config_dst}")
                shutil.rmtree(config_dst)
            print(f"   移动config: {config_src} -> {config_dst}")
            shutil.move(str(config_src), str(config_dst))
            print("   ✓ config文件夹已移动到exe同级目录")
        else:
            print(f"   ⚠️ 未找到config目录: {config_src}")
        
        # 移动config.yaml
        config_yaml_src = internal_dir / "config.yaml"
        config_yaml_dst = output_dir / "config.yaml"
        if config_yaml_src.exists():
            if config_yaml_dst.exists():
                print(f"   删除旧的config.yaml: {config_yaml_dst}")
                config_yaml_dst.unlink()
            print(f"   移动config.yaml: {config_yaml_src} -> {config_yaml_dst}")
            shutil.move(str(config_yaml_src), str(config_yaml_dst))
            print("   ✓ config.yaml已移动到exe同级目录")
        
        print("\n✓ 目录结构调整完成！")
        print(f"输出目录: D:\\溪盟商城自动化助手_打包")
        print("\n最终目录结构:")
        print("  溪盟商城自动化助手\\")
        print("  ├── 溪盟商城自动化助手.exe")
        print("  ├── models\\          ← 模型文件")
        print("  ├── config\\          ← 配置文件")
        print("  ├── config.yaml")
        print("  └── _internal\\       ← 依赖文件")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print("=" * 60)
        print(f"\n✗ 打包失败: {e}")
        return False
    except Exception as e:
        print(f"\n✗ 后处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
