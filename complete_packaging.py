#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""完成打包的后续步骤"""

import os
import shutil
from pathlib import Path

APP_NAME = "溪盟商城自动化助手"
OUTPUT_DIR = "D:/溪盟商城自动化助手_打包"

def copy_additional_files():
    """复制额外文件"""
    print("\n[1/3] 复制额外文件...")
    
    dist_dir = f'dist/{APP_NAME}'
    if not os.path.exists(dist_dir):
        print(f"  错误：找不到打包目录: {dist_dir}")
        return False
    
    # 复制 src 目录到 _internal
    print("  复制 src 目录到 _internal...")
    src_dir = 'src'
    internal_src_dst = os.path.join(dist_dir, '_internal', 'src')
    
    if os.path.exists(src_dir):
        if os.path.exists(internal_src_dst):
            shutil.rmtree(internal_src_dst)
        shutil.copytree(src_dir, internal_src_dst)
        file_count = sum(1 for _ in Path(internal_src_dst).rglob('*') if _.is_file())
        print(f"    已复制 src 模块 ({file_count} 个文件)")
    
    # 修复文件结构
    print("  修复文件结构...")
    folders_to_fix = ['config', 'models']
    for folder in folders_to_fix:
        src = os.path.join(dist_dir, '_internal', folder)
        dst = os.path.join(dist_dir, folder)
        
        if os.path.exists(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"    已复制 {folder} 到根目录")
    
    # 复制 config.yaml
    if os.path.exists('config.yaml'):
        shutil.copy2('config.yaml', os.path.join(dist_dir, 'config.yaml'))
        print("  已复制 config.yaml")
    
    # 创建运行时目录
    runtime_dirs = ['data', 'login_cache', 'screenshots', 'logs', 'reports', 
                    'runtime_data', 'checkin_screenshots', 'no_checkin_screenshots']
    for dir_name in runtime_dirs:
        os.makedirs(os.path.join(dist_dir, dir_name), exist_ok=True)
    print(f"  已创建 {len(runtime_dirs)} 个运行时目录")
    
    return True

def move_to_output():
    """移动到输出目录"""
    print(f"\n[2/3] 移动到输出目录: {OUTPUT_DIR}")
    
    dist_dir = f'dist/{APP_NAME}'
    if not os.path.exists(dist_dir):
        print(f"  错误：找不到打包目录")
        return False
    
    # 删除旧版本
    if os.path.exists(OUTPUT_DIR):
        print(f"  删除旧版本...")
        shutil.rmtree(OUTPUT_DIR)
    
    # 创建输出目录并移动内容
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for item in os.listdir(dist_dir):
        src = os.path.join(dist_dir, item)
        dst = os.path.join(OUTPUT_DIR, item)
        shutil.move(src, dst)
    
    print(f"  已移动到: {OUTPUT_DIR}")
    return True

def optimize_package():
    """优化打包大小"""
    print(f"\n[3/3] 优化打包大小...")
    
    if not os.path.exists(OUTPUT_DIR):
        print(f"  错误：输出目录不存在")
        return False
    
    total_deleted = 0
    total_size = 0
    
    # 1. 清理重复的 YOLO 模型
    print("  清理重复的 YOLO 模型...")
    models_dir = os.path.join(OUTPUT_DIR, 'models')
    
    keep_patterns = [
        'yolo_runs/button_detector/weights/best.pt',
        'yolo_runs/checkin_detector/weights/best.pt',
        'yolo_runs/homepage_detector/weights/best.pt',
        'yolo_runs/login_detector/weights/best.pt',
        'yolo_runs/transfer_detector/weights/best.pt',
    ]
    
    keep_files = set()
    for pattern in keep_patterns:
        full_path = os.path.join(models_dir, pattern.replace('/', os.sep))
        if os.path.exists(full_path):
            keep_files.add(os.path.normpath(full_path))
    
    deleted_count = 0
    for file_path in Path(models_dir).rglob('best.pt'):
        if os.path.normpath(file_path) not in keep_files:
            size = file_path.stat().st_size
            file_path.unlink()
            deleted_count += 1
            total_size += size
    
    print(f"    删除冗余模型: {deleted_count} 个")
    
    # 2. 删除 runs 目录
    runs_dir = os.path.join(models_dir, 'runs')
    if os.path.exists(runs_dir):
        size = sum(f.stat().st_size for f in Path(runs_dir).rglob('*') if f.is_file())
        shutil.rmtree(runs_dir)
        total_size += size
        print(f"    删除训练目录: runs")
    
    # 3. 删除 FFmpeg DLL
    internal_dir = os.path.join(OUTPUT_DIR, '_internal')
    for file_path in Path(internal_dir).rglob('*ffmpeg*.dll'):
        if file_path.is_file():
            size = file_path.stat().st_size
            file_path.unlink()
            total_size += size
    print(f"    删除 FFmpeg DLL")
    
    # 4. 删除 _internal 中重复的 models
    internal_models = os.path.join(internal_dir, 'models')
    if os.path.exists(internal_models):
        size = sum(f.stat().st_size for f in Path(internal_models).rglob('*') if f.is_file())
        shutil.rmtree(internal_models)
        total_size += size
        print(f"    删除 _internal/models")
    
    print(f"  优化完成，释放: {total_size / 1024 / 1024:.2f} MB")
    
    # 显示最终大小
    final_size = sum(f.stat().st_size for f in Path(OUTPUT_DIR).rglob('*') if f.is_file())
    print(f"\n最终大小: {final_size / 1024 / 1024:.2f} MB ({final_size / 1024 / 1024 / 1024:.2f} GB)")
    
    return True

if __name__ == "__main__":
    print("="*60)
    print("完成打包后续步骤")
    print("="*60)
    
    if copy_additional_files():
        if move_to_output():
            optimize_package()
            print("\n打包完成！")
        else:
            print("\n移动失败")
    else:
        print("\n复制文件失败")
