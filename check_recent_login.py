#!/usr/bin/env python3
"""检查最近的登录活动"""

from pathlib import Path
from datetime import datetime

def check_recent_activity():
    """检查最近的登录和缓存活动"""
    
    print("="*60)
    print("检查最近的登录和缓存活动")
    print("="*60)
    
    # 1. 检查缓存目录
    cache_dir = Path("login_cache")
    if cache_dir.exists():
        print(f"\n📁 缓存目录: {cache_dir}")
        
        # 获取所有账号缓存目录
        account_dirs = [d for d in cache_dir.iterdir() if d.is_dir()]
        account_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        print(f"找到 {len(account_dirs)} 个账号缓存\n")
        
        # 显示最近修改的10个
        for i, account_dir in enumerate(account_dirs[:10], 1):
            mtime = datetime.fromtimestamp(account_dir.stat().st_mtime)
            
            # 检查目录内的文件
            files = list(account_dir.glob("*"))
            if files:
                file_times = [f.stat().st_mtime for f in files]
                oldest_file_time = datetime.fromtimestamp(min(file_times))
                newest_file_time = datetime.fromtimestamp(max(file_times))
                
                print(f"{i}. {account_dir.name}")
                print(f"   目录修改时间: {mtime}")
                print(f"   最旧文件时间: {oldest_file_time}")
                print(f"   最新文件时间: {newest_file_time}")
                
                # 检查是否只更新了目录时间
                if mtime > newest_file_time:
                    print(f"   ⚠️ 目录时间比文件新 - 可能只创建了目录但没保存文件")
                
                # 列出文件
                print(f"   文件列表:")
                for f in files:
                    f_mtime = datetime.fromtimestamp(f.stat().st_mtime)
                    print(f"     - {f.name} ({f_mtime})")
                print()
    
    # 2. 检查日志中的登录活动
    print("\n" + "="*60)
    print("检查日志中的登录活动")
    print("="*60)
    
    log_file = Path("logs/NoxAutomation_20260319.log")
    if log_file.exists():
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # 查找登录相关的行
        login_lines = []
        for line in lines:
            if any(keyword in line for keyword in [
                '登录成功', '登录失败', '开始登录',
                '恢复缓存', '保存缓存',
                '快速签到模式', '完整流程'
            ]):
                login_lines.append(line.strip())
        
        if login_lines:
            print("\n最近的登录活动:")
            for line in login_lines[-30:]:  # 显示最后30行
                print(line)
        else:
            print("未找到登录活动记录")

if __name__ == "__main__":
    check_recent_activity()
