"""
实时监控日志文件，查看缓存保存情况
"""

import time
import os
from pathlib import Path
from datetime import datetime


def monitor_logs():
    """监控日志文件，显示缓存相关信息"""
    
    print("="*80)
    print("开始监控日志文件...")
    print("="*80)
    print("\n等待日志文件生成...\n")
    
    logs_dir = Path("logs")
    
    # 等待日志文件生成
    while True:
        log_files = list(logs_dir.glob("*.log"))
        if log_files:
            break
        time.sleep(1)
    
    # 获取最新的日志文件
    latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
    print(f"监控文件: {latest_log.name}")
    print("="*80)
    print()
    
    # 记录已读取的行数
    last_position = 0
    
    # 关键词列表
    keywords = [
        "保存登录缓存",
        "缓存已保存",
        "缓存保存失败",
        "save_login_cache",
        "用户ID:",
        "user_id",
        "enable_cache",
        "cache_manager",
    ]
    
    try:
        while True:
            # 检查文件是否存在
            if not latest_log.exists():
                print("\n⚠️ 日志文件已删除，重新查找...")
                log_files = list(logs_dir.glob("*.log"))
                if log_files:
                    latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
                    print(f"新的监控文件: {latest_log.name}\n")
                    last_position = 0
                else:
                    time.sleep(1)
                    continue
            
            # 读取新内容
            try:
                with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
                    f.seek(last_position)
                    new_lines = f.readlines()
                    last_position = f.tell()
                
                # 显示包含关键词的行
                for line in new_lines:
                    line = line.strip()
                    if any(keyword in line for keyword in keywords):
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        print(f"[{timestamp}] {line}")
            
            except Exception as e:
                print(f"读取错误: {e}")
            
            time.sleep(0.5)  # 每0.5秒检查一次
    
    except KeyboardInterrupt:
        print("\n\n监控已停止")
        print("="*80)


if __name__ == "__main__":
    monitor_logs()
