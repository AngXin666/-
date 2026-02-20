#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
监控打包后的EXE运行状态
实时追踪进程、内存、CPU使用情况
"""

import psutil
import time
import os
from datetime import datetime

def get_process_info(proc):
    """获取进程详细信息"""
    try:
        info = {
            'pid': proc.pid,
            'name': proc.name(),
            'status': proc.status(),
            'cpu_percent': proc.cpu_percent(interval=0.1),
            'memory_mb': proc.memory_info().rss / 1024 / 1024,
            'num_threads': proc.num_threads(),
            'create_time': datetime.fromtimestamp(proc.create_time()).strftime('%H:%M:%S')
        }
        return info
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None

def monitor_exe():
    """监控打包后的EXE"""
    print("=" * 80)
    print("监控打包后的EXE运行状态")
    print("=" * 80)
    print("\n按 Ctrl+C 停止监控\n")
    
    exe_name = "溪盟商城自动化助手.exe"
    tracked_pids = set()
    process_count = 0
    
    try:
        while True:
            # 查找所有匹配的进程
            current_pids = set()
            processes = []
            
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    if proc.info['name'] == exe_name:
                        current_pids.add(proc.info['pid'])
                        processes.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # 检测新进程
            new_pids = current_pids - tracked_pids
            if new_pids:
                for pid in new_pids:
                    process_count += 1
                    print(f"\n{'='*80}")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🆕 检测到新进程 #{process_count}")
                    print(f"PID: {pid}")
                    print(f"{'='*80}")
            
            # 检测进程消失
            disappeared_pids = tracked_pids - current_pids
            if disappeared_pids:
                for pid in disappeared_pids:
                    print(f"\n{'='*80}")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ 进程已退出")
                    print(f"PID: {pid}")
                    print(f"{'='*80}")
            
            # 更新追踪列表
            tracked_pids = current_pids
            
            # 显示当前所有进程状态
            if processes:
                print(f"\r[{datetime.now().strftime('%H:%M:%S')}] ", end='')
                print(f"运行中: {len(processes)} 个进程 | ", end='')
                
                for proc in processes:
                    info = get_process_info(proc)
                    if info:
                        print(f"PID:{info['pid']} ", end='')
                        print(f"CPU:{info['cpu_percent']:.1f}% ", end='')
                        print(f"内存:{info['memory_mb']:.0f}MB ", end='')
                        print(f"线程:{info['num_threads']} | ", end='')
                
                # 检测异常情况
                if len(processes) > 1:
                    print(f"\n⚠️  警告：检测到多个进程同时运行！", end='')
                
                # 检测高CPU使用
                for proc in processes:
                    info = get_process_info(proc)
                    if info and info['cpu_percent'] > 80:
                        print(f"\n⚠️  警告：PID {info['pid']} CPU使用率过高 ({info['cpu_percent']:.1f}%)", end='')
                
                # 检测内存泄漏
                for proc in processes:
                    info = get_process_info(proc)
                    if info and info['memory_mb'] > 2000:
                        print(f"\n⚠️  警告：PID {info['pid']} 内存使用过高 ({info['memory_mb']:.0f}MB)", end='')
            else:
                print(f"\r[{datetime.now().strftime('%H:%M:%S')}] 等待程序启动...", end='')
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n监控已停止")
        print(f"\n总计检测到 {process_count} 个进程启动")
        
        if len(tracked_pids) > 0:
            print(f"当前仍有 {len(tracked_pids)} 个进程在运行")
            print("\n是否要终止所有进程？(y/n): ", end='')
            choice = input().strip().lower()
            if choice == 'y':
                for pid in tracked_pids:
                    try:
                        proc = psutil.Process(pid)
                        proc.terminate()
                        print(f"✓ 已终止进程 PID: {pid}")
                    except:
                        pass

if __name__ == "__main__":
    monitor_exe()
