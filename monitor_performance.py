"""
性能监控工具 - 实时监控主程序运行性能
"""

import time
import re
from pathlib import Path
from datetime import datetime

def monitor_log_file(log_file='logs/debug_20260128.log'):
    """监控日志文件，实时显示性能数据"""
    
    log_path = Path(log_file)
    if not log_path.exists():
        print(f"日志文件不存在: {log_file}")
        print("等待日志文件创建...")
        while not log_path.exists():
            time.sleep(1)
    
    print("=" * 80)
    print("性能监控工具 - 实时监控主程序运行")
    print("=" * 80)
    print(f"监控日志: {log_file}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()
    
    # 打开日志文件
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        # 移动到文件末尾
        f.seek(0, 2)
        
        # 性能统计
        step_times = {}
        current_step = None
        step_start_time = None
        
        print("等待日志输出...")
        print()
        
        while True:
            line = f.readline()
            
            if not line:
                time.sleep(0.1)
                continue
            
            line = line.strip()
            
            # 检测关键步骤
            if '[签到]' in line or '[登录]' in line or '[导航]' in line:
                print(f"  {line}")
                
                # 提取耗时信息
                time_match = re.search(r'耗时[：:]\s*(\d+\.?\d*)\s*秒', line)
                if time_match:
                    elapsed = float(time_match.group(1))
                    
                    # 高亮显示耗时
                    if elapsed < 1.0:
                        color = '\033[92m'  # 绿色 - 很快
                    elif elapsed < 3.0:
                        color = '\033[93m'  # 黄色 - 正常
                    else:
                        color = '\033[91m'  # 红色 - 较慢
                    
                    print(f"    {color}⏱️  耗时: {elapsed:.2f}秒\033[0m")
                    print()
            
            # 检测YOLO检测
            elif '[YOLO]' in line:
                if '检测到' in line or '未检测到' in line:
                    print(f"  {line}")
            
            # 检测智能等待器
            elif '[智能等待]' in line or '智能等待' in line:
                print(f"  {line}")
            
            # 检测GPU信息
            elif 'GPU' in line and ('加速' in line or 'CUDA' in line):
                print(f"  {line}")
                print()
            
            # 检测签到结果
            elif '签到完成' in line or '获得奖励' in line:
                print(f"\n{'='*60}")
                print(f"  ✓ {line}")
                print(f"{'='*60}\n")
            
            # 检测错误
            elif '错误' in line or '失败' in line or '❌' in line:
                print(f"  \033[91m{line}\033[0m")

if __name__ == '__main__':
    try:
        monitor_log_file()
    except KeyboardInterrupt:
        print("\n\n监控已停止")
    except Exception as e:
        print(f"\n监控出错: {e}")
