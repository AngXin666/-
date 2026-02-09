import os
from pathlib import Path

# 查找最新的账号日志
log_dir = Path("logs/accounts/20260209")
if log_dir.exists():
    log_files = sorted(log_dir.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
    if log_files:
        latest_log = log_files[0]
        print(f"最新日志: {latest_log.name}")
        print("=" * 80)
        
        with open(latest_log, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 查找签到相关信息
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if '签到' in line or '余额' in line or 'balance' in line.lower():
                print(line)
