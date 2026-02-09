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
            lines = f.readlines()
            
        # 查找关键信息
        for i, line in enumerate(lines):
            if any(keyword in line for keyword in ['步骤4', '获取最终余额', 'checkin_balance_after', '签到奖励', '签到成功']):
                # 打印前后3行
                start = max(0, i-2)
                end = min(len(lines), i+3)
                for j in range(start, end):
                    prefix = ">>> " if j == i else "    "
                    print(f"{prefix}{lines[j].rstrip()}")
                print()
