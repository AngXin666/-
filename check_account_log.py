import os
from pathlib import Path

phone = "17773361153"
log_file = Path(f"logs/accounts/20260209/{phone}.log")

if log_file.exists():
    print(f"检查账号: {phone}")
    print("=" * 80)
    
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 查找签到和余额相关信息
    for i, line in enumerate(lines):
        if any(kw in line for kw in ['签到成功', '签到奖励', '余额前', '余额后', 'balance_before', 'checkin_balance_after', 'checkin_reward', '执行总结']):
            print(line.rstrip())
            
            # 如果是执行总结,打印后续10行
            if '执行总结' in line:
                for j in range(i+1, min(i+11, len(lines))):
                    print(lines[j].rstrip())
                break
else:
    print(f"日志文件不存在: {log_file}")
