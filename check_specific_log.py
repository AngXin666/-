import os
from pathlib import Path

# 检查签到奖励为0的账号: 15570754480
phone = "15570754480"
log_file = Path(f"logs/accounts/20260209/{phone}.log")

if log_file.exists():
    print(f"检查账号: {phone}")
    print("=" * 80)
    
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 查找关键步骤
    in_checkin = False
    in_balance = False
    
    for i, line in enumerate(lines):
        # 签到相关
        if '步骤2' in line or '签到' in line:
            in_checkin = True
        
        # 获取最终余额相关
        if '步骤3' in line or '步骤4' in line or '获取最终余额' in line:
            in_balance = True
        
        # 打印关键信息
        if in_checkin or in_balance:
            if any(kw in line for kw in ['步骤', '签到', '余额', 'balance', 'checkin_balance_after', '签到奖励', '今日已签到', '签到成功']):
                print(line.rstrip())
        
        # 执行总结后停止
        if '执行总结' in line:
            # 打印总结部分
            for j in range(i, min(i+10, len(lines))):
                print(lines[j].rstrip())
            break
else:
    print(f"日志文件不存在: {log_file}")
