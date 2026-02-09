import os
from pathlib import Path

# 检查签到奖励为0的账号
phone = "15570754480"
log_file = Path(f"logs/accounts/20260209/{phone}.log")

if log_file.exists():
    print(f"检查账号: {phone}")
    print("=" * 80)
    
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找执行总结部分
    if '执行总结' in content:
        idx = content.find('执行总结')
        summary = content[idx:idx+500]
        print(summary)
    else:
        print("未找到执行总结")
        
    print("\n" + "=" * 80)
    print("查找签到奖励相关信息:")
    print("=" * 80)
    
    lines = content.split('\n')
    for line in lines:
        if '签到奖励' in line or 'checkin_reward' in line:
            print(line)
else:
    print(f"日志文件不存在: {log_file}")
