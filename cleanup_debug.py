#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""清理调试日志和修复注释"""

with open('src/ximeng_automation.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"原始行数: {len(lines)}")

# 删除第1722行的调试日志
if "[调试] 接收到的签到结果" in lines[1721]:
    print(f"删除第1722行: {lines[1721].strip()}")
    del lines[1721]

# 修复注释中的 success'):
for i, line in enumerate(lines):
    if "# 保存签到后余额（用于计算签到奖励）success'):" in line:
        lines[i] = line.replace("success'):", "")
        print(f"修复第{i+1}行: 删除 success'):")
        break

with open('src/ximeng_automation.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f"\n✓ 清理完成")
print(f"  修复后行数: {len(lines)}")
