"""从备份恢复并只移除导致语法错误的字符"""
import re
import subprocess

# 从备份恢复
import shutil
shutil.copy('src/daily_checkin.py.backup_invalid_char', 'src/daily_checkin.py')
print('✓ 已从原始备份恢复')

# 读取文件
with open('src/daily_checkin.py', 'rb') as f:
    content = f.read().decode('utf-8', errors='ignore')

print(f'原始大小: {len(content)} 字符')

# 只移除私有使用区字符（U+E000-U+F8FF）
# 这些是真正的无效字符，不会破坏代码结构
content = re.sub(r'[\uE000-\uF8FF]', '', content)

# 保存
with open('src/daily_checkin.py', 'w', encoding='utf-8', newline='\n') as f:
    f.write(content)

print('✓ 已保存')

# 验证
result = subprocess.run(['python', '-m', 'py_compile', 'src/daily_checkin.py'], 
                      capture_output=True, text=True)

if result.returncode == 0:
    print('\n✓✓✓ 修复成功！✓✓✓\n')
else:
    print('\n还有其他字符需要清理...')
    # 提取错误字符
    import re
    match = re.search(r"U\+([0-9A-F]+)", result.stderr)
    if match:
        char_code = int(match.group(1), 16)
        char = chr(char_code)
        print(f'需要清理: U+{match.group(1)} ({char})')
