"""清理私有使用区字符"""
import re as regex_module
import subprocess

input_file = "src/daily_checkin.py"

# 读取
with open(input_file, 'rb') as f:
    raw = f.read()

content = raw.decode('utf-8', errors='ignore')
print(f"原始: {len(content)} 字符")

# 移除私有使用区字符 U+E000-U+F8FF
pattern = r'[\uE000-\uF8FF]'
cleaned = regex_module.sub(pattern, '', content)

removed = len(content) - len(cleaned)
print(f"移除了 {removed} 个私有使用区字符")

# 保存
with open(input_file, 'w', encoding='utf-8', newline='\n') as f:
    f.write(cleaned)

print('✓ 已保存')

# 验证
result = subprocess.run(['python', '-m', 'py_compile', input_file], 
                      capture_output=True, text=True)

if result.returncode == 0:
    print('\n✓✓✓ 语法验证通过！✓✓✓\n')
else:
    print('\n✗ 仍有错误:')
    for line in result.stderr.split('\n'):
        if 'File' in line or 'SyntaxError' in line or 'invalid' in line:
            print(f'  {line}')
