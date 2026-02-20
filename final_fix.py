"""最终修复 - 清理所有导致语法错误的字符"""
import re
import subprocess

# 从备份读取
with open('src/daily_checkin.py.backup_invalid_char', 'rb') as f:
    raw = f.read()

content = raw.decode('utf-8', errors='ignore')
print(f'原始大小: {len(content)} 字符')

# 定义所有需要清理的字符范围
patterns = [
    (r'[\uE000-\uF8FF]', '私有使用区'),
    (r'[\u20A0-\u20CF]', '货币符号'),
    (r'[\u3000-\u303F]', 'CJK符号和标点'),
    (r'[\u3200-\u32FF]', '带圈字符'),
    (r'[\uFF00-\uFFEF]', '全角字符'),
    (r'[\u2100-\u214F]', '字母式符号'),
    (r'[\u2500-\u257F]', '制表符绘图字符'),
]

total_removed = 0
for pattern, name in patterns:
    before = len(content)
    content = re.sub(pattern, '', content)
    after = len(content)
    removed = before - after
    if removed > 0:
        print(f'清理{name}: {removed}个字符')
        total_removed += removed

print(f'\n总共清理: {total_removed}个字符')

# 保存
with open('src/daily_checkin.py', 'w', encoding='utf-8', newline='\n') as f:
    f.write(content)

print('✓ 文件已保存')

# 验证
result = subprocess.run(['python', '-m', 'py_compile', 'src/daily_checkin.py'], 
                      capture_output=True, text=True)

if result.returncode == 0:
    print('\n✓✓✓ 修复成功！文件可以正常导入！✓✓✓')
else:
    print('\n还有问题，查看错误:')
    print(result.stderr[:500])
