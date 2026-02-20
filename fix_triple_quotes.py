"""修复三引号不匹配问题"""

# 读取文件
with open('src/daily_checkin.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 统计三引号
count = content.count('"""')
print(f'三引号数量: {count}')
print(f'是否成对: {"是" if count % 2 == 0 else "否（不匹配）"}')

# 如果不成对，在文件末尾添加一个三引号
if count % 2 != 0:
    # 确保末尾有换行
    if not content.endswith('\n'):
        content += '\n'
    content += '"""\n'
    
    with open('src/daily_checkin.py', 'w', encoding='utf-8', newline='\n') as f:
        f.write(content)
    
    print('✓ 已在文件末尾添加缺失的三引号')
else:
    print('✓ 三引号已成对，无需修复')
