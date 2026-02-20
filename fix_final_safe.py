"""安全修复 - 使用正则表达式精确替换问题字符"""
import re
import subprocess

def fix_file():
    input_file = "src/daily_checkin.py"
    
    # 读取文件
    with open(input_file, 'rb') as f:
        raw = f.read()
    
    content = raw.decode('utf-8', errors='ignore')
    
    print(f"原始文件大小: {len(content)} 字符")
    
    # 定义要移除的字符范围（使用正则表达式）
    # 私有使用区：U+E000-U+F8FF
    pattern = r'[\uE000-\uF8FF]'
    
    # 统计要移除的字符
    matches = re.findall(pattern, content)
    print(f"找到 {len(matches)} 个私有使用区字符")
    
    # 替换（移除）这些字符
    cleaned = re.sub(pattern, '', content)
    
    print(f"清理后大小: {len(cleaned)} 字符")
    print(f"移除了 {len(content) - len(cleaned)} 个字符")
    
    # 保存
    with open(input_file, 'w', encoding='utf-8', newline='\n') as f:
        f.write(cleaned)
    
    print('✓ 文件已保存')
    
    # 验证语法
    result = subprocess.run(['python', '-m', 'py_compile', input_file], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print('\n✓✓✓ 语法验证通过！修复成功！✓✓✓\n')
        return True
    else:
        print('\n✗ 仍有语法错误:')
        # 提取关键错误信息
        for line in result.stderr.split('\n'):
            if 'File' in line or 'SyntaxError' in line or 'invalid character' in line:
                print(f'  {line}')
        
        # 如果还有其他字符问题，显示字符码
        if 'invalid character' in result.stderr:
            import re
            match = re.search(r"U\+([0-9A-F]+)", result.stderr)
            if match:
                char_code = match.group(1)
                print(f'\n需要额外清理的字符: U+{char_code}')
        
        return False

if __name__ == "__main__":
    fix_file()
