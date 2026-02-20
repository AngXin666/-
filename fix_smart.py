"""智能修复 - 只移除私有使用区字符（U+E000-U+F8FF）"""

def fix_file():
    input_file = "src/daily_checkin.py"
    
    # 读取文件
    with open(input_file, 'rb') as f:
        raw = f.read()
    
    content = raw.decode('utf-8', errors='ignore')
    
    # 只移除私有使用区字符（这些是真正的无效字符）
    cleaned = ''
    removed = 0
    
    for char in content:
        code = ord(char)
        # 只移除私有使用区字符
        if 0xE000 <= code <= 0xF8FF:
            removed += 1
            continue
        cleaned += char
    
    print(f'移除了 {removed} 个私有使用区字符')
    
    # 保存
    with open(input_file, 'w', encoding='utf-8', newline='\n') as f:
        f.write(cleaned)
    
    print('✓ 文件已保存')
    
    # 验证
    import subprocess
    result = subprocess.run(['python', '-m', 'py_compile', input_file], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        print('✓ 语法验证通过！')
        return True
    else:
        print(f'✗ 仍有语法错误:')
        # 提取错误信息
        lines = result.stderr.split('\n')
        for line in lines:
            if 'SyntaxError' in line or 'invalid character' in line or 'File' in line:
                print(f'  {line}')
        return False

if __name__ == "__main__":
    fix_file()
