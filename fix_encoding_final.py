"""最终修复方案 - 只移除真正的非打印控制字符"""

def is_valid_char(char):
    """判断字符是否有效"""
    code = ord(char)
    
    # 保留ASCII可打印字符和常见空白
    if 32 <= code <= 126 or char in ['\t', '\n', '\r']:
        return True
    
    # 保留所有Unicode字符（除了私有使用区和特殊控制字符）
    # 私有使用区：U+E000-U+F8FF, U+F0000-U+FFFFD, U+100000-U+10FFFD
    if 0xE000 <= code <= 0xF8FF:
        return False  # 私有使用区
    
    # 保留其他所有Unicode字符（包括中文、标点等）
    if code >= 128:
        return True
    
    # 其他ASCII控制字符（除了\t, \n, \r）
    return False

def fix_file():
    input_file = "src/daily_checkin.py"
    
    print(f"正在修复文件: {input_file}")
    
    # 读取文件
    with open(input_file, 'rb') as f:
        raw_bytes = f.read()
    
    # 解码
    content = raw_bytes.decode('utf-8', errors='ignore')
    
    # 清理
    cleaned = ''
    removed_count = 0
    
    for char in content:
        if is_valid_char(char):
            cleaned += char
        else:
            removed_count += 1
    
    print(f"✓ 移除了 {removed_count} 个无效字符")
    
    # 保存
    with open(input_file, 'w', encoding='utf-8', newline='\n') as f:
        f.write(cleaned)
    
    print(f"✓ 文件已保存")
    
    # 验证
    import subprocess
    result = subprocess.run(['python', '-m', 'py_compile', input_file], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        print("✓ 语法验证通过！")
        return True
    else:
        print(f"✗ 语法验证失败:\n{result.stderr}")
        return False

if __name__ == "__main__":
    fix_file()
