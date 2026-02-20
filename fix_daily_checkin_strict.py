"""严格修复 daily_checkin.py 文件 - 移除所有无效Unicode字符"""
import re

def fix_file_strict():
    input_file = "src/daily_checkin.py"
    
    print(f"正在严格修复文件: {input_file}")
    
    # 读取原始字节
    with open(input_file, 'rb') as f:
        raw_bytes = f.read()
    
    print(f"✓ 文件大小: {len(raw_bytes)} 字节")
    
    # 尝试用UTF-8解码，忽略错误
    try:
        content = raw_bytes.decode('utf-8', errors='ignore')
        print(f"✓ UTF-8解码成功，字符数: {len(content)}")
    except Exception as e:
        print(f"✗ UTF-8解码失败: {e}")
        return False
    
    # 清理策略：只保留以下字符
    # 1. ASCII可打印字符 (32-126)
    # 2. 常见空白字符：空格、制表符、换行符
    # 3. 中文字符范围（CJK统一汉字）：U+4E00 到 U+9FFF
    # 4. 其他常见中文标点：U+3000-U+303F, U+FF00-U+FFEF
    
    cleaned_lines = []
    total_removed = 0
    problem_lines = []
    
    for line_num, line in enumerate(content.split('\n'), 1):
        cleaned_line = ''
        line_removed = 0
        
        for char_pos, char in enumerate(line, 1):
            code = ord(char)
            
            # 保留的字符范围
            keep = False
            
            # ASCII可打印字符和常见空白
            if 32 <= code <= 126 or char in ['\t', ' ']:
                keep = True
            # CJK统一汉字
            elif 0x4E00 <= code <= 0x9FFF:
                keep = True
            # CJK符号和标点
            elif 0x3000 <= code <= 0x303F:
                keep = True
            # 全角ASCII、全角标点
            elif 0xFF00 <= code <= 0xFFEF:
                keep = True
            
            if keep:
                cleaned_line += char
            else:
                line_removed += 1
                total_removed += 1
                if line_num not in [l[0] for l in problem_lines]:
                    problem_lines.append((line_num, f"U+{code:04X}"))
        
        cleaned_lines.append(cleaned_line)
        
        if line_removed > 0:
            print(f"  第 {line_num} 行: 移除了 {line_removed} 个无效字符")
    
    print(f"\n✓ 总共移除了 {total_removed} 个无效字符")
    print(f"✓ 涉及 {len(problem_lines)} 行")
    
    if problem_lines[:10]:
        print(f"\n前10个问题行:")
        for line_num, char_code in problem_lines[:10]:
            print(f"  行 {line_num}: 包含 {char_code}")
    
    # 重新组合内容
    cleaned_content = '\n'.join(cleaned_lines)
    
    # 保存为UTF-8编码
    try:
        with open(input_file, 'w', encoding='utf-8', newline='\n') as f:
            f.write(cleaned_content)
        print(f"\n✓ 文件已保存: {input_file}")
        print(f"✓ 总行数: {len(cleaned_lines)}")
        return True
    except Exception as e:
        print(f"\n✗ 保存文件失败: {e}")
        return False

if __name__ == "__main__":
    success = fix_file_strict()
    if success:
        print("\n✓ 严格修复完成！")
        print("\n正在验证语法...")
        import subprocess
        result = subprocess.run(['python', '-m', 'py_compile', 'src/daily_checkin.py'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ 语法验证通过！")
        else:
            print(f"✗ 语法验证失败:\n{result.stderr}")
    else:
        print("\n✗ 严格修复失败！")
