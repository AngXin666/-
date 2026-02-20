"""修复 daily_checkin.py 文件的编码问题和非打印字符"""

def fix_encoding():
    input_file = "src/daily_checkin.py"
    output_file = "src/daily_checkin.py"
    
    print(f"正在修复文件: {input_file}")
    
    # 尝试多种编码读取文件
    encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'utf-8-sig']
    content = None
    used_encoding = None
    
    for encoding in encodings:
        try:
            with open(input_file, 'r', encoding=encoding, errors='ignore') as f:
                content = f.read()
            used_encoding = encoding
            print(f"✓ 成功使用 {encoding} 编码读取文件")
            break
        except Exception as e:
            print(f"✗ 使用 {encoding} 编码失败: {e}")
            continue
    
    if content is None:
        print("✗ 无法读取文件")
        return False
    
    # 清理非打印字符（保留常见的空白字符）
    cleaned_lines = []
    removed_chars = set()
    
    for line_num, line in enumerate(content.split('\n'), 1):
        cleaned_line = ''
        for char in line:
            # 保留可打印字符、制表符、空格
            if char.isprintable() or char in ['\t', ' ']:
                cleaned_line += char
            else:
                # 记录被移除的字符
                removed_chars.add(f"U+{ord(char):04X}")
        cleaned_lines.append(cleaned_line)
    
    if removed_chars:
        print(f"✓ 清理了以下非打印字符: {', '.join(sorted(removed_chars))}")
    else:
        print("✓ 未发现非打印字符")
    
    # 重新组合内容
    cleaned_content = '\n'.join(cleaned_lines)
    
    # 保存为UTF-8编码
    try:
        with open(output_file, 'w', encoding='utf-8', newline='\n') as f:
            f.write(cleaned_content)
        print(f"✓ 文件已保存为 UTF-8 编码: {output_file}")
        print(f"✓ 总行数: {len(cleaned_lines)}")
        return True
    except Exception as e:
        print(f"✗ 保存文件失败: {e}")
        return False

if __name__ == "__main__":
    success = fix_encoding()
    if success:
        print("\n✓ 修复完成！")
    else:
        print("\n✗ 修复失败！")
