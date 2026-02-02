"""
修复时间格式显示 - 将所有小数点格式改为整数
"""
import re

def fix_time_format(file_path):
    """修复文件中的时间格式"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # 模式1: {变量名:.3f}秒 -> {int(变量名)}秒
    # 匹配: {xxx:.3f}秒
    pattern1 = r'\{([a-zA-Z_][a-zA-Z0-9_\.]*):\.3f\}秒'
    content = re.sub(pattern1, r'{int(\1)}秒', content)
    
    # 模式2: {变量名:.2f}秒 -> {int(变量名)}秒
    # 匹配: {xxx:.2f}秒
    pattern2 = r'\{([a-zA-Z_][a-zA-Z0-9_\.]*):\.2f\}秒'
    content = re.sub(pattern2, r'{int(\1)}秒', content)
    
    # 模式3: {表达式:.3f}秒 -> {int(表达式)}秒
    # 匹配: {time.time() - xxx:.3f}秒
    pattern3 = r'\{(time\.time\(\)\s*-\s*[a-zA-Z_][a-zA-Z0-9_\.]*):\.3f\}秒'
    content = re.sub(pattern3, r'{int(\1)}秒', content)
    
    # 模式4: {表达式:.2f}秒 -> {int(表达式)}秒
    # 匹配: {time.time() - xxx:.2f}秒
    pattern4 = r'\{(time\.time\(\)\s*-\s*[a-zA-Z_][a-zA-Z0-9_\.]*):\.2f\}秒'
    content = re.sub(pattern4, r'{int(\1)}秒', content)
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ 已修复文件: {file_path}")
        return True
    else:
        print(f"ℹ️ 文件无需修改: {file_path}")
        return False

if __name__ == '__main__':
    file_path = 'src/ximeng_automation.py'
    fix_time_format(file_path)
    print("\n完成！")
