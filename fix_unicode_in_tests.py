"""
修复所有测试文件中的Unicode符号
"""

import os
import glob

# Unicode符号替换映射
replacements = {
    '✓': '[OK]',
    '✗': '[ERROR]',
    '⚠': '[WARNING]',
    '❌': '[FAILED]',
    '✅': '[PASSED]',
    '⊘': '[SKIPPED]',
    '📋': '[INFO]'
}

# 需要处理的文件模式
patterns = [
    'test_*.py',
    'tests/**/*.py',
]

def fix_file(filepath):
    """修复单个文件"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否包含Unicode符号
        has_unicode = any(symbol in content for symbol in replacements.keys())
        
        if has_unicode:
            # 替换所有Unicode符号
            for old, new in replacements.items():
                content = content.replace(old, new)
            
            # 写回文件
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"[OK] 已修复: {filepath}")
            return True
        else:
            return False
    except Exception as e:
        print(f"[ERROR] 处理失败 {filepath}: {e}")
        return False

def main():
    """主函数"""
    print("=" * 80)
    print("修复测试文件中的Unicode符号")
    print("=" * 80)
    
    fixed_count = 0
    total_count = 0
    
    # 处理所有匹配的文件
    for pattern in patterns:
        for filepath in glob.glob(pattern, recursive=True):
            if os.path.isfile(filepath):
                total_count += 1
                if fix_file(filepath):
                    fixed_count += 1
    
    print("\n" + "=" * 80)
    print(f"处理完成: 共 {total_count} 个文件，修复 {fixed_count} 个文件")
    print("=" * 80)

if __name__ == "__main__":
    main()
