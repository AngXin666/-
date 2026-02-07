"""
测试运行脚本
Test Runner Script

运行所有自动化测试，确保代码质量。
"""

import sys
import os
import pytest

def main():
    """运行测试"""
    print("=" * 60)
    print("运行自动化测试")
    print("=" * 60)
    print()
    
    # 设置测试参数
    args = [
        'tests/test_imports.py',
        'tests/test_core_methods.py',
        '-v',  # 详细输出
        '--tb=short',  # 简短的错误追踪
        '--color=yes',  # 彩色输出
    ]
    
    # 如果有命令行参数，使用它们
    if len(sys.argv) > 1:
        args = sys.argv[1:]
    
    print(f"测试参数: {' '.join(args)}")
    print()
    
    # 运行测试
    exit_code = pytest.main(args)
    
    print()
    print("=" * 60)
    if exit_code == 0:
        print("✅ 所有测试通过！")
    else:
        print("❌ 测试失败！")
    print("=" * 60)
    
    return exit_code

if __name__ == '__main__':
    sys.exit(main())
