"""
加密模板文件脚本
Encrypt Template Files Script
"""

import sys
import os
from pathlib import Path

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.template_encryptor import TemplateEncryptor


def main():
    print("="*60)
    print("模板文件加密工具")
    print("="*60)
    
    # 模板目录
    template_dir = "dist/JT"
    encrypted_dir = "dist/JT_encrypted"
    
    if not os.path.exists(template_dir):
        print(f"\n✗ 模板目录不存在: {template_dir}")
        return
    
    print(f"\n输入目录: {template_dir}")
    print(f"输出目录: {encrypted_dir}")
    print(f"\n开始加密...")
    print("-"*60)
    
    # 创建加密器
    encryptor = TemplateEncryptor()
    
    # 加密目录中的所有图片
    results = encryptor.encrypt_directory(template_dir, encrypted_dir)
    
    print("-"*60)
    print(f"\n✓ 加密完成！")
    print(f"  共加密 {len(results)} 个文件")
    print(f"  加密文件保存在: {encrypted_dir}")
    
    print(f"\n下一步操作：")
    print(f"  1. 备份原始模板文件（{template_dir}）")
    print(f"  2. 删除原始模板文件")
    print(f"  3. 将加密文件移动到 {template_dir}")
    print(f"  4. 修改代码以支持加密模板")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
