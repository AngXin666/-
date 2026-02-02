"""
解密模板文件脚本（用于测试或恢复）
Decrypt Template Files Script
"""

import sys
import os
from pathlib import Path

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.template_encryptor import TemplateEncryptor


def main():
    print("="*60)
    print("模板文件解密工具")
    print("="*60)
    
    # 加密模板目录
    encrypted_dir = "dist/JT_encrypted"
    decrypted_dir = "dist/JT_decrypted"
    
    if not os.path.exists(encrypted_dir):
        print(f"\n✗ 加密目录不存在: {encrypted_dir}")
        return
    
    print(f"\n输入目录: {encrypted_dir}")
    print(f"输出目录: {decrypted_dir}")
    print(f"\n开始解密...")
    print("-"*60)
    
    # 创建解密器
    encryptor = TemplateEncryptor()
    
    # 解密目录中的所有加密文件
    results = encryptor.decrypt_directory(encrypted_dir, decrypted_dir)
    
    print("-"*60)
    print(f"\n✓ 解密完成！")
    print(f"  共解密 {len(results)} 个文件")
    print(f"  解密文件保存在: {decrypted_dir}")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
