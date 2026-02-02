"""
模板文件加密/解密工具
Template File Encryptor/Decryptor
"""

import os
import base64
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.backends import default_backend


class TemplateEncryptor:
    """模板文件加密器"""
    
    # 使用固定的盐值（在实际应用中应该更安全地存储）
    SALT = b'ximeng_template_salt_2026'
    
    def __init__(self, password: str = "ximeng_automation_2026"):
        """初始化加密器
        
        Args:
            password: 加密密码
        """
        self.password = password.encode()
        self.key = self._derive_key()
        self.cipher = Fernet(self.key)
    
    def _derive_key(self) -> bytes:
        """从密码派生加密密钥"""
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.SALT,
            iterations=100000,
            backend=default_backend()
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.password))
        return key
    
    def encrypt_file(self, input_path: str, output_path: str = None) -> str:
        """加密文件
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径（可选，默认为 input_path + .enc）
        
        Returns:
            加密后的文件路径
        """
        if output_path is None:
            output_path = input_path + '.enc'
        
        # 读取原始文件
        with open(input_path, 'rb') as f:
            data = f.read()
        
        # 加密数据
        encrypted_data = self.cipher.encrypt(data)
        
        # 写入加密文件
        with open(output_path, 'wb') as f:
            f.write(encrypted_data)
        
        return output_path
    
    def decrypt_file(self, input_path: str, output_path: str = None) -> str:
        """解密文件
        
        Args:
            input_path: 输入文件路径（加密文件）
            output_path: 输出文件路径（可选）
        
        Returns:
            解密后的文件路径
        """
        if output_path is None:
            # 如果输入文件以 .enc 结尾，去掉后缀
            if input_path.endswith('.enc'):
                output_path = input_path[:-4]
            else:
                output_path = input_path + '.dec'
        
        # 读取加密文件
        with open(input_path, 'rb') as f:
            encrypted_data = f.read()
        
        # 解密数据
        decrypted_data = self.cipher.decrypt(encrypted_data)
        
        # 写入解密文件
        with open(output_path, 'wb') as f:
            f.write(decrypted_data)
        
        return output_path
    
    def decrypt_to_memory(self, input_path: str) -> bytes:
        """解密文件到内存（不写入磁盘）
        
        Args:
            input_path: 输入文件路径（加密文件）
        
        Returns:
            解密后的数据
        """
        # 读取加密文件
        with open(input_path, 'rb') as f:
            encrypted_data = f.read()
        
        # 解密数据
        decrypted_data = self.cipher.decrypt(encrypted_data)
        
        return decrypted_data
    
    def encrypt_directory(self, input_dir: str, output_dir: str = None, 
                         extensions: list = None) -> dict:
        """加密目录中的所有文件
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录（可选，默认为 input_dir + _encrypted）
            extensions: 要加密的文件扩展名列表（默认为图片格式）
        
        Returns:
            加密结果字典 {原始文件: 加密文件}
        """
        if output_dir is None:
            output_dir = input_dir + '_encrypted'
        
        if extensions is None:
            extensions = ['.png', '.jpg', '.jpeg', '.bmp']
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        input_path = Path(input_dir)
        
        # 遍历目录中的所有文件
        for file_path in input_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                # 构建输出文件路径
                output_file = Path(output_dir) / (file_path.name + '.enc')
                
                # 加密文件
                encrypted_path = self.encrypt_file(str(file_path), str(output_file))
                results[str(file_path)] = encrypted_path
                
                print(f"✓ 已加密: {file_path.name} -> {output_file.name}")
        
        return results
    
    def decrypt_directory(self, input_dir: str, output_dir: str = None) -> dict:
        """解密目录中的所有加密文件
        
        Args:
            input_dir: 输入目录（包含加密文件）
            output_dir: 输出目录（可选）
        
        Returns:
            解密结果字典 {加密文件: 解密文件}
        """
        if output_dir is None:
            output_dir = input_dir + '_decrypted'
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        input_path = Path(input_dir)
        
        # 遍历目录中的所有 .enc 文件
        for file_path in input_path.glob('*.enc'):
            if file_path.is_file():
                # 构建输出文件路径（去掉 .enc 后缀）
                output_file = Path(output_dir) / file_path.stem
                
                # 解密文件
                decrypted_path = self.decrypt_file(str(file_path), str(output_file))
                results[str(file_path)] = decrypted_path
                
                print(f"✓ 已解密: {file_path.name} -> {output_file.name}")
        
        return results


def main():
    """命令行工具入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='模板文件加密/解密工具')
    parser.add_argument('action', choices=['encrypt', 'decrypt'], help='操作类型')
    parser.add_argument('input', help='输入路径（文件或目录）')
    parser.add_argument('-o', '--output', help='输出路径（可选）')
    parser.add_argument('-p', '--password', default='ximeng_automation_2026', 
                       help='加密密码（默认：ximeng_automation_2026）')
    
    args = parser.parse_args()
    
    encryptor = TemplateEncryptor(args.password)
    
    if os.path.isfile(args.input):
        # 处理单个文件
        if args.action == 'encrypt':
            result = encryptor.encrypt_file(args.input, args.output)
            print(f"✓ 文件已加密: {result}")
        else:
            result = encryptor.decrypt_file(args.input, args.output)
            print(f"✓ 文件已解密: {result}")
    
    elif os.path.isdir(args.input):
        # 处理目录
        if args.action == 'encrypt':
            results = encryptor.encrypt_directory(args.input, args.output)
            print(f"\n✓ 共加密 {len(results)} 个文件")
        else:
            results = encryptor.decrypt_directory(args.input, args.output)
            print(f"\n✓ 共解密 {len(results)} 个文件")
    
    else:
        print(f"✗ 路径不存在: {args.input}")


if __name__ == '__main__':
    main()
