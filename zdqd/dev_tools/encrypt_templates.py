"""
加密模板文件工具
将 dist/JT 目录中的所有模板文件加密，防止用户查看
"""
import sys
import os
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from crypto_utils import crypto


def encrypt_templates():
    """加密所有模板文件"""
    template_dir = Path("dist/JT")
    
    if not template_dir.exists():
        print("❌ 错误: dist/JT 目录不存在")
        return False
    
    # 获取所有文件
    files = list(template_dir.glob("*"))
    if not files:
        print("❌ 错误: 没有找到模板文件")
        return False
    
    print(f"找到 {len(files)} 个模板文件")
    print()
    
    encrypted_count = 0
    skipped_count = 0
    
    for file_path in files:
        if file_path.is_file():
            # 跳过已加密的文件
            if file_path.suffix == '.encrypted':
                skipped_count += 1
                continue
            
            try:
                # 读取原始文件
                with open(file_path, 'rb') as f:
                    data = f.read()
                
                # 加密
                encrypted_data = crypto.encrypt_file_content(data)
                
                # 保存加密文件
                encrypted_path = file_path.with_suffix(file_path.suffix + '.encrypted')
                with open(encrypted_path, 'wb') as f:
                    f.write(encrypted_data)
                
                # 删除原始文件
                file_path.unlink()
                
                print(f"✅ {file_path.name} → {encrypted_path.name}")
                encrypted_count += 1
                
            except Exception as e:
                print(f"❌ 加密失败 {file_path.name}: {e}")
    
    print()
    print(f"加密完成: {encrypted_count} 个文件")
    if skipped_count > 0:
        print(f"跳过: {skipped_count} 个已加密文件")
    
    return True


def decrypt_templates():
    """解密所有模板文件（用于开发调试）"""
    template_dir = Path("dist/JT")
    
    if not template_dir.exists():
        print("❌ 错误: dist/JT 目录不存在")
        return False
    
    # 获取所有加密文件
    files = list(template_dir.glob("*.encrypted"))
    if not files:
        print("❌ 错误: 没有找到加密的模板文件")
        return False
    
    print(f"找到 {len(files)} 个加密文件")
    print()
    
    decrypted_count = 0
    
    for file_path in files:
        try:
            # 读取加密文件
            with open(file_path, 'rb') as f:
                encrypted_data = f.read()
            
            # 解密
            data = crypto.decrypt_file_content(encrypted_data)
            
            # 保存解密文件（去掉 .encrypted 后缀）
            original_path = file_path.with_suffix('')
            with open(original_path, 'wb') as f:
                f.write(data)
            
            # 删除加密文件
            file_path.unlink()
            
            print(f"✅ {file_path.name} → {original_path.name}")
            decrypted_count += 1
            
        except Exception as e:
            print(f"❌ 解密失败 {file_path.name}: {e}")
    
    print()
    print(f"解密完成: {decrypted_count} 个文件")
    
    return True


if __name__ == "__main__":
    print("========================================")
    print("模板文件加密工具")
    print("========================================")
    print()
    print("选择操作:")
    print("  1 - 加密模板文件 (发布给用户)")
    print("  2 - 解密模板文件 (开发调试)")
    print()
    
    choice = input("请选择 (1 或 2): ").strip()
    print()
    
    if choice == "1":
        print("正在加密模板文件...")
        print()
        if encrypt_templates():
            print()
            print("✅ 模板文件已加密，可以安全发布给用户")
            print("⚠️  用户无法查看模板文件内容")
    elif choice == "2":
        print("正在解密模板文件...")
        print()
        if decrypt_templates():
            print()
            print("✅ 模板文件已解密，可以查看和编辑")
            print("⚠️  发布前记得重新加密")
    else:
        print("❌ 无效的选择")
