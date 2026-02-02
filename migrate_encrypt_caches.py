"""
数据迁移脚本 - 加密现有的缓存文件

v2.0.7 安全更新：
- 将现有的登录缓存文件加密（机器绑定）
- 将 .account_cache.json 加密
- 备份原始文件

使用方法：
    python migrate_encrypt_caches.py
"""

import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

# 添加 src 到路径
sys.path.insert(0, 'src')

from crypto_utils import CryptoUtils


def backup_file(file_path: Path) -> Path:
    """备份文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        备份文件路径
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = Path(str(file_path) + f'.backup_{timestamp}')
    shutil.copy2(file_path, backup_path)
    return backup_path


def migrate_login_cache():
    """迁移登录缓存文件"""
    print("\n" + "=" * 60)
    print("迁移登录缓存文件")
    print("=" * 60)
    
    cache_dir = Path("login_cache")
    if not cache_dir.exists():
        print("  [跳过] login_cache 目录不存在")
        return
    
    crypto = CryptoUtils()
    total_files = 0
    encrypted_files = 0
    skipped_files = 0
    
    # 遍历所有账号目录
    for account_dir in cache_dir.iterdir():
        if not account_dir.is_dir():
            continue
        
        print(f"\n处理账号: {account_dir.name}")
        
        # 遍历目录中的所有文件
        for file_path in account_dir.iterdir():
            # 跳过已加密的文件
            if file_path.suffix == '.enc':
                continue
            
            # 跳过元数据文件
            if file_path.name == 'metadata.txt':
                continue
            
            # 跳过备份文件
            if '.backup_' in file_path.name:
                continue
            
            total_files += 1
            
            try:
                # 读取原始文件
                with open(file_path, 'rb') as f:
                    plain_data = f.read()
                
                # 检查文件大小
                if len(plain_data) == 0:
                    print(f"  [跳过] {file_path.name} (空文件)")
                    skipped_files += 1
                    continue
                
                # 备份原始文件
                backup_path = backup_file(file_path)
                print(f"  [备份] {file_path.name} -> {backup_path.name}")
                
                # 加密
                encrypted_data = crypto.encrypt_with_machine_binding(plain_data)
                
                # 写入加密文件
                encrypted_file = Path(str(file_path) + '.enc')
                with open(encrypted_file, 'wb') as f:
                    f.write(encrypted_data)
                
                # 删除原始文件
                file_path.unlink()
                
                print(f"  [加密] {file_path.name} -> {encrypted_file.name}")
                encrypted_files += 1
                
            except Exception as e:
                print(f"  [错误] 加密失败 {file_path.name}: {e}")
    
    print(f"\n登录缓存迁移完成:")
    print(f"  总文件数: {total_files}")
    print(f"  已加密: {encrypted_files}")
    print(f"  跳过: {skipped_files}")


def migrate_account_cache():
    """迁移账号缓存文件"""
    print("\n" + "=" * 60)
    print("迁移账号缓存文件")
    print("=" * 60)
    
    cache_file = Path(".account_cache.json")
    if not cache_file.exists():
        print("  [跳过] .account_cache.json 不存在")
        return
    
    try:
        crypto = CryptoUtils()
        
        # 读取原始文件
        with open(cache_file, 'rb') as f:
            plain_data = f.read()
        
        # 备份原始文件
        backup_path = backup_file(cache_file)
        print(f"  [备份] {cache_file.name} -> {backup_path.name}")
        
        # 加密
        encrypted_data = crypto.encrypt_with_machine_binding(plain_data)
        
        # 写入加密文件
        encrypted_file = Path(".account_cache.json.enc")
        with open(encrypted_file, 'wb') as f:
            f.write(encrypted_data)
        
        # 删除原始文件
        cache_file.unlink()
        
        print(f"  [加密] {cache_file.name} -> {encrypted_file.name}")
        print(f"\n账号缓存迁移完成")
        
    except Exception as e:
        print(f"  [错误] 加密失败: {e}")


def main():
    """主函数"""
    print("=" * 60)
    print("数据加密迁移脚本 v2.0.7")
    print("=" * 60)
    print("\n此脚本将加密以下数据:")
    print("  1. login_cache/ 目录中的所有缓存文件")
    print("  2. .account_cache.json 文件")
    print("\n加密后的数据只能在当前机器上使用")
    print("原始文件将被备份（.backup_时间戳）")
    
    # 确认
    response = input("\n是否继续？(y/n): ")
    if response.lower() != 'y':
        print("已取消")
        return
    
    # 迁移登录缓存
    migrate_login_cache()
    
    # 迁移账号缓存
    migrate_account_cache()
    
    print("\n" + "=" * 60)
    print("迁移完成！")
    print("=" * 60)
    print("\n重要提示:")
    print("  1. 备份文件已保存，如有问题可以恢复")
    print("  2. 加密后的数据只能在当前机器使用")
    print("  3. 如果更换机器，需要重新登录账号")
    print("  4. 建议测试程序是否正常运行")


if __name__ == '__main__':
    main()
