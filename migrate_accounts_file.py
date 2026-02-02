"""
账号文件加密迁移脚本
Migrate Accounts File to Encrypted Format

将明文账号文件升级为加密格式（机器绑定加密）
"""

import sys
from pathlib import Path

# 添加 src 目录到路径
sys.path.insert(0, 'src')

from encrypted_accounts_file import EncryptedAccountsFile


def migrate_accounts_file(accounts_file: str = None):
    """迁移账号文件到加密格式
    
    Args:
        accounts_file: 账号文件路径（可选，默认从配置读取）
    """
    print("=" * 60)
    print("账号文件加密迁移工具")
    print("=" * 60)
    
    # 如果未指定文件，从配置读取
    if not accounts_file:
        try:
            from config import ConfigLoader
            config = ConfigLoader().load()
            accounts_file = config.accounts_file
        except Exception as e:
            print(f"\n❌ 读取配置失败: {e}")
            print("\n请手动指定账号文件路径：")
            print(f"  python {sys.argv[0]} <账号文件路径>")
            return False
    
    if not accounts_file:
        print("\n❌ 未配置账号文件路径")
        return False
    
    print(f"\n账号文件: {accounts_file}")
    
    # 创建加密文件管理器
    encrypted_file = EncryptedAccountsFile(accounts_file)
    
    # 检查文件状态
    has_plain = encrypted_file.has_plain_file()
    has_encrypted = encrypted_file.has_encrypted_file()
    
    print(f"\n文件状态:")
    print(f"  明文文件: {'✓ 存在' if has_plain else '✗ 不存在'}")
    print(f"  加密文件: {'✓ 存在' if has_encrypted else '✗ 不存在'}")
    
    # 如果已经加密，无需迁移
    if has_encrypted:
        print(f"\n✓ 账号文件已加密，无需迁移")
        
        # 验证加密文件
        try:
            accounts = encrypted_file.read_accounts()
            print(f"\n验证加密文件:")
            print(f"  账号数量: {len(accounts)}")
            print(f"  状态: ✓ 正常")
        except Exception as e:
            print(f"\n⚠️ 加密文件验证失败: {e}")
            print(f"\n可能原因:")
            print(f"  1. 文件在其他机器上创建（机器绑定加密）")
            print(f"  2. 文件已损坏")
            print(f"\n建议:")
            print(f"  1. 如果有明文备份，删除加密文件后重新迁移")
            print(f"  2. 如果没有备份，需要重新导入账号")
        
        return True
    
    # 如果没有明文文件，无法迁移
    if not has_plain:
        print(f"\n⚠️ 没有找到账号文件")
        print(f"\n提示:")
        print(f"  1. 请确认账号文件路径是否正确")
        print(f"  2. 如果是新安装，请先添加账号")
        return False
    
    # 执行升级
    print(f"\n正在升级账号文件...")
    print(f"  - 读取明文文件")
    print(f"  - 使用机器绑定加密")
    print(f"  - 保存加密文件")
    print(f"  - 删除明文文件")
    
    try:
        if encrypted_file.upgrade_to_encrypted():
            print(f"\n✓ 迁移成功！")
            
            # 验证加密文件
            accounts = encrypted_file.read_accounts()
            print(f"\n迁移统计:")
            print(f"  账号数量: {len(accounts)}")
            print(f"  加密方式: 机器绑定加密（AES-256-GCM）")
            
            print(f"\n重要提示:")
            print(f"  1. ✓ 账号文件已加密，只能在当前机器使用")
            print(f"  2. ✓ 如果更换机器，需要重新导入账号")
            print(f"  3. ✓ 明文文件已自动删除")
            print(f"  4. ⚠️ 请妥善保管账号信息的备份")
            
            return True
        else:
            print(f"\n✗ 迁移失败")
            return False
    except Exception as e:
        print(f"\n✗ 迁移失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    # 从命令行参数获取账号文件路径
    accounts_file = sys.argv[1] if len(sys.argv) > 1 else None
    
    # 执行迁移
    success = migrate_accounts_file(accounts_file)
    
    # 退出码
    sys.exit(0 if success else 1)
