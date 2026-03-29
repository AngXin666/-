"""检查缓存文件完整性"""
from pathlib import Path
from src.login_cache_manager import LoginCacheManager
from src.adb_bridge import ADBBridge

def check_cache(phone: str):
    """检查指定账号的缓存文件
    
    Args:
        phone: 手机号
    """
    print(f"=" * 60)
    print(f"检查账号 {phone} 的缓存文件")
    print(f"=" * 60)
    
    # 初始化
    adb = ADBBridge()
    cache_manager = LoginCacheManager(adb)
    
    # 获取预期的user_id
    expected_user_id = cache_manager._get_expected_user_id(phone)
    print(f"\n预期用户ID: {expected_user_id or '未知'}")
    
    # 检查缓存目录
    if expected_user_id:
        cache_dir = cache_manager._get_account_cache_dir(phone, expected_user_id)
    else:
        cache_dir = cache_manager._get_account_cache_dir(phone)
    
    print(f"缓存目录: {cache_dir}")
    
    if not cache_dir.exists():
        print("\n❌ 缓存目录不存在")
        return False
    
    print("✓ 缓存目录存在\n")
    
    # 检查每个缓存文件
    print("缓存文件列表:")
    print("-" * 60)
    
    all_required_exist = True
    
    for file_path in cache_manager.CACHE_FILES:
        cache_file_name = file_path.replace('/', '_')
        encrypted_file = cache_dir / f"{cache_file_name}.enc"
        
        is_required = file_path in cache_manager.REQUIRED_FILES
        status = "必需" if is_required else "可选"
        
        if encrypted_file.exists():
            size = encrypted_file.stat().st_size
            print(f"  ✓ {file_path:<35} ({status}) - {size:>6} 字节")
        else:
            if is_required:
                print(f"  ❌ {file_path:<35} ({status}) - 不存在")
                all_required_exist = False
            else:
                print(f"  - {file_path:<35} ({status}) - 不存在（正常）")
    
    print("\n" + "=" * 60)
    
    if all_required_exist:
        print("✓ 所有必需的缓存文件都存在")
        print("\n问题可能是：")
        print("1. 缓存文件恢复后，应用没有正确加载")
        print("2. 应用需要额外的文件或配置")
        print("3. 缓存文件的时间戳或权限问题")
        return True
    else:
        print("❌ 缺少必需的缓存文件")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python check_cache_files.py <手机号>")
        print("示例: python check_cache_files.py 18825724627")
        sys.exit(1)
    
    phone = sys.argv[1]
    check_cache(phone)
