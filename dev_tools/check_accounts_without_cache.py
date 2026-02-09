"""
检查哪些账号没有登录缓存
"""

import os
from pathlib import Path


def check_accounts_without_cache():
    """检查哪些账号没有登录缓存"""
    
    print("="*80)
    print("检查账号缓存状态")
    print("="*80)
    
    # 简单读取账号文件（假设格式：手机号----密码----归属）
    accounts_file = Path("data/accounts.txt.enc")
    
    if not accounts_file.exists():
        print("\n❌ 账号文件不存在")
        return
    
    # 读取所有账号（从缓存目录的映射文件）
    mapping_file = Path("login_cache/phone_userid_mapping.txt")
    
    if not mapping_file.exists():
        print("\n⚠️ 映射文件不存在，使用缓存目录分析")
    
    # 获取所有缓存目录
    cache_dir = Path("login_cache")
    if not cache_dir.exists():
        print("\n❌ login_cache 目录不存在")
        return
    
    # 获取所有缓存的手机号
    cached_phones = set()
    cache_info = {}
    
    for folder in cache_dir.iterdir():
        if folder.is_dir():
            # 提取手机号（格式：手机号_用户ID）
            parts = folder.name.split("_")
            if len(parts) >= 2:
                phone = parts[0]
                user_id = parts[1]
                cached_phones.add(phone)
                cache_info[phone] = user_id
    
    print(f"\n已有缓存的账号数: {len(cached_phones)}")
    
    # 从映射文件读取所有账号
    all_phones = set()
    
    if mapping_file.exists():
        try:
            with open(mapping_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and '=' in line:
                        phone = line.split('=')[0].strip()
                        all_phones.add(phone)
        except Exception as e:
            print(f"⚠️ 读取映射文件失败: {e}")
    
    # 如果映射文件没有数据，从缓存目录获取
    if not all_phones:
        all_phones = cached_phones.copy()
        print("⚠️ 使用缓存目录作为账号列表")
    
    print(f"总账号数（从映射文件）: {len(all_phones)}")
    
    # 检查哪些账号没有缓存
    phones_without_cache = all_phones - cached_phones
    phones_with_cache = all_phones & cached_phones
    
    # 显示没有缓存的账号
    print("\n" + "="*80)
    print(f"没有缓存的账号 ({len(phones_without_cache)} 个):")
    print("="*80)
    
    if phones_without_cache:
        for i, phone in enumerate(sorted(phones_without_cache), 1):
            print(f"{i:3d}. {phone}")
    else:
        print("\n✓ 所有账号都有缓存")
    
    # 显示有缓存的账号（前20个）
    print("\n" + "="*80)
    print(f"有缓存的账号 ({len(phones_with_cache)} 个，显示前20个):")
    print("="*80)
    
    if phones_with_cache:
        for i, phone in enumerate(sorted(phones_with_cache)[:20], 1):
            user_id = cache_info.get(phone, "未知")
            print(f"{i:3d}. {phone} (ID: {user_id})")
        
        if len(phones_with_cache) > 20:
            print(f"     ... 还有 {len(phones_with_cache) - 20} 个账号")
    
    print("\n" + "="*80)
    print("总结:")
    print(f"  总账号数: {len(all_phones)}")
    print(f"  有缓存: {len(phones_with_cache)}")
    print(f"  无缓存: {len(phones_without_cache)}")
    print("="*80)


if __name__ == "__main__":
    check_accounts_without_cache()
