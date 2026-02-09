"""
测试缓存保存修复
验证登录后缓存是否正确保存
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_check_missing_accounts():
    """检查哪些账号没有缓存"""
    print("="*60)
    print("检查缺失缓存的账号")
    print("="*60)
    
    # 读取账号映射文件
    account_mapping_file = "runtime_data/account_user_mapping.json"
    import json
    with open(account_mapping_file, 'r', encoding='utf-8') as f:
        account_mapping = json.load(f)
    
    # 读取缓存映射文件
    cache_mapping_file = "login_cache/phone_userid_mapping.txt"
    cached_phones = set()
    if os.path.exists(cache_mapping_file):
        with open(cache_mapping_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '=' in line:
                    phone = line.split('=')[0]
                    cached_phones.add(phone)
    
    # 找出没有缓存的账号
    all_phones = set(account_mapping.keys())
    missing_phones = all_phones - cached_phones
    
    print(f"\n总账号数: {len(all_phones)}")
    print(f"有缓存: {len(cached_phones)}")
    print(f"无缓存: {len(missing_phones)}")
    
    if missing_phones:
        print(f"\n没有缓存的账号 ({len(missing_phones)} 个):")
        for i, phone in enumerate(sorted(missing_phones), 1):
            owner = account_mapping.get(phone, "未知")
            print(f"  {i}. {phone} (所有者: {owner})")
    else:
        print("\n✓ 所有账号都有缓存")
    
    return missing_phones

def test_check_recent_cache_saves():
    """检查最近是否有缓存保存"""
    print("\n" + "="*60)
    print("检查最近的缓存保存")
    print("="*60)
    
    import time
    from datetime import datetime, timedelta
    
    cache_dir = "login_cache"
    now = time.time()
    one_hour_ago = now - 3600
    
    recent_saves = []
    
    # 遍历所有缓存目录
    for item in os.listdir(cache_dir):
        item_path = os.path.join(cache_dir, item)
        if os.path.isdir(item_path):
            # 检查目录的修改时间
            mtime = os.path.getmtime(item_path)
            if mtime > one_hour_ago:
                phone = item.split('_')[0] if '_' in item else item
                user_id = item.split('_')[1] if '_' in item else "未知"
                time_diff = now - mtime
                recent_saves.append((phone, user_id, mtime, time_diff))
    
    if recent_saves:
        print(f"\n✓ 找到 {len(recent_saves)} 个最近1小时内保存的缓存:")
        for phone, user_id, mtime, time_diff in sorted(recent_saves, key=lambda x: x[2], reverse=True):
            dt = datetime.fromtimestamp(mtime)
            minutes_ago = int(time_diff / 60)
            print(f"  - {phone} (ID: {user_id})")
            print(f"    保存时间: {dt.strftime('%Y-%m-%d %H:%M:%S')} ({minutes_ago}分钟前)")
    else:
        print("\n✗ 最近1小时内没有缓存保存")
        print("  这可能意味着缓存保存功能没有正常工作")
    
    return len(recent_saves) > 0

if __name__ == "__main__":
    print("测试缓存保存修复")
    print("="*60)
    
    # 测试1: 检查缺失的账号
    missing = test_check_missing_accounts()
    
    # 测试2: 检查最近的缓存保存
    has_recent_saves = test_check_recent_cache_saves()
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    if missing:
        print(f"✗ 发现 {len(missing)} 个账号没有缓存")
        print("  建议：重新运行这些账号，观察是否保存缓存")
    else:
        print("✓ 所有账号都有缓存")
    
    if has_recent_saves:
        print("✓ 最近1小时内有缓存保存")
    else:
        print("✗ 最近1小时内没有缓存保存")
        print("  建议：运行一个账号，观察日志中是否出现'保存登录缓存'和'✓ 登录缓存已保存'")
