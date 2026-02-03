"""检查账号缓存中的昵称数据"""
import sys
sys.path.insert(0, 'src')

from account_cache import get_account_cache

# 获取缓存实例
cache = get_account_cache()

# 获取所有缓存的账号
print("=" * 80)
print("账号缓存中的昵称数据:")
print("=" * 80)
print(f"{'手机号':<15} {'昵称':<20} {'用户ID':<10}")
print("-" * 80)

# 访问内部缓存数据
if hasattr(cache, '_cache'):
    cache_data = cache._cache
    
    count = 0
    for phone, info in cache_data.items():
        nickname = info.get('nickname', 'N/A')
        user_id = info.get('user_id', 'N/A')
        print(f"{phone:<15} {nickname:<20} {user_id:<10}")
        count += 1
    
    print(f"\n总计: {count} 个账号")
    
    # 统计异常昵称
    null_count = sum(1 for info in cache_data.values() if not info.get('nickname'))
    valid_count = sum(1 for info in cache_data.values() if info.get('nickname'))
    
    print(f"\n昵称统计:")
    print(f"  有效昵称: {valid_count}")
    print(f"  空昵称: {null_count}")
    if count > 0:
        print(f"  异常比例: {null_count/count*100:.1f}%")
    
    # 显示异常昵称示例
    print("\n" + "=" * 80)
    print("异常昵称示例（长度<=6）:")
    print("=" * 80)
    print(f"{'手机号':<15} {'昵称':<20} {'用户ID':<10}")
    print("-" * 80)
    
    for phone, info in cache_data.items():
        nickname = info.get('nickname', '')
        user_id = info.get('user_id', 'N/A')
        if nickname and len(nickname) <= 6:
            print(f"{phone:<15} {nickname:<20} {user_id:<10}")
else:
    print("无法访问缓存数据")
