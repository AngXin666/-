"""深度分析昵称识别模式 - 找出真正的问题"""
import sys
sys.path.insert(0, 'src')

from account_cache import get_account_cache
from collections import Counter
import re

# 获取缓存实例
cache = get_account_cache()

print("=" * 80)
print("昵称模式深度分析")
print("=" * 80)

if hasattr(cache, '_cache'):
    cache_data = cache._cache
    
    # 收集所有昵称
    nicknames = []
    for phone, info in cache_data.items():
        nickname = info.get('nickname', '')
        if nickname:
            nicknames.append(nickname)
    
    print(f"\n总账号数: {len(cache_data)}")
    print(f"有昵称的账号: {len(nicknames)}")
    print(f"空昵称账号: {len(cache_data) - len(nicknames)}")
    
    # 分析1: 昵称长度分布
    print("\n" + "=" * 80)
    print("1. 昵称长度分布")
    print("=" * 80)
    length_counter = Counter([len(n) for n in nicknames])
    for length in sorted(length_counter.keys()):
        count = length_counter[length]
        percentage = count / len(nicknames) * 100
        bar = "█" * int(percentage / 2)
        print(f"  {length:2d}字符: {count:3d}个 ({percentage:5.1f}%) {bar}")
    
    # 分析2: 单字昵称详细分析
    print("\n" + "=" * 80)
    print("2. 单字昵称详细分析（这些最可疑）")
    print("=" * 80)
    single_char_nicknames = [n for n in nicknames if len(n) == 1]
    single_char_counter = Counter(single_char_nicknames)
    
    print(f"单字昵称总数: {len(single_char_nicknames)}")
    print(f"不同的单字: {len(single_char_counter)}")
    print("\n出现次数排序:")
    for char, count in single_char_counter.most_common():
        percentage = count / len(single_char_nicknames) * 100
        print(f"  '{char}': {count}次 ({percentage:.1f}%)")
    
    # 分析3: 6字符昵称分析（可能的"乱码"）
    print("\n" + "=" * 80)
    print("3. 6字符昵称分析（检查是否真的是乱码）")
    print("=" * 80)
    six_char_nicknames = [n for n in nicknames if len(n) == 6]
    print(f"6字符昵称总数: {len(six_char_nicknames)}")
    
    # 检查特征
    mixed_case = []  # 大小写混合
    has_numbers = []  # 包含数字
    all_alpha = []   # 纯字母
    
    for nickname in six_char_nicknames:
        if re.search(r'[a-z]', nickname) and re.search(r'[A-Z]', nickname):
            mixed_case.append(nickname)
        if re.search(r'\d', nickname):
            has_numbers.append(nickname)
        if nickname.isalpha():
            all_alpha.append(nickname)
    
    print(f"\n特征分析:")
    print(f"  大小写混合: {len(mixed_case)}个 ({len(mixed_case)/len(six_char_nicknames)*100:.1f}%)")
    print(f"  包含数字: {len(has_numbers)}个 ({len(has_numbers)/len(six_char_nicknames)*100:.1f}%)")
    print(f"  纯字母: {len(all_alpha)}个 ({len(all_alpha)/len(six_char_nicknames)*100:.1f}%)")
    
    print(f"\n示例（前20个）:")
    for i, nickname in enumerate(six_char_nicknames[:20], 1):
        print(f"  {i:2d}. {nickname}")
    
    # 分析4: 字符类型分布
    print("\n" + "=" * 80)
    print("4. 字符类型分布")
    print("=" * 80)
    
    chinese_only = []    # 纯中文
    english_only = []    # 纯英文
    mixed_lang = []      # 中英混合
    has_special = []     # 包含特殊字符
    
    for nickname in nicknames:
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', nickname))
        has_english = bool(re.search(r'[a-zA-Z]', nickname))
        has_digit = bool(re.search(r'\d', nickname))
        has_special_char = bool(re.search(r'[^a-zA-Z0-9\u4e00-\u9fff]', nickname))
        
        if has_chinese and not has_english and not has_digit:
            chinese_only.append(nickname)
        elif has_english and not has_chinese:
            english_only.append(nickname)
        elif has_chinese and has_english:
            mixed_lang.append(nickname)
        
        if has_special_char:
            has_special.append(nickname)
    
    total = len(nicknames)
    print(f"  纯中文: {len(chinese_only)}个 ({len(chinese_only)/total*100:.1f}%)")
    print(f"  纯英文: {len(english_only)}个 ({len(english_only)/total*100:.1f}%)")
    print(f"  中英混合: {len(mixed_lang)}个 ({len(mixed_lang)/total*100:.1f}%)")
    print(f"  包含特殊字符: {len(has_special)}个 ({len(has_special)/total*100:.1f}%)")
    
    # 分析5: 可疑昵称列表
    print("\n" + "=" * 80)
    print("5. 高度可疑的昵称（需要查看截图验证）")
    print("=" * 80)
    
    suspicious = []
    
    # 规则1: "西"字（出现15次，极不正常）
    if '西' in single_char_counter and single_char_counter['西'] > 5:
        suspicious.append(('西', single_char_counter['西'], '单字重复过多'))
    
    # 规则2: 其他重复的单字
    for char, count in single_char_counter.items():
        if count > 3 and char != '西':
            suspicious.append((char, count, '单字重复'))
    
    # 规则3: 大小写混合的6字符
    for nickname in mixed_case[:10]:
        suspicious.append((nickname, 1, '大小写混合'))
    
    print(f"\n可疑昵称列表:")
    for nickname, count, reason in suspicious:
        print(f"  '{nickname}' - 出现{count}次 - 原因: {reason}")
    
    # 分析6: 找出这些账号的手机号（用于查看截图）
    print("\n" + "=" * 80)
    print("6. 需要检查截图的账号（'西'字昵称）")
    print("=" * 80)
    
    xi_accounts = []
    for phone, info in cache_data.items():
        nickname = info.get('nickname', '')
        if nickname == '西':
            xi_accounts.append(phone)
    
    print(f"'西'字昵称的账号（共{len(xi_accounts)}个）:")
    for phone in xi_accounts[:5]:  # 只显示前5个
        print(f"  {phone}")
    
    if len(xi_accounts) > 5:
        print(f"  ... 还有 {len(xi_accounts) - 5} 个")
    
    print("\n建议: 查看这些账号的截图，确认'西'字是从哪里识别出来的")
    print("截图位置: zdqd/checkin_screenshots/ 或 zdqd/login_cache/{phone}_*/")

else:
    print("无法访问缓存数据")

print("\n" + "=" * 80)
print("分析完成")
print("=" * 80)
