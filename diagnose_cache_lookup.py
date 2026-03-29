"""
诊断缓存查找问题
检查 phone_userid_mapping.txt 和缓存目录，找出哪些账号有缓存但找不到
"""

from pathlib import Path
import os

def main():
    print("=" * 80)
    print("缓存查找诊断工具")
    print("=" * 80)
    
    cache_dir = Path("login_cache")
    mapping_file = cache_dir / "phone_userid_mapping.txt"
    
    # 1. 检查映射文件
    print("\n【1. 检查 phone_userid_mapping.txt】")
    if not mapping_file.exists():
        print("❌ 映射文件不存在！")
        return
    
    print(f"✓ 映射文件存在: {mapping_file}")
    
    # 读取映射
    mappings = {}
    with open(mapping_file, "r", encoding="utf-8") as f:
        for line in f:
            if '=' in line:
                phone, user_id = line.strip().split('=', 1)
                mappings[phone] = user_id
    
    print(f"✓ 共有 {len(mappings)} 个账号映射")
    print("\n映射内容（前10个）：")
    for i, (phone, user_id) in enumerate(list(mappings.items())[:10]):
        print(f"  {i+1}. {phone} -> {user_id}")
    
    if len(mappings) > 10:
        print(f"  ... 还有 {len(mappings) - 10} 个")
    
    # 2. 检查缓存目录
    print("\n【2. 检查缓存目录】")
    if not cache_dir.exists():
        print("❌ 缓存目录不存在！")
        return
    
    # 获取所有缓存目录
    cache_dirs = []
    for item in cache_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            cache_dirs.append(item)
    
    print(f"✓ 共有 {len(cache_dirs)} 个缓存目录")
    
    # 3. 分析缓存目录格式
    print("\n【3. 分析缓存目录格式】")
    old_format_dirs = []  # 只有手机号
    new_format_dirs = []  # 手机号_用户ID
    
    for cache_dir_item in cache_dirs:
        dir_name = cache_dir_item.name
        if '_' in dir_name:
            new_format_dirs.append(cache_dir_item)
        else:
            old_format_dirs.append(cache_dir_item)
    
    print(f"  旧格式（只有手机号）: {len(old_format_dirs)} 个")
    print(f"  新格式（手机号_用户ID）: {len(new_format_dirs)} 个")
    
    # 4. 检查每个映射的账号是否有对应的缓存目录
    print("\n【4. 检查映射账号的缓存目录】")
    
    found_count = 0
    not_found_count = 0
    not_found_list = []
    
    for phone, user_id in mappings.items():
        # 检查新格式目录
        new_format_dir = cache_dir / f"{phone}_{user_id}"
        old_format_dir = cache_dir / phone
        
        has_new = new_format_dir.exists()
        has_old = old_format_dir.exists()
        
        if has_new or has_old:
            found_count += 1
        else:
            not_found_count += 1
            not_found_list.append((phone, user_id))
    
    print(f"  ✓ 找到缓存: {found_count} 个")
    print(f"  ❌ 未找到缓存: {not_found_count} 个")
    
    if not_found_list:
        print("\n  未找到缓存的账号（前10个）：")
        for i, (phone, user_id) in enumerate(not_found_list[:10]):
            print(f"    {i+1}. {phone} (user_id: {user_id})")
        if len(not_found_list) > 10:
            print(f"    ... 还有 {len(not_found_list) - 10} 个")
    
    # 5. 检查缓存目录中的文件
    print("\n【5. 检查缓存文件完整性】")
    
    required_files = [
        "shared_prefs_lcdpr.xml.enc",
        "databases_DCStorage.enc"
    ]
    
    complete_count = 0
    incomplete_count = 0
    incomplete_list = []
    
    for cache_dir_item in cache_dirs:
        has_all_files = True
        missing_files = []
        
        for required_file in required_files:
            file_path = cache_dir_item / required_file
            if not file_path.exists():
                has_all_files = False
                missing_files.append(required_file)
        
        if has_all_files:
            complete_count += 1
        else:
            incomplete_count += 1
            incomplete_list.append((cache_dir_item.name, missing_files))
    
    print(f"  ✓ 完整缓存: {complete_count} 个")
    print(f"  ⚠️ 不完整缓存: {incomplete_count} 个")
    
    if incomplete_list:
        print("\n  不完整的缓存（前5个）：")
        for i, (dir_name, missing) in enumerate(incomplete_list[:5]):
            print(f"    {i+1}. {dir_name}")
            print(f"       缺失文件: {', '.join(missing)}")
        if len(incomplete_list) > 5:
            print(f"    ... 还有 {len(incomplete_list) - 5} 个")
    
    # 6. 检查是否有缓存目录但没有映射
    print("\n【6. 检查孤立的缓存目录（有缓存但没有映射）】")
    
    orphan_dirs = []
    for cache_dir_item in cache_dirs:
        dir_name = cache_dir_item.name
        
        # 提取手机号
        if '_' in dir_name:
            phone = dir_name.split('_')[0]
        else:
            phone = dir_name
        
        # 检查是否在映射中
        if phone not in mappings:
            orphan_dirs.append(dir_name)
    
    if orphan_dirs:
        print(f"  ⚠️ 找到 {len(orphan_dirs)} 个孤立缓存目录")
        print("\n  孤立的缓存目录（前10个）：")
        for i, dir_name in enumerate(orphan_dirs[:10]):
            print(f"    {i+1}. {dir_name}")
        if len(orphan_dirs) > 10:
            print(f"    ... 还有 {len(orphan_dirs) - 10} 个")
    else:
        print("  ✓ 没有孤立的缓存目录")
    
    # 7. 总结
    print("\n" + "=" * 80)
    print("【诊断总结】")
    print("=" * 80)
    print(f"映射文件中的账号数: {len(mappings)}")
    print(f"缓存目录总数: {len(cache_dirs)}")
    print(f"  - 旧格式: {len(old_format_dirs)}")
    print(f"  - 新格式: {len(new_format_dirs)}")
    print(f"映射账号中找到缓存: {found_count}")
    print(f"映射账号中未找到缓存: {not_found_count}")
    print(f"完整缓存: {complete_count}")
    print(f"不完整缓存: {incomplete_count}")
    print(f"孤立缓存目录: {len(orphan_dirs)}")
    
    # 8. 建议
    print("\n【建议】")
    if not_found_count > 0:
        print(f"⚠️ 有 {not_found_count} 个账号在映射文件中但找不到缓存目录")
        print("   可能原因：缓存目录被删除或移动")
        print("   建议：清理映射文件中的无效条目")
    
    if orphan_dirs:
        print(f"⚠️ 有 {len(orphan_dirs)} 个缓存目录没有对应的映射")
        print("   可能原因：映射文件被清空或重建")
        print("   建议：重新扫描缓存目录并重建映射文件")
    
    if incomplete_count > 0:
        print(f"⚠️ 有 {incomplete_count} 个缓存目录不完整")
        print("   可能原因：缓存保存过程中断或文件损坏")
        print("   建议：删除不完整的缓存目录")
    
    if not_found_count == 0 and len(orphan_dirs) == 0 and incomplete_count == 0:
        print("✓ 缓存系统状态良好，没有发现问题")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
