"""
重建缓存映射文件
扫描所有缓存目录，重建 phone_userid_mapping.txt
"""

from pathlib import Path

def main():
    print("=" * 80)
    print("重建缓存映射文件")
    print("=" * 80)
    
    cache_dir = Path("login_cache")
    mapping_file = cache_dir / "phone_userid_mapping.txt"
    
    if not cache_dir.exists():
        print("❌ 缓存目录不存在！")
        return
    
    # 1. 备份原映射文件
    if mapping_file.exists():
        backup_file = cache_dir / "phone_userid_mapping.txt.backup"
        import shutil
        shutil.copy2(mapping_file, backup_file)
        print(f"✓ 已备份原映射文件到: {backup_file}")
    
    # 2. 扫描所有缓存目录
    print("\n正在扫描缓存目录...")
    
    mappings = {}
    skipped = []
    
    for item in cache_dir.iterdir():
        if not item.is_dir() or item.name.startswith('.'):
            continue
        
        dir_name = item.name
        
        # 解析目录名：手机号_用户ID
        if '_' in dir_name:
            parts = dir_name.split('_')
            if len(parts) == 2:
                phone, user_id = parts
                mappings[phone] = user_id
            else:
                skipped.append(dir_name)
        else:
            # 旧格式（只有手机号），跳过
            skipped.append(dir_name)
    
    print(f"✓ 扫描完成")
    print(f"  - 找到 {len(mappings)} 个有效映射")
    print(f"  - 跳过 {len(skipped)} 个无效目录")
    
    if skipped:
        print(f"\n跳过的目录（前5个）：")
        for i, dir_name in enumerate(skipped[:5]):
            print(f"  {i+1}. {dir_name}")
        if len(skipped) > 5:
            print(f"  ... 还有 {len(skipped) - 5} 个")
    
    # 3. 写入新的映射文件
    print(f"\n正在写入新的映射文件...")
    
    with open(mapping_file, "w", encoding="utf-8") as f:
        for phone, user_id in sorted(mappings.items()):
            f.write(f"{phone}={user_id}\n")
    
    print(f"✓ 映射文件已更新: {mapping_file}")
    print(f"✓ 共写入 {len(mappings)} 个映射")
    
    # 4. 验证
    print(f"\n正在验证...")
    
    with open(mapping_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    print(f"✓ 验证成功，文件包含 {len(lines)} 行")
    
    # 5. 显示前10个映射
    print(f"\n新映射文件内容（前10个）：")
    for i, (phone, user_id) in enumerate(list(sorted(mappings.items()))[:10]):
        print(f"  {i+1}. {phone} -> {user_id}")
    
    if len(mappings) > 10:
        print(f"  ... 还有 {len(mappings) - 10} 个")
    
    print("\n" + "=" * 80)
    print("✓ 重建完成！")
    print("=" * 80)
    print(f"原映射文件已备份到: {cache_dir / 'phone_userid_mapping.txt.backup'}")
    print(f"新映射文件: {mapping_file}")
    print(f"映射数量: {len(mappings)}")
    print("\n现在可以重新运行程序，缓存查找应该正常工作了。")
    print("=" * 80)

if __name__ == "__main__":
    main()
