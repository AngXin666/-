"""
检查登录缓存状态
"""

import os
from pathlib import Path
from datetime import datetime
import time

def check_login_cache():
    """检查登录缓存目录"""
    cache_dir = Path("login_cache")
    
    if not cache_dir.exists():
        print("❌ login_cache 目录不存在")
        return
    
    print("="*80)
    print("登录缓存检查")
    print("="*80)
    
    # 获取所有缓存目录
    cache_folders = [d for d in cache_dir.iterdir() if d.is_dir()]
    
    print(f"\n总缓存目录数: {len(cache_folders)}")
    
    # 按修改时间排序
    cache_folders_with_time = []
    for folder in cache_folders:
        try:
            mtime = folder.stat().st_mtime
            cache_folders_with_time.append((folder, mtime))
        except Exception as e:
            print(f"⚠️ 无法获取 {folder.name} 的修改时间: {e}")
    
    # 排序（最新的在前）
    cache_folders_with_time.sort(key=lambda x: x[1], reverse=True)
    
    # 显示最近10个修改的缓存
    print("\n" + "="*80)
    print("最近修改的10个缓存目录:")
    print("="*80)
    
    current_time = time.time()
    
    for i, (folder, mtime) in enumerate(cache_folders_with_time[:10], 1):
        # 计算时间差
        time_diff = current_time - mtime
        
        # 格式化时间差
        if time_diff < 60:
            time_str = f"{int(time_diff)}秒前"
        elif time_diff < 3600:
            time_str = f"{int(time_diff/60)}分钟前"
        elif time_diff < 86400:
            time_str = f"{int(time_diff/3600)}小时前"
        else:
            time_str = f"{int(time_diff/86400)}天前"
        
        # 格式化修改时间
        mod_time = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        
        # 检查目录内容
        files = list(folder.glob("*"))
        file_count = len(files)
        
        # 提取手机号和用户ID
        parts = folder.name.split("_")
        phone = parts[0] if len(parts) > 0 else "未知"
        user_id = parts[1] if len(parts) > 1 else "未知"
        
        print(f"\n{i}. {folder.name}")
        print(f"   手机号: {phone}")
        print(f"   用户ID: {user_id}")
        print(f"   修改时间: {mod_time} ({time_str})")
        print(f"   文件数: {file_count}")
        
        # 显示文件列表
        if file_count > 0:
            print(f"   文件:")
            for f in files[:5]:  # 只显示前5个文件
                file_size = f.stat().st_size if f.is_file() else 0
                size_str = f"{file_size:,} bytes" if file_size > 0 else "0 bytes"
                print(f"     - {f.name} ({size_str})")
            if file_count > 5:
                print(f"     ... 还有 {file_count - 5} 个文件")
    
    # 检查最近1小时内修改的缓存
    print("\n" + "="*80)
    print("最近1小时内修改的缓存:")
    print("="*80)
    
    recent_caches = []
    one_hour_ago = current_time - 3600
    
    for folder, mtime in cache_folders_with_time:
        if mtime > one_hour_ago:
            recent_caches.append((folder, mtime))
    
    if recent_caches:
        print(f"\n找到 {len(recent_caches)} 个最近1小时内修改的缓存:")
        for folder, mtime in recent_caches:
            mod_time = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            parts = folder.name.split("_")
            phone = parts[0] if len(parts) > 0 else "未知"
            user_id = parts[1] if len(parts) > 1 else "未知"
            
            files = list(folder.glob("*"))
            file_count = len(files)
            
            print(f"  - {phone} (ID: {user_id}) - {mod_time} - {file_count}个文件")
    else:
        print("\n❌ 没有找到最近1小时内修改的缓存")
        print("   这可能意味着缓存没有被保存")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    check_login_cache()
