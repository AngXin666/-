"""
清理个人页OCR区域学习数据
只保留签到页的OCR区域学习数据
"""

import json
from pathlib import Path


def clean_ocr_data():
    """清理OCR区域学习数据"""
    
    # 要保留的OCR区域（签到页）
    keep_regions = {
        'checkin_total_times',
        'checkin_remaining_times'
    }
    
    # 要删除的OCR区域（个人页）
    remove_regions = {
        'profile_balance',
        'profile_points',
        'profile_vouchers',
        'profile_coupons'
    }
    
    # 处理全局数据文件
    global_file = Path("runtime_data/ocr_regions/global.json")
    if global_file.exists():
        print(f"处理全局数据文件: {global_file}")
        with open(global_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 统计删除的数据
        removed_count = 0
        for region_name in remove_regions:
            if region_name in data:
                count = data[region_name].get('total_count', 0)
                removed_count += count
                print(f"  删除 {region_name}: {count} 个样本")
                del data[region_name]
        
        # 保存清理后的数据
        with open(global_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 全局数据文件已清理，删除 {removed_count} 个样本")
        print(f"✓ 保留的区域: {', '.join(keep_regions)}")
    else:
        print(f"⚠️ 全局数据文件不存在: {global_file}")
    
    # 处理设备专属数据文件
    ocr_dir = Path("runtime_data/ocr_regions")
    device_files = list(ocr_dir.glob("device_*.json"))
    
    if device_files:
        print(f"\n处理 {len(device_files)} 个设备专属数据文件:")
        for device_file in device_files:
            print(f"  处理: {device_file.name}")
            with open(device_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 删除个人页OCR区域
            removed_count = 0
            for region_name in remove_regions:
                if region_name in data:
                    count = data[region_name].get('total_count', 0)
                    removed_count += count
                    del data[region_name]
            
            # 保存清理后的数据
            with open(device_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"    删除 {removed_count} 个样本")
    else:
        print(f"\n⚠️ 没有找到设备专属数据文件")
    
    print("\n" + "=" * 60)
    print("✓ OCR区域学习数据清理完成")
    print("=" * 60)


if __name__ == "__main__":
    clean_ocr_data()
