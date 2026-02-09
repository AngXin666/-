"""检查学习器数据收集情况"""
import json
from pathlib import Path

def check_button_positions():
    """检查按钮位置学习数据"""
    print("=" * 60)
    print("按钮位置学习数据统计")
    print("=" * 60)
    
    # 检查全局数据
    global_file = Path("runtime_data/button_positions/global.json")
    if global_file.exists():
        with open(global_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("\n【全局数据】")
        for button_name, button_data in data.items():
            total = button_data.get('total_count', 0)
            samples = len(button_data.get('positions', []))
            last_updated = button_data.get('last_updated', 'N/A')
            print(f"  {button_name}:")
            print(f"    - 总计数: {total}")
            print(f"    - 当前样本: {samples}")
            print(f"    - 最后更新: {last_updated}")
    
    # 检查设备专属数据
    device_files = list(Path("runtime_data/button_positions").glob("device_*.json"))
    if device_files:
        print("\n【设备专属数据】")
        for device_file in device_files:
            device_id = device_file.stem.replace("device_", "")
            with open(device_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"\n  设备 {device_id}:")
            for button_name, button_data in data.items():
                total = button_data.get('total_count', 0)
                samples = len(button_data.get('positions', []))
                print(f"    {button_name}: {total}次 (当前{samples}个样本)")

def check_ocr_regions():
    """检查OCR区域学习数据"""
    print("\n" + "=" * 60)
    print("OCR区域学习数据统计")
    print("=" * 60)
    
    # 检查全局数据
    global_file = Path("runtime_data/ocr_regions/global.json")
    if global_file.exists():
        with open(global_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("\n【全局数据】")
        for region_name, region_data in data.items():
            total = region_data.get('total_count', 0)
            samples = len(region_data.get('regions', []))
            last_updated = region_data.get('last_updated', 'N/A')
            print(f"  {region_name}:")
            print(f"    - 总计数: {total}")
            print(f"    - 当前样本: {samples}")
            print(f"    - 最后更新: {last_updated}")
    
    # 检查设备专属数据
    device_files = list(Path("runtime_data/ocr_regions").glob("device_*.json"))
    if device_files:
        print("\n【设备专属数据】")
        for device_file in device_files:
            device_id = device_file.stem.replace("device_", "")
            with open(device_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"\n  设备 {device_id}:")
            for region_name, region_data in data.items():
                total = region_data.get('total_count', 0)
                samples = len(region_data.get('regions', []))
                print(f"    {region_name}: {total}次 (当前{samples}个样本)")

def main():
    print("\n学习器数据收集情况报告")
    print("生成时间:", Path("runtime_data/button_positions/global.json").stat().st_mtime if Path("runtime_data/button_positions/global.json").exists() else "N/A")
    
    check_button_positions()
    check_ocr_regions()
    
    print("\n" + "=" * 60)
    print("统计完成")
    print("=" * 60)

if __name__ == "__main__":
    main()
