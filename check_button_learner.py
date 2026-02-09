"""
检查按钮位置学习器的样本数据
"""
import json
from pathlib import Path
from collections import defaultdict

def check_button_learner():
    """检查按钮学习器数据"""
    button_dir = Path("runtime_data") / "button_positions"
    
    if not button_dir.exists():
        print("❌ 按钮位置目录不存在")
        return
    
    print("=" * 80)
    print("按钮位置学习器数据统计")
    print("=" * 80)
    
    # 统计所有按钮的样本数
    button_stats = defaultdict(lambda: {"global": 0, "devices": defaultdict(int), "total": 0})
    
    # 检查全局数据文件
    global_file = button_dir / "global.json"
    if global_file.exists():
        try:
            with open(global_file, 'r', encoding='utf-8') as f:
                global_data = json.load(f)
            
            for button_name, button_data in global_data.items():
                sample_count = len(button_data.get("positions", []))
                button_stats[button_name]["global"] = sample_count
                button_stats[button_name]["total"] += sample_count
        except Exception as e:
            print(f"⚠️ 读取全局数据失败: {e}")
    
    # 检查设备专属数据文件
    for device_file in button_dir.glob("device_*.json"):
        device_id = device_file.stem.replace("device_", "")
        
        try:
            with open(device_file, 'r', encoding='utf-8') as f:
                device_data = json.load(f)
            
            for button_name, button_data in device_data.items():
                sample_count = len(button_data.get("positions", []))
                button_stats[button_name]["devices"][device_id] = sample_count
                button_stats[button_name]["total"] += sample_count
        except Exception as e:
            print(f"⚠️ 读取设备数据失败 ({device_file.name}): {e}")
    
    if not button_stats:
        print("✓ 没有找到任何按钮学习数据")
        print("\n这是正常的，学习器会在运行过程中自动记录成功的按钮点击。")
        return
    
    # 输出统计结果
    print(f"\n找到 {len(button_stats)} 个按钮的学习数据:\n")
    
    for button_name, stats in sorted(button_stats.items()):
        total = stats["total"]
        global_count = stats["global"]
        devices = stats["devices"]
        
        # 判断样本是否足够（需要至少5个样本）
        status = "✓" if total >= 5 else "⚠️"
        
        print(f"{status} {button_name}:")
        print(f"    总样本数: {total}")
        print(f"    全局样本: {global_count}")
        
        if devices:
            print(f"    设备专属样本:")
            for device_id, count in sorted(devices.items()):
                print(f"      - {device_id}: {count} 个样本")
        
        if total < 5:
            print(f"    ⚠️ 样本不足（需要至少5个，当前{total}个）")
        
        print()
    
    # 统计总体情况
    total_buttons = len(button_stats)
    sufficient_buttons = sum(1 for stats in button_stats.values() if stats["total"] >= 5)
    insufficient_buttons = total_buttons - sufficient_buttons
    
    print("=" * 80)
    print("总体统计:")
    print("=" * 80)
    print(f"  总按钮数: {total_buttons}")
    print(f"  样本充足: {sufficient_buttons} 个 (≥5个样本)")
    print(f"  样本不足: {insufficient_buttons} 个 (<5个样本)")
    
    if insufficient_buttons > 0:
        print(f"\n⚠️ 有 {insufficient_buttons} 个按钮样本不足，可能影响识别准确性")
    else:
        print(f"\n✓ 所有按钮样本充足")

if __name__ == "__main__":
    check_button_learner()
