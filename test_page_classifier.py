"""
测试页面分类器识别首页和首页广告弹窗的准确性
"""
import asyncio
import sys
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.adb_bridge import ADBBridge
from src.page_detector_integrated import PageDetectorIntegrated


async def main():
    """主函数"""
    print("=" * 60)
    print("页面分类器识别测试")
    print("=" * 60)
    
    # 初始化ADB
    print("\n[1] 初始化ADB...")
    
    # 从配置加载模拟器路径
    import yaml
    import os
    config_path = 'config.yaml'
    nox_path = None
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            nox_path = config_data.get('nox_path', '')
    
    # 初始化模拟器控制器以获取ADB路径
    adb_path = None
    if nox_path:
        from src.emulator_controller import EmulatorController
        emulator_controller = EmulatorController(nox_path)
        adb_path = emulator_controller.get_adb_path()
        if adb_path:
            print(f"✓ 找到ADB路径: {adb_path}")
        else:
            print("⚠️ 未找到ADB路径")
    else:
        print("⚠️ 未配置模拟器路径")
    
    adb = ADBBridge(adb_path)
    
    # 使用固定设备ID（MuMu模拟器）
    device_id = "127.0.0.1:7555"
    print(f"✓ 使用设备: {device_id}")
    
    # 初始化页面检测器
    print("\n[2] 初始化页面检测器...")
    detector = PageDetectorIntegrated(adb)
    
    # 连续检测同一页面10次，测试识别稳定性
    print("\n[3] 连续检测同一页面10次，测试识别稳定性...")
    print("⚠️ 请保持模拟器页面不变，不要切换页面！")
    print()
    
    results = []
    for i in range(10):
        print(f"第 {i+1}/10 次检测...", end=" ")
        
        # 检测当前页面（不使用缓存）
        result = await detector.detect_page(device_id, use_cache=False, detect_elements=False)
        
        results.append({
            'state': result.state.value,
            'confidence': result.confidence,
            'details': result.details
        })
        
        print(f"{result.state.value} (置信度: {result.confidence:.2%})")
        
        # 短暂等待
        await asyncio.sleep(0.5)
    
    # 统计结果
    print("\n" + "=" * 60)
    print("识别结果统计：")
    print("=" * 60)
    
    from collections import Counter
    state_counts = Counter([r['state'] for r in results])
    
    print(f"\n识别到的页面类型：")
    for state, count in state_counts.most_common():
        percentage = count / len(results) * 100
        print(f"  {state}: {count}次 ({percentage:.1f}%)")
    
    print(f"\n置信度范围：")
    confidences = [r['confidence'] for r in results]
    print(f"  最低: {min(confidences):.2%}")
    print(f"  最高: {max(confidences):.2%}")
    print(f"  平均: {sum(confidences)/len(confidences):.2%}")
    
    print(f"\n识别稳定性：")
    if len(state_counts) == 1:
        print(f"  ✓ 非常稳定 - 10次识别结果完全一致")
    elif len(state_counts) == 2:
        print(f"  ⚠️ 不稳定 - 识别出2种不同结果")
    else:
        print(f"  ✗ 非常不稳定 - 识别出{len(state_counts)}种不同结果")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
