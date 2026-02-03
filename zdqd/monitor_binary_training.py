"""
监控二分类模型训练进度
"""
import time
import json
from pathlib import Path

def monitor_training():
    """监控训练进度"""
    output_dir = Path("binary_models")
    
    print("=" * 60)
    print("监控二分类模型训练进度")
    print("=" * 60)
    
    while True:
        # 检查已完成的模型
        completed_models = list(output_dir.glob("*_info.json"))
        
        print(f"\n[{time.strftime('%H:%M:%S')}] 已完成模型: {len(completed_models)}/16")
        
        if completed_models:
            print("\n已完成的模型:")
            for info_file in sorted(completed_models):
                with open(info_file, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                print(f"  ✓ {info['class_name']:<20} "
                      f"验证准确率: {info['val_accuracy']:.2%} "
                      f"精确率: {info['val_precision']:.2%} "
                      f"召回率: {info['val_recall']:.2%}")
        
        # 检查是否全部完成
        if len(completed_models) >= 16:
            print("\n" + "=" * 60)
            print("所有模型训练完成！")
            print("=" * 60)
            break
        
        # 等待15秒
        time.sleep(15)

if __name__ == '__main__':
    monitor_training()
