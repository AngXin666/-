"""
监控阶段1训练进度
每 10 轮汇报一次质量
"""
import time
from pathlib import Path


def parse_results(results_file):
    """解析训练结果"""
    if not results_file.exists():
        return None
    
    with open(results_file, 'r') as f:
        lines = f.readlines()
    
    if len(lines) < 2:
        return None
    
    last_line = lines[-1].strip()
    parts = last_line.split(',')
    
    if len(parts) < 10:
        return None
    
    try:
        epoch = int(parts[0].strip())
        map50 = float(parts[7].strip())
        map50_95 = float(parts[8].strip())
        return {
            'epoch': epoch,
            'map50': map50,
            'map50_95': map50_95
        }
    except:
        return None


def monitor_training():
    """监控训练"""
    print("=" * 60)
    print("监控阶段1训练进度")
    print("=" * 60)
    print("每 10 轮汇报一次质量\n")
    
    results_file = Path("yolo_runs/stage1_buttons/results.csv")
    last_reported_epoch = 0
    
    while True:
        if results_file.exists():
            result = parse_results(results_file)
            
            if result:
                current_epoch = result['epoch']
                
                # 每 10 轮汇报一次
                if current_epoch >= last_reported_epoch + 10:
                    print(f"\n轮次 {current_epoch}:")
                    print(f"  mAP50: {result['map50']:.2%}")
                    print(f"  mAP50-95: {result['map50_95']:.2%}")
                    last_reported_epoch = current_epoch
        
        time.sleep(30)


if __name__ == "__main__":
    try:
        monitor_training()
    except KeyboardInterrupt:
        print("\n\n监控已停止")
