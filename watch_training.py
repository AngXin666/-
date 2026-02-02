"""
实时监控 YOLO 训练进度
"""
import time
from pathlib import Path
import pandas as pd

def watch_training():
    """实时监控训练进度"""
    results_file = Path("yolo_runs/button_detector/results.csv")
    
    print("=" * 80)
    print("YOLO 训练实时监控")
    print("=" * 80)
    
    if not results_file.exists():
        print("\n等待训练开始...")
        while not results_file.exists():
            time.sleep(2)
    
    last_epoch = 0
    
    while True:
        try:
            # 读取结果
            df = pd.read_csv(results_file)
            
            if len(df) > last_epoch:
                # 显示新的训练结果
                new_rows = df.iloc[last_epoch:]
                
                for _, row in new_rows.iterrows():
                    epoch = int(row['epoch'])
                    train_loss = row['train/box_loss'] + row['train/cls_loss'] + row['train/dfl_loss']
                    val_loss = row['val/box_loss'] + row['val/cls_loss'] + row['val/dfl_loss']
                    
                    print(f"\nEpoch {epoch:3d}/100:")
                    print(f"  训练损失: {train_loss:.4f}")
                    print(f"  验证损失: {val_loss:.4f}")
                    print(f"  用时: {row['time']:.2f}s")
                    
                    # 如果有 mAP 数据
                    if row['metrics/mAP50(B)'] > 0:
                        print(f"  mAP50: {row['metrics/mAP50(B)']:.4f}")
                        print(f"  mAP50-95: {row['metrics/mAP50-95(B)']:.4f}")
                
                last_epoch = len(df)
                
                # 显示进度条
                progress = (last_epoch / 100) * 100
                bar_length = 50
                filled = int(bar_length * last_epoch / 100)
                bar = '█' * filled + '░' * (bar_length - filled)
                print(f"\n进度: [{bar}] {progress:.1f}%")
            
            time.sleep(5)  # 每 5 秒检查一次
            
        except KeyboardInterrupt:
            print("\n\n监控已停止")
            break
        except Exception as e:
            print(f"\n读取错误: {e}")
            time.sleep(5)

if __name__ == "__main__":
    watch_training()
