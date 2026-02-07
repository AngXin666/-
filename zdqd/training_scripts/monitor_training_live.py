"""
实时监控 YOLO 训练进度（200轮）
"""
import time
from pathlib import Path
import pandas as pd
import os

def monitor_training_live():
    """实时监控训练进度"""
    results_file = Path("yolo_runs/button_detector/results.csv")
    
    print("=" * 80)
    print("YOLO 训练实时监控（200 轮 / 640x640）")
    print("=" * 80)
    print("\n数据集: 3476 张图片 (训练集: 2780, 验证集: 696)")
    print("配置: batch=16, workers=8, cache=disk, amp=True")
    print("\n等待训练开始...")
    
    # 等待结果文件生成
    while not results_file.exists():
        time.sleep(2)
        print(".", end="", flush=True)
    
    print("\n\n训练已开始！")
    print("-" * 80)
    
    last_epoch = 0
    best_map50 = 0
    best_epoch = 0
    
    try:
        while True:
            if not results_file.exists():
                time.sleep(2)
                continue
            
            try:
                # 读取结果
                df = pd.read_csv(results_file)
                
                if len(df) > last_epoch:
                    # 清屏（可选）
                    # os.system('cls' if os.name == 'nt' else 'clear')
                    
                    # 显示新的训练结果
                    new_rows = df.iloc[last_epoch:]
                    
                    for _, row in new_rows.iterrows():
                        epoch = int(row['epoch'])
                        
                        # 计算总损失
                        train_loss = row['train/box_loss'] + row['train/cls_loss'] + row['train/dfl_loss']
                        val_loss = row['val/box_loss'] + row['val/cls_loss'] + row['val/dfl_loss']
                        
                        # 获取指标
                        map50 = row['metrics/mAP50(B)']
                        map50_95 = row['metrics/mAP50-95(B)']
                        precision = row['metrics/precision(B)']
                        recall = row['metrics/recall(B)']
                        
                        # 更新最佳 mAP
                        if map50 > best_map50:
                            best_map50 = map50
                            best_epoch = epoch
                        
                        # 显示进度
                        print(f"\n[Epoch {epoch:3d}/200] ", end="")
                        
                        # 进度条
                        progress = epoch / 200
                        bar_length = 30
                        filled = int(bar_length * progress)
                        bar = '█' * filled + '░' * (bar_length - filled)
                        print(f"[{bar}] {progress*100:.1f}%")
                        
                        # 显示指标
                        print(f"  Loss: 训练={train_loss:.3f}, 验证={val_loss:.3f}")
                        
                        if map50 > 0:
                            print(f"  mAP50: {map50:.4f} ({map50*100:.2f}%)", end="")
                            if epoch == best_epoch:
                                print(" 🏆 最佳", end="")
                            print()
                            print(f"  mAP50-95: {map50_95:.4f} ({map50_95*100:.2f}%)")
                            print(f"  精确率: {precision:.4f}, 召回率: {recall:.4f}")
                        
                        # 显示最佳记录
                        if best_map50 > 0:
                            print(f"  最佳: mAP50={best_map50:.4f} (Epoch {best_epoch})")
                        
                        # 预估剩余时间
                        if epoch > 0:
                            avg_time = row['time'] / epoch
                            remaining = (200 - epoch) * avg_time
                            mins = int(remaining / 60)
                            secs = int(remaining % 60)
                            print(f"  预计剩余: {mins}分{secs}秒")
                    
                    last_epoch = len(df)
                    
                    # 检查是否完成
                    if last_epoch >= 200:
                        print("\n" + "=" * 80)
                        print("训练完成！")
                        print("=" * 80)
                        print(f"\n最佳模型: Epoch {best_epoch}, mAP50={best_map50:.4f} ({best_map50*100:.2f}%)")
                        print(f"模型位置: yolo_runs/button_detector/weights/best.pt")
                        break
                
                time.sleep(3)  # 每 3 秒检查一次
                
            except Exception as e:
                print(f"\n读取错误: {e}")
                time.sleep(3)
                
    except KeyboardInterrupt:
        print("\n\n监控已停止")
        print(f"当前进度: {last_epoch}/200 轮")
        if best_map50 > 0:
            print(f"当前最佳: mAP50={best_map50:.4f} (Epoch {best_epoch})")

if __name__ == "__main__":
    monitor_training_live()
