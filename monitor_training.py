"""
监控 YOLO 训练进度
"""
import time
from pathlib import Path

def monitor_training():
    """监控训练日志"""
    log_dir = Path("yolo_runs/button_detector")
    
    print("=" * 60)
    print("YOLO 训练监控")
    print("=" * 60)
    print("\n等待训练开始...")
    
    # 等待训练目录创建
    while not log_dir.exists():
        time.sleep(1)
    
    print(f"\n训练目录已创建: {log_dir}")
    
    # 查找结果文件
    results_file = log_dir / "results.csv"
    
    if results_file.exists():
        print(f"\n找到结果文件: {results_file}")
        print("\n训练进度:")
        print("-" * 60)
        
        # 读取并显示最新结果
        with open(results_file, 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:
                # 显示表头
                print(lines[0].strip())
                # 显示最后几行
                for line in lines[-5:]:
                    print(line.strip())
    else:
        print("\n等待结果文件生成...")
    
    # 检查权重文件
    weights_dir = log_dir / "weights"
    if weights_dir.exists():
        print(f"\n权重目录: {weights_dir}")
        best_pt = weights_dir / "best.pt"
        last_pt = weights_dir / "last.pt"
        
        if best_pt.exists():
            size_mb = best_pt.stat().st_size / (1024 * 1024)
            print(f"  ✓ best.pt ({size_mb:.1f} MB)")
        
        if last_pt.exists():
            size_mb = last_pt.stat().st_size / (1024 * 1024)
            print(f"  ✓ last.pt ({size_mb:.1f} MB)")
    
    print("\n" + "=" * 60)
    print("提示: 训练正在后台进行，可以随时运行此脚本查看进度")
    print("=" * 60)

if __name__ == "__main__":
    monitor_training()
