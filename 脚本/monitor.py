"""
统一训练监控脚本 - 简化版
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

def main():
    print("=" * 60)
    print("训练监控脚本")
    print("=" * 60)
    print("\n请选择监控类型:")
    print("  1. 训练进度监控")
    print("  2. 性能监控")
    print("  0. 退出")
    
    choice = input("\n请输入选项 (0-2): ").strip()
    
    if choice == '1':
        from monitor_improved_training import monitor_training
        monitor_training()
    elif choice == '2':
        from monitor_performance import monitor_performance
        monitor_performance()
    elif choice == '0':
        print("退出")
    else:
        print("无效选项")

if __name__ == '__main__':
    main()
