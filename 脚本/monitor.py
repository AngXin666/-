"""
统一训练监控脚本
Unified Training Monitor Script

用法:
    python monitor.py --type training
    python monitor.py --type performance
"""
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='统一训练监控脚本')
    parser.add_argument('--type', type=str, required=True,
                        choices=['training', 'performance'],
                        help='监控类型')
    
    args = parser.parse_args()
    
    # 根据类型调用对应的监控函数
    if args.type == 'training':
        from monitor_improved_training import monitor_training
        monitor_training()
    elif args.type == 'performance':
        from monitor_performance import monitor_performance
        monitor_performance()

if __name__ == '__main__':
    main()
