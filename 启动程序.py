"""
自动签到助手 - 启动脚本
Automation Script - Launcher

使用方法：
    python 启动程序.py
    
或者双击运行此文件
"""

if __name__ == "__main__":
    import sys
    
    # 添加当前目录到Python路径
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # 导入并运行GUI
    from src.gui import main
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序已退出")
        sys.exit(0)
    except Exception as e:
        print(f"\n程序运行出错: {e}")
        import traceback
        traceback.print_exc()
        input("\n按回车键退出...")
        sys.exit(1)
