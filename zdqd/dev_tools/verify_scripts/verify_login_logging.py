"""
验证登录流程日志输出

演示登录流程的简洁日志输出效果
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.concise_logger import ConciseLogger, LogFormatter


class MockGUILogger:
    """模拟 GUI 日志记录器"""
    
    def info(self, message):
        print(message)
    
    def error(self, message):
        print(message)


def demo_login_flow_logging():
    """演示登录流程的日志输出"""
    
    # 创建模拟的 GUI 日志记录器
    gui_logger = MockGUILogger()
    
    # 创建简洁日志记录器
    concise_logger = ConciseLogger(
        module_name="auto_login",
        gui_logger=gui_logger,
        file_logger=None  # 不使用文件日志
    )
    
    print("=" * 60)
    print("登录流程日志输出演示")
    print("=" * 60)
    print()
    
    # 模拟登录流程
    concise_logger.step(2, "登录账号")
    concise_logger.action("导航到登录页")
    concise_logger.action("输入账号信息")
    concise_logger.action("点击登录")
    concise_logger.success("登录成功")
    
    print()
    print("=" * 60)
    print("日志格式验证")
    print("=" * 60)
    print()
    
    # 验证各种日志格式
    print("步骤日志格式:")
    print(f"  {LogFormatter.format_step(2, '登录账号')}")
    print()
    
    print("操作日志格式:")
    print(f"  {LogFormatter.format_action('导航到登录页')}")
    print(f"  {LogFormatter.format_action('输入账号信息')}")
    print(f"  {LogFormatter.format_action('点击登录')}")
    print()
    
    print("成功日志格式:")
    print(f"  {LogFormatter.format_success('登录成功')}")
    print()
    
    print("错误日志格式:")
    print(f"  {LogFormatter.format_error('无法导航到登录页面')}")
    print()
    
    print("=" * 60)
    print("验证完成！")
    print("=" * 60)


if __name__ == "__main__":
    demo_login_flow_logging()
