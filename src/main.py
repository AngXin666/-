"""
自动签到助手 - 主程序入口
Emulator Automation Script - Main Entry Point
"""

import asyncio
import sys
from pathlib import Path

from .config import ConfigLoader
from .emulator_controller import EmulatorController
from .adb_bridge import ADBBridge
from .account_manager import AccountManager
from .orchestrator import Orchestrator
from .logger import get_logger


def find_target_app(adb_bridge: ADBBridge, device_id: str, 
                    keyword: str = "溪盟") -> str:
    """查找目标应用包名
    
    Args:
        adb_bridge: ADB 桥接器
        device_id: 设备 ID
        keyword: 应用关键字
        
    Returns:
        包名
    """
    import asyncio
    loop = asyncio.get_event_loop()
    package = loop.run_until_complete(
        adb_bridge.find_package_by_name(device_id, keyword)
    )
    return package or ""


async def main_async():
    """异步主函数"""
    # 初始化日志
    logger = get_logger()
    logger.info("自动签到助手启动")
    
    # 加载配置
    config_loader = ConfigLoader()
    config = config_loader.load()
    
    # 自动检测模拟器路径
    if not config.nox_path:
        emulator_path = EmulatorController.auto_detect_path()
        if emulator_path:
            config.nox_path = emulator_path
            logger.info(f"自动检测到模拟器路径: {emulator_path}")
        else:
            logger.error("未能检测到模拟器安装路径，请在 config.yaml 中手动配置")
            return
    
    # 构建 ADB 路径
    if not config.adb_path and config.nox_path:
        # 尝试在模拟器路径中查找 adb.exe
        adb_path = Path(config.nox_path) / "adb.exe"
        if adb_path.exists():
            config.adb_path = str(adb_path)
    
    # 初始化账号管理器
    account_manager = AccountManager(config.accounts_file)
    accounts = account_manager.load_accounts()
    
    if not accounts:
        logger.error(f"未能加载账号，请检查账号文件: {config.accounts_file}")
        return
    
    logger.info(f"已加载 {len(accounts)} 个账号")
    
    # 初始化编排器
    orchestrator = Orchestrator(config, account_manager)
    
    try:
        # 如果未配置目标应用包名，尝试自动搜索
        if not config.target_app_package:
            logger.info("正在搜索溪盟商城应用...")
            # 需要先启动一个实例来搜索
            emulator_controller = EmulatorController(config.nox_path)
            if await emulator_controller.launch_instance(0, timeout=120):
                adb_bridge = ADBBridge(config.adb_path)
                adb_port = await emulator_controller.get_adb_port(0)
                if adb_port:
                    device_id = f"127.0.0.1:{adb_port}"
                    await adb_bridge.connect(device_id)
                    
                    keyword = config.target_app_keyword
                    package = await adb_bridge.find_package_by_name(device_id, keyword)
                    
                    if package:
                        config.target_app_package = package
                        logger.info(f"找到目标应用: {package}")
                    else:
                        logger.error(f"未找到包含 '{keyword}' 的应用")
                        await emulator_controller.quit_instance(0)
                        return
                
                await emulator_controller.quit_instance(0)
        
        # 执行所有账号任务
        logger.info("开始处理所有账号...")
        await orchestrator.run_all_accounts()
        
        # 输出统计信息
        stats = account_manager.get_statistics()
        logger.info(f"处理完成 - 总计:{stats['total']}, 成功:{stats['completed']}, "
                   f"失败:{stats['failed']}, 成功率:{stats['success_rate']:.1f}%")
        logger.info(f"总抽奖金额: {stats['total_draw_amount']}")
        
    except KeyboardInterrupt:
        logger.warning("用户中断操作")
    except Exception as e:
        logger.error(f"运行出错: {e}")
    finally:
        await orchestrator.cleanup()
        logger.info("程序结束")


def main():
    """主函数入口"""
    asyncio.run(main_async())


# 注意：不要在这里调用 main()，因为这个模块可能被其他模块导入
# 如果需要单独运行，请使用 run.py
