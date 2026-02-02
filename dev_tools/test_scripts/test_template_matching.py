"""
测试模板匹配功能
Test Template Matching Functionality
"""

import asyncio
import sys
import traceback
from pathlib import Path
from datetime import datetime

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.template_matcher import TemplateMatcher
from src.adb_bridge import ADBBridge
from src.emulator_controller import EmulatorController


def log_to_file(message):
    """记录日志到文件"""
    log_file = Path(__file__).parent / 'template_test.log'
    with open(log_file, 'a', encoding='utf-8') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {message}\n")
    
    # 打印时移除所有特殊 Unicode 字符，避免 GBK 编码错误
    try:
        # 尝试编码为 GBK，如果失败则替换无法编码的字符
        safe_message = message.encode('gbk', errors='replace').decode('gbk')
        print(safe_message)
    except Exception:
        # 如果还是失败，只保留 ASCII 字符
        safe_message = ''.join(c if ord(c) < 128 else '?' for c in message)
        print(safe_message)


async def test_template_matching():
    """测试模板匹配"""
    
    try:
        log_to_file("=" * 60)
        log_to_file("测试模板匹配功能")
        log_to_file("=" * 60)
        
        # 1. 初始化模板匹配器
        template_dir = Path(__file__).parent / 'dist' / 'JT'
        if not template_dir.exists():
            log_to_file(f"✗ 模板目录不存在: {template_dir}")
            return
        
        log_to_file(f"\n1. 加载模板...")
        log_to_file(f"   模板目录: {template_dir}")
        
        matcher = TemplateMatcher(str(template_dir))
        log_to_file(f"   ✓ 已加载 {len(matcher.templates)} 个模板")
        
        # 显示所有模板
        log_to_file(f"\n   模板列表:")
        for name in sorted(matcher.templates.keys()):
            log_to_file(f"     - {name}")
        
        # 2. 连接模拟器
        log_to_file(f"\n2. 连接模拟器...")
        
        # 检测模拟器
        found = EmulatorController.detect_all_emulators()
        if not found:
            log_to_file("✗ 未检测到模拟器")
            return
        
        emulator_type, emulator_path = found[0]
        log_to_file(f"   ✓ 检测到模拟器: {emulator_path}")
        
        controller = EmulatorController(emulator_path)
        
        # 启动模拟器（如果未运行）
        try:
            is_running = await controller._is_running(0)
        except Exception as e:
            log_to_file(f"   检测模拟器状态时出错，假设已运行")
            is_running = True
        if not is_running:
            log_to_file(f"   启动模拟器...")
            success = await controller.launch_instance(0, timeout=120)
            if not success:
                log_to_file("✗ 模拟器启动失败")
                return
            log_to_file(f"   ✓ 模拟器已启动")
        else:
            log_to_file(f"   ✓ 模拟器已在运行")
        
        # 获取 ADB 端口
        adb_port = await controller.get_adb_port(0)
        adb_path = controller.get_adb_path()
        
        if not adb_path:
            log_to_file("✗ 未找到 ADB 路径")
            return
        
        adb = ADBBridge(adb_path)
        device_id = f"127.0.0.1:{adb_port}"
        
        log_to_file(f"   连接 ADB: {device_id}")
        connected = await adb.connect(device_id)
        
        if not connected:
            log_to_file("✗ ADB 连接失败")
            return
        
        log_to_file(f"   ✓ ADB 连接成功")
        
        # 3. 测试模板匹配
        log_to_file(f"\n3. 测试模板匹配...")
        log_to_file(f"   请确保溪盟商城应用已打开")
        log_to_file(f"   按 Enter 开始测试...")
        input()
        
        # 获取当前截图
        log_to_file(f"\n   获取截图...")
        screenshot_data = await adb.screencap(device_id)
        
        if not screenshot_data:
            log_to_file("✗ 获取截图失败")
            return
        
        log_to_file(f"   ✓ 截图大小: {len(screenshot_data)} 字节")
        
        # 匹配所有模板
        log_to_file(f"\n   匹配所有模板...")
        all_matches = matcher.match_all_templates(screenshot_data)
        
        if not all_matches:
            log_to_file("   ✗ 没有找到匹配的模板")
            return
        
        # 显示匹配结果（按相似度排序）
        log_to_file(f"\n   匹配结果（按相似度排序）:")
        log_to_file(f"   {'排名':<6} {'模板名称':<30} {'相似度':<10}")
        log_to_file(f"   {'-'*50}")
        
        for i, match in enumerate(all_matches[:10], 1):  # 只显示前10个
            name = match['template_name']
            similarity = match['similarity']
            color = '✓' if similarity >= 0.85 else '○'
            log_to_file(f"   {color} {i:<4} {name:<30} {similarity:>6.2%}")
        
        # 显示最佳匹配
        best_match = all_matches[0]
        log_to_file(f"\n   最佳匹配:")
        log_to_file(f"     模板: {best_match['template_name']}")
        log_to_file(f"     相似度: {best_match['similarity']:.2%}")
        
        if best_match['similarity'] >= 0.85:
            log_to_file(f"     ✓ 匹配成功（阈值: 85%）")
        else:
            log_to_file(f"     ✗ 相似度低于阈值（85%）")
        
        # 4. 交互式测试
        log_to_file(f"\n4. 交互式测试")
        log_to_file(f"   你可以切换到不同页面，然后按 Enter 测试匹配")
        log_to_file(f"   输入 'q' 退出")
        
        while True:
            user_input = input(f"\n   按 Enter 测试当前页面（或输入 'q' 退出）: ")
            
            if user_input.lower() == 'q':
                break
            
            # 获取截图并匹配
            screenshot_data = await adb.screencap(device_id)
            if not screenshot_data:
                log_to_file("   ✗ 获取截图失败")
                continue
            
            best_match = matcher.match_screenshot(screenshot_data)
            
            if best_match:
                log_to_file(f"   最佳匹配: {best_match['template_name']} (相似度: {best_match['similarity']:.2%})")
                
                if best_match['similarity'] >= 0.85:
                    log_to_file(f"   ✓ 匹配成功")
                else:
                    log_to_file(f"   ○ 相似度较低")
            else:
                log_to_file("   ✗ 没有找到匹配的模板")
        
        log_to_file(f"\n测试完成！")
    
    except Exception as e:
        log_to_file(f"\n✗ 发生错误:")
        log_to_file(f"{str(e)}")
        log_to_file(f"\n详细错误信息:")
        log_to_file(traceback.format_exc())


if __name__ == "__main__":
    try:
        asyncio.run(test_template_matching())
    except Exception as e:
        log_to_file(f"\n✗ 程序异常:")
        log_to_file(f"{str(e)}")
        log_to_file(traceback.format_exc())
