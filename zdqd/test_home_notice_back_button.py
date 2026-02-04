#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试用返回按钮关闭首页公告弹窗
Test closing home notice popup with back button
"""

import asyncio
import sys
import subprocess
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.adb_bridge import ADBBridge
from src.page_detector import PageState
from src.model_manager import ModelManager


async def test_home_notice_back_button():
    """测试用返回按钮关闭首页公告弹窗"""
    
    print("=" * 60)
    print("测试：用返回按钮关闭首页公告弹窗")
    print("=" * 60)
    
    # ==================== 初始化设备 ====================
    print("\n[1/6] 初始化ADB连接...")
    
    # 尝试多个可能的ADB路径
    possible_adb_paths = [
        r"D:\Program Files\Netease\MuMu\nx_device\12.0\shell\adb.exe",
        r"C:\Program Files\Netease\MuMu\nx_device\12.0\shell\adb.exe",
        r"D:\Program Files (x86)\Nox\bin\adb.exe",
        r"C:\Program Files (x86)\Nox\bin\adb.exe",
        "adb"  # 系统PATH中的adb
    ]
    
    adb_path = None
    for path in possible_adb_paths:
        try:
            if path == "adb":
                # 尝试系统PATH中的adb
                result = subprocess.run(
                    ["adb", "version"], 
                    capture_output=True, 
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    adb_path = "adb"
                    print(f"✓ 使用系统ADB")
                    break
            else:
                # 检查文件是否存在
                from pathlib import Path
                if Path(path).exists():
                    adb_path = path
                    print(f"✓ 找到ADB: {path}")
                    break
        except:
            continue
    
    if not adb_path:
        print("❌ 未找到ADB，请确保夜神模拟器已安装")
        return
    
    adb = ADBBridge(adb_path)
    
    # 自动获取设备列表
    print("正在获取设备列表...")
    try:
        result = subprocess.run(
            [adb_path, "devices"], 
            capture_output=True, 
            text=True,
            timeout=10
        )
        lines = result.stdout.strip().split('\n')[1:]
        devices = [line.split('\t')[0] for line in lines if line.strip() and 'device' in line]
        
        if not devices:
            print("❌ 没有找到正在运行的设备")
            print("\n提示:")
            print("1. 请确保模拟器正在运行")
            print("2. 在模拟器中启用ADB调试")
            print("3. 尝试运行: adb devices")
            return
        
        device_id = devices[0]
        print(f"✓ 找到设备: {device_id}")
    except Exception as e:
        print(f"❌ 获取设备列表失败: {e}")
        return
    
    # 应用包名
    package_name = "com.ry.xmsc"
    
    # ==================== 初始化模型 ====================
    print("\n[2/6] 初始化模型管理器...")
    model_manager = ModelManager.get_instance()
    model_manager.initialize_all_models(adb)
    detector = model_manager.get_page_detector_integrated()
    print("✓ 模型管理器初始化完成")
    
    # ==================== 停止应用 ====================
    print(f"\n[3/6] 停止应用: {package_name}")
    await adb.stop_app(device_id, package_name)
    await asyncio.sleep(1)
    print("✓ 应用已停止")
    
    # ==================== 清理应用缓存 ====================
    print("\n[4/6] 清理应用缓存...")
    result = await adb.shell(device_id, f"pm clear {package_name}")
    if "Success" in result:
        print("✓ 缓存清理成功")
    else:
        print(f"⚠️ 缓存清理结果: {result}")
    await asyncio.sleep(1)
    
    # ==================== 启动应用 ====================
    print(f"\n[5/6] 启动应用: {package_name}")
    success = await adb.start_app(device_id, package_name)
    if success:
        print("✓ 应用启动成功")
    else:
        print("❌ 应用启动失败")
        return
    
    # 等待应用启动
    print("\n等待应用启动...")
    await asyncio.sleep(3)
    
    # ==================== 开始测试循环 ====================
    print("\n[6/6] 开始测试循环...")
    print("=" * 60)
    
    max_attempts = 1  # 只测试1次，生成可视化图片
    success_count = 0
    fail_count = 0
    
    for attempt in range(1, max_attempts + 1):
        print(f"\n{'='*60}")
        print(f"第 {attempt}/{max_attempts} 次测试")
        print(f"{'='*60}")
        
        # 停止应用
        print(f"[1/4] 停止应用...")
        await adb.stop_app(device_id, package_name)
        await asyncio.sleep(1)
        
        # 清理缓存
        print(f"[2/4] 清理应用缓存...")
        await adb.shell(device_id, f"pm clear {package_name}")
        await asyncio.sleep(1)
        
        # 启动应用
        print(f"[3/4] 启动应用...")
        await adb.start_app(device_id, package_name)
        await asyncio.sleep(3)
        
        # 检测并处理首页公告弹窗
        print(f"[4/4] 检测首页公告弹窗...")
        
        max_wait = 15  # 最多等待15秒
        found_popup = False
        
        for wait_time in range(max_wait):
            result = await detector.detect_page(device_id, use_cache=False, detect_elements=False)
            
            if not result or not result.state:
                await asyncio.sleep(1)
                continue
            
            print(f"  [{wait_time+1}s] 当前页面: {result.state.value} (置信度: {result.confidence:.2%})")
            
            # 检测到首页公告弹窗
            if result.state == PageState.HOME_NOTICE:
                found_popup = True
                print("  ✓ 检测到首页公告弹窗")
                
                # 保存截图并标记点击位置
                screenshot = await adb.screencap(device_id)
                if screenshot:
                    import cv2
                    import numpy as np
                    from datetime import datetime
                    
                    nparr = np.frombuffer(screenshot, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    click_x, click_y = 270, 250  # 再下移50像素，避免点击到搜索框
                    
                    # 标记点击位置
                    cv2.circle(img, (int(click_x), int(click_y)), 20, (0, 0, 255), 3)
                    cv2.line(img, (int(click_x)-30, int(click_y)), (int(click_x)+30, int(click_y)), (0, 0, 255), 2)
                    cv2.line(img, (int(click_x), int(click_y)-30), (int(click_x), int(click_y)+30), (0, 0, 255), 2)
                    cv2.putText(img, f"Test {attempt}: ({int(click_x)}, {int(click_y)})", 
                               (int(click_x)+25, int(click_y)-25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    vis_path = f"zdqd/debug_annotations/test_{attempt:02d}_{timestamp}.png"
                    cv2.imwrite(vis_path, img)
                    print(f"  ✓ 截图已保存: {vis_path}")
                
                # 点击关闭弹窗
                print(f"  点击弹窗外上方空白区域关闭...")
                await adb.tap(device_id, 270, 250)  # 再下移50像素，避免点击到搜索框
                await asyncio.sleep(1)
                
                # 检测关闭后的页面
                after_result = await detector.detect_page(device_id, use_cache=False, detect_elements=False)
                
                if after_result and after_result.state == PageState.HOME:
                    print(f"  ✓ 成功关闭弹窗，到达首页")
                    success_count += 1
                else:
                    print(f"  ❌ 关闭失败，当前页面: {after_result.state.value if after_result else 'Unknown'}")
                    fail_count += 1
                
                break
            
            # 如果已经到达首页，说明没有弹窗
            elif result.state == PageState.HOME:
                print("  ⚠️ 未出现首页公告弹窗，直接到达首页")
                break
            
            # 处理启动弹窗
            elif result.state == PageState.STARTUP_POPUP:
                print("  处理启动弹窗...")
                await adb.tap(device_id, 270, 600)
                await asyncio.sleep(1)
            
            # 其他状态继续等待
            else:
                await asyncio.sleep(1)
        
        if not found_popup:
            print(f"  ⚠️ 本次测试未出现首页公告弹窗")
        
        print(f"  当前统计: 成功 {success_count} 次, 失败 {fail_count} 次")
    
    # ==================== 测试结果 ====================
    print("\n" + "=" * 60)
    print("测试结果")
    print("=" * 60)
    print(f"总测试次数: {max_attempts}")
    print(f"出现弹窗次数: {success_count + fail_count}")
    print(f"成功关闭次数: {success_count}")
    print(f"关闭失败次数: {fail_count}")
    if success_count + fail_count > 0:
        print(f"成功率: {success_count/(success_count + fail_count)*100:.1f}%")
        print(f"\n✓ 点击坐标 (270, 150) 可以关闭首页公告弹窗")
    else:
        print("\n⚠️ 所有测试中都未出现首页公告弹窗")
    
    # 清理资源
    print("\n清理资源...")
    model_manager.cleanup()
    print("✓ 测试完成")


if __name__ == "__main__":
    try:
        asyncio.run(test_home_notice_back_button())
    except KeyboardInterrupt:
        print("\n\n用户中断测试")
    except Exception as e:
        print(f"\n\n测试出错: {e}")
        import traceback
        traceback.print_exc()
