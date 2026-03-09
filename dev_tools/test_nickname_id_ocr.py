"""
测试OCR识别当前页面的昵称和ID
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import subprocess
from io import BytesIO
from PIL import Image
import re

async def main():
    print("=" * 80)
    print("测试OCR识别昵称和ID - 多设备版本")
    print("=" * 80)
    
    # 从配置文件读取路径
    adb_path = None
    try:
        import yaml
        with open("config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            
            # 优先使用 adb_path
            adb_path = config.get("adb_path")
            
            # 如果 adb_path 为空，从 nox_path 推导
            if not adb_path or adb_path == '':
                nox_path = config.get("nox_path")
                if nox_path:
                    # MuMu模拟器的ADB路径
                    # nox_path: D:\Program Files\Netease\MuMu\nx_device\12.0\shell
                    # adb_path: D:\Program Files\Netease\MuMu\nx_device\12.0\shell\adb.exe
                    adb_path = os.path.join(nox_path, "adb.exe")
                    if os.path.exists(adb_path):
                        print(f"✓ 从nox_path推导ADB路径: {adb_path}")
                    else:
                        adb_path = None
            else:
                print(f"✓ 从配置文件读取ADB路径: {adb_path}")
    except Exception as e:
        print(f"⚠️ 读取配置文件失败: {e}")
    
    # 如果还是没有，尝试常见路径
    if not adb_path:
        possible_paths = [
            r"D:\Program Files\Netease\MuMu\nx_device\12.0\shell\adb.exe",
            r"C:\Program Files\Netease\MuMu\nx_device\12.0\shell\adb.exe",
            r"D:\Program Files\Nox\bin\nox_adb.exe",
            r"C:\Program Files\Nox\bin\nox_adb.exe",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                adb_path = path
                print(f"✓ 找到ADB: {adb_path}")
                break
    
    if not adb_path or not os.path.exists(adb_path):
        print("❌ 未找到ADB路径")
        print("请检查：")
        print("  1. config.yaml 中的 nox_path 是否正确")
        print("  2. 模拟器是否已安装")
        return
    
    # 获取所有连接的设备
    print("\n[0] 获取设备列表...")
    try:
        result = subprocess.run(
            [adb_path, "devices"],
            capture_output=True,
            timeout=5
        )
        
        if result.returncode != 0:
            print(f"❌ 获取设备列表失败")
            return
        
        # 解析设备列表
        devices = []
        lines = result.stdout.decode('utf-8', errors='ignore').strip().split('\n')
        for line in lines[1:]:  # 跳过第一行 "List of devices attached"
            if '\tdevice' in line or '  device' in line:
                device_id = line.split()[0]
                devices.append(device_id)
        
        if not devices:
            print("❌ 未找到任何设备")
            return
        
        print(f"✓ 找到 {len(devices)} 个设备:")
        for i, device_id in enumerate(devices):
            print(f"  [{i}] {device_id}")
        
    except Exception as e:
        print(f"❌ 获取设备列表失败: {e}")
        return
    
    # 测试每个设备
    for device_index, device_id in enumerate(devices):
        print("\n" + "=" * 80)
        print(f"测试设备 [{device_index}]: {device_id}")
        print("=" * 80)
        
        await test_device(adb_path, device_id, device_index)


async def test_device(adb_path: str, device_id: str, device_index: int):
    """测试单个设备的昵称识别"""
    
    # 1. 截图
    print("\n[1] 正在截图...")
    try:
        result = subprocess.run(
            [adb_path, "-s", device_id, "exec-out", "screencap", "-p"],
            capture_output=True,
            timeout=5
        )
        
        if result.returncode != 0:
            print(f"❌ 截图失败: {result.stderr.decode('utf-8', errors='ignore')}")
            return
        
        screenshot_data = result.stdout
        image = Image.open(BytesIO(screenshot_data))
        print(f"✓ 截图成功: {image.size}")
        
        # 保存截图（带设备编号）
        screenshot_filename = f"test_screenshot_{device_index}.png"
        image.save(screenshot_filename)
        print(f"✓ 截图已保存: {screenshot_filename}")
        
    except Exception as e:
        print(f"❌ 截图失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 2. OCR识别
    print("\n[2] 正在OCR识别...")
    try:
        from rapidocr import RapidOCR
        
        ocr = RapidOCR()
        ocr_result = ocr(image)
        
        # 调试：打印OCR结果的类型和属性
        print(f"[调试] OCR结果类型: {type(ocr_result)}")
        print(f"[调试] OCR结果属性: {dir(ocr_result)}")
        
        # 处理返回结果（新版本返回对象）
        if hasattr(ocr_result, 'data'):
            result = ocr_result.data
            print(f"[调试] ocr_result.data 类型: {type(result)}")
            if result:
                print(f"[调试] ocr_result.data 长度: {len(result)}")
        else:
            result = ocr_result
            print(f"[调试] 直接使用 ocr_result")
        
        if not result:
            print("❌ OCR未识别到任何文本")
            return
        
        print(f"✓ OCR识别到 {len(result)} 个文本")
        
    except Exception as e:
        print(f"❌ OCR识别失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. 提取昵称和ID
    print("\n[3] 提取昵称和ID...")
    print("-" * 80)
    
    # 处理OCR结果格式 - RapidOCROutput对象有txts和boxes属性
    if hasattr(result, 'txts'):
        texts = result.txts
        boxes = result.boxes
        scores = result.scores if hasattr(result, 'scores') else None
    else:
        # 旧格式：列表 [[box, text, score], ...]
        texts = [item[1] for item in result]
        boxes = [item[0] for item in result]
        scores = [item[2] for item in result]
    
    if not texts:
        print("❌ OCR数据为空")
        return
    
    all_texts = []
    for i, text in enumerate(texts):
        confidence = scores[i] if scores and i < len(scores) else 1.0
        all_texts.append(text)
        print(f"  文本: {text:<30} 置信度: {confidence:.2%}")
    
    print("-" * 80)
    
    # 提取昵称
    nickname = None
    # [2026-03-05] 修复昵称识别：删除"溪盟"、"山泉"等可能是昵称一部分的关键字
    exclude_keywords = ["手机", "余额", "积分", "抵扣券", "优惠券", "我的", "设置", "首页", "分类", "商城", "订单", "元", "张", "次"]
    member_keywords = ["钻石会员", "黄金会员", "白金会员", "铂金会员", "普通会员", "初级会员", "银牌会员", "VIP会员", "SVIP", "VIP", "vip会员", "vip", "Vip", "会员"]
    
    # 先找到ID的位置
    id_index = -1
    for i, text in enumerate(all_texts):
        text_no_space = text.replace(" ", "")
        if "ID" in text_no_space or "id" in text_no_space.lower():
            # 确认是用户ID（包含数字）
            if re.search(r'(?:用户)?[Ii][Dd][:：]?(\d+)', text_no_space):
                id_index = i
                print(f"\n  [昵称提取] 找到ID位置: 索引 {i}, 文本: '{text}'")
                break
    
    # 在ID之前的文本中查找昵称
    if id_index >= 0:
        print(f"  [昵称提取] 在ID之前查找昵称...")
        
        # [2026-03-05] 修复昵称识别：扩大检查范围到ID之前的5个文本
        # 检查ID之前的5个文本（覆盖更多可能的昵称位置）
        for i in range(max(0, id_index - 5), id_index):
            text = all_texts[i].strip()
            
            print(f"  [昵称提取] 检查ID之前的文本 {i}: '{text}'")
            
            # 跳过空文本
            if not text:
                print(f"    - 跳过：空文本")
                continue
            
            # 跳过纯数字
            if text.isdigit():
                print(f"    - 跳过：纯数字")
                continue
            
            # 跳过时间格式
            if re.match(r'\d+:\d+', text):
                print(f"    - 跳过：时间格式")
                continue
            
            # 跳过包含冒号的文本
            if ':' in text or '：' in text:
                print(f"    - 跳过：包含冒号")
                continue
            
            # 处理会员标签
            nickname_candidate = text
            for member_kw in member_keywords:
                if member_kw in text:
                    nickname_candidate = text.split(member_kw)[0].strip()
                    print(f"    - 发现会员标签 '{member_kw}'，提取昵称: '{nickname_candidate}'")
                    break
            
            if not nickname_candidate:
                print(f"    - 跳过：提取后为空")
                continue
            
            # 检查排除关键字
            has_keyword = False
            for kw in exclude_keywords:
                if kw in nickname_candidate:
                    has_keyword = True
                    print(f"    - 跳过：包含排除关键字 '{kw}'")
                    break
            if has_keyword:
                continue
            
            # 长度检查
            text_len = len(nickname_candidate)
            if 1 <= text_len <= 20:
                # 检查是否包含中文
                has_chinese = any('\u4e00' <= c <= '\u9fff' for c in nickname_candidate)
                if has_chinese:
                    nickname = nickname_candidate
                    print(f"  [昵称提取] ✓ 基于ID位置找到昵称: '{nickname}'")
                    break
                else:
                    print(f"    - 跳过：不包含中文")
            else:
                print(f"    - 跳过：长度不符 ({text_len} 字符)")
    
    # 如果没找到，使用原来的简单逻辑
    if not nickname:
        print(f"  [昵称提取] 策略1失败，使用简单逻辑...")
        for text in all_texts:
            text = text.strip()
            if not text:
                continue
            
            # 检查是否包含排除关键字
            if any(kw in text for kw in exclude_keywords):
                continue
            
            # 处理会员标签
            for member_kw in member_keywords:
                if member_kw in text:
                    text = text.split(member_kw)[0].strip()
                    break
            
            if not text:
                continue
            
            # 检查长度
            if 2 <= len(text) <= 15:
                # 检查是否包含中文
                has_chinese = any('\u4e00' <= c <= '\u9fff' for c in text)
                if has_chinese:
                    nickname = text
                    break
    
    # 提取用户ID
    user_id = None
    for text in all_texts:
        text_no_space = text.replace(" ", "")
        
        # 模式1: "ID:数字" 或 "用户ID:数字"
        if "ID" in text_no_space or "id" in text_no_space.lower():
            match = re.search(r'(?:用户)?[Ii][Dd][:：]?(\d+)', text_no_space)
            if match:
                user_id = match.group(1)
                break
        
        # 模式2: 纯数字（6-10位）
        if text_no_space.isdigit() and 6 <= len(text_no_space) <= 10:
            user_id = text_no_space
            break
    
    # 4. 显示结果
    print("\n" + "=" * 80)
    print(f"设备 [{device_index}] 识别结果：")
    print("=" * 80)
    print(f"昵称: {nickname if nickname else '未识别'}")
    print(f"用户ID: {user_id if user_id else '未识别'}")
    print("=" * 80)
    
    if not nickname or not user_id:
        print("\n⚠️ 识别失败，请检查：")
        print("  1. 是否在个人页面")
        print("  2. 页面是否完整显示")
        print(f"  3. 查看 test_screenshot_{device_index}.png 确认截图内容")

if __name__ == '__main__':
    asyncio.run(main())
