"""调试YOLO昵称检测 - 查看YOLO识别了哪些区域"""
import sys
import os

# 设置工作目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, 'src')

import cv2
import numpy as np
from pathlib import Path

# 直接导入模型管理器
from model_manager import ModelManager

# 初始化模型管理器
print("=" * 80)
print("初始化模型管理器...")
print("=" * 80)

model_manager = ModelManager()

# 获取profile_logged模型（用于检测个人资料页的昵称区域）
print("\n加载 profile_logged 模型...")
profile_model = model_manager.get_model('profile_logged')

# 查找最近的截图
screenshot_dir = Path("checkin_screenshots/20260202")
if not screenshot_dir.exists():
    print(f"截图目录不存在: {screenshot_dir}")
    sys.exit(1)

screenshots = sorted(screenshot_dir.glob("*.png"))
print(f"\n找到 {len(screenshots)} 个截图")

# 选择前5个截图进行测试
test_screenshots = screenshots[:5]

print("\n" + "=" * 80)
print("开始检测...")
print("=" * 80)

for i, screenshot_path in enumerate(test_screenshots, 1):
    print(f"\n{'=' * 80}")
    print(f"截图 {i}: {screenshot_path.name}")
    print(f"{'=' * 80}")
    
    # 读取图片
    image = cv2.imread(str(screenshot_path))
    if image is None:
        print(f"  ✗ 无法读取图片")
        continue
    
    print(f"  图片尺寸: {image.shape[1]}x{image.shape[0]}")
    
    # 使用YOLO检测
    try:
        # 使用profile_logged模型检测
        results = profile_model(image, conf=0.3, verbose=False)
        
        if not results or len(results) == 0:
            print(f"  ✗ YOLO未检测到任何区域")
            continue
        
        # 获取第一个结果（通常只有一个）
        result = results[0]
        
        if result.boxes is None or len(result.boxes) == 0:
            print(f"  ✗ YOLO未检测到任何区域")
            continue
        
        boxes = result.boxes
        print(f"  ✓ YOLO检测到 {len(boxes)} 个区域:")
        print()
        
        # 显示每个检测结果
        for j, box in enumerate(boxes, 1):
            # 获取类别
            cls_id = int(box.cls[0])
            class_name = result.names[cls_id] if result.names else f"class_{cls_id}"
            
            # 获取置信度
            confidence = float(box.conf[0])
            
            # 获取边界框坐标
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = xyxy
            width = x2 - x1
            height = y2 - y1
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            print(f"    区域 {j}:")
            print(f"      标签: {class_name}")
            print(f"      置信度: {confidence:.3f}")
            print(f"      位置: ({x1:.0f}, {y1:.0f}) -> ({x2:.0f}, {y2:.0f})")
            print(f"      尺寸: {width:.0f}x{height:.0f}")
            print(f"      中心: ({center_x:.0f}, {center_y:.0f})")
            print()
        
        # 查找昵称区域
        nickname_boxes = []
        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = result.names[cls_id] if result.names else f"class_{cls_id}"
            if 'nickname' in class_name.lower() or 'name' in class_name.lower():
                nickname_boxes.append((box, class_name))
        
        if nickname_boxes:
            print(f"  ✓ 找到 {len(nickname_boxes)} 个昵称相关区域:")
            for box, class_name in nickname_boxes:
                confidence = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                print(f"    - {class_name} (置信度: {confidence:.3f})")
                print(f"      位置: ({x1:.0f}, {y1:.0f}) -> ({x2:.0f}, {y2:.0f})")
        else:
            print(f"  ✗ 未找到昵称相关区域")
            print(f"  提示: 这可能导致降级到全屏OCR")
            print(f"  检测到的类别: {[result.names[int(box.cls[0])] for box in boxes]}")
        
        # 可视化检测结果（保存到debug目录）
        debug_dir = Path("debug_yolo_output")
        debug_dir.mkdir(exist_ok=True)
        
        # 在图片上绘制检测框
        vis_image = image.copy()
        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = result.names[cls_id] if result.names else f"class_{cls_id}"
            confidence = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            
            # 昵称区域用绿色，其他用蓝色
            if 'nickname' in class_name.lower() or 'name' in class_name.lower():
                color = (0, 255, 0)  # 绿色
                thickness = 3
            else:
                color = (255, 0, 0)  # 蓝色
                thickness = 2
            
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)
            
            # 添加标签
            text = f"{class_name} {confidence:.2f}"
            cv2.putText(vis_image, text, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 保存可视化结果
        output_path = debug_dir / f"yolo_detection_{screenshot_path.name}"
        cv2.imwrite(str(output_path), vis_image)
        print(f"\n  可视化结果已保存: {output_path}")
        
    except Exception as e:
        print(f"  ✗ 检测失败: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("检测完成")
print("=" * 80)
print(f"\n可视化结果保存在: debug_yolo_output/")
print("绿色框 = 昵称相关区域")
print("蓝色框 = 其他检测区域")
