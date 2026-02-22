"""
可视化签到按钮坐标
在首页截图上标注默认签到按钮位置
"""
import cv2
from pathlib import Path

# 默认签到按钮坐标（下移5像素后）
CHECKIN_BUTTON = (477, 548)

# 合理范围（下移5像素后）
CHECKIN_BUTTON_VALID_RANGE = (400, 540, 450, 600)  # (x_min, x_max, y_min, y_max)

# 使用最新的首页截图
screenshot_path = Path("checkin_screenshots/20260222/245.png")

if not screenshot_path.exists():
    print(f"截图不存在: {screenshot_path}")
    exit(1)

# 读取截图
img = cv2.imread(str(screenshot_path))
if img is None:
    print(f"无法读取截图: {screenshot_path}")
    exit(1)

print(f"截图尺寸: {img.shape[1]}x{img.shape[0]}")

# 标注默认坐标（红色圆点）
x, y = CHECKIN_BUTTON
cv2.circle(img, (x, y), 20, (0, 0, 255), 3)  # 红色圆圈
cv2.circle(img, (x, y), 5, (0, 0, 255), -1)  # 红色实心点
cv2.putText(img, f"Default: ({x}, {y})", (x + 25, y), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

# 标注合理范围（绿色矩形）
x_min, x_max, y_min, y_max = CHECKIN_BUTTON_VALID_RANGE
cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
cv2.putText(img, "Valid Range", (x_min, y_min - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# 标注错误坐标（如果有）
wrong_coord = (540, 920)
if wrong_coord[1] <= img.shape[0]:  # 如果在图片范围内
    wx, wy = wrong_coord
    cv2.circle(img, (wx, wy), 20, (255, 0, 0), 3)  # 蓝色圆圈
    cv2.circle(img, (wx, wy), 5, (255, 0, 0), -1)  # 蓝色实心点
    cv2.putText(img, f"Wrong: ({wx}, {wy})", (wx + 25, wy), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
else:
    print(f"错误坐标 {wrong_coord} 超出图片范围")

# 保存标注后的图片
output_path = Path("dev_tools/checkin_button_visualization.png")
cv2.imwrite(str(output_path), img)
print(f"✓ 已保存标注图片: {output_path}")

# 自动打开图片
import os
os.startfile(str(output_path))
print("✓ 已自动打开图片")
