
import sys
import os
sys.path.insert(0, r'D:\360MoveData\Users\Administrator\Desktop\模拟\标注工具_完整独立版\..\脚本')

# 设置训练参数
category = '转账页'
print(f"开始训练类别: {category}")

# 这里调用实际的YOLO训练函数
# 你需要根据实际的训练脚本来调整
from train_yolo_stage1 import train_model
train_model(category=category)
