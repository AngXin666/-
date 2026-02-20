# 打包优化方案

## 问题分析

打包后的程序约5GB，主要占用：
1. **torch文件夹：2.4GB** - PyTorch深度学习框架（包含CUDA）
2. **models文件夹：1.1GB** - 包含大量训练中间文件
3. **其他依赖：1.5GB** - numpy, pandas, cv2, sklearn等

## 优化措施

### 1. 清理models文件夹（节省约1GB）
**问题**：models文件夹包含大量YOLO训练过程文件
- 多个`best.pt`和`last.pt`（每个95-96MB）
- 训练过程图片（train_batch*.jpg, val_batch*.jpg）
- 训练曲线图（BoxF1_curve.png等）

**解决方案**：只保留必需的模型文件
- `page_classifier_pytorch_best.pth` - 页面分类器
- `yolo26n.pt` / `yolov8n.pt` - YOLO模型
- 配置文件（json）

**实施**：打包前自动清理

### 2. 排除不需要的库（节省约500MB）
**排除的库**：
- `sklearn` - 如果不使用机器学习
- `wandb` - 实验跟踪工具
- `tensorboard` - 可视化工具
- `tensorrt` - NVIDIA推理加速（如果不用）
- `polars` - 数据处理库（150MB）
- 测试模块（pandas.tests, numpy.tests等）

### 3. 排除OpenCV视频模块（节省约30MB）
**问题**：项目只使用OpenCV的图像处理功能，不需要视频功能
- `opencv_videoio_ffmpeg4120_64.dll` (14MB)
- `opencv_videoio_ffmpeg4100_64.dll` (14MB)

**解决方案**：排除cv2.videoio和cv2.video模块

### 4. PyTorch优化（可选）
**问题**：torch文件夹2.4GB，主要是CUDA库

**方案A**：创建两个版本
- CPU版本（约2.5GB）- 不包含CUDA
- GPU版本（约5GB）- 包含完整CUDA支持

**方案B**：延迟下载
- 打包时不包含torch
- 首次运行检测GPU，按需下载

**当前方案**：保留完整torch（因为项目使用GPU加速）

## 预期效果

优化后预计大小：
- 清理models：-1GB
- 排除不需要的库：-500MB
- 排除OpenCV视频：-30MB
- **总计节省：约1.5GB**
- **优化后大小：约3.5GB**

## CMD窗口问题修复

### 根本原因
PyInstaller打包后，subprocess调用外部程序（如adb.exe）时会显示CMD窗口

### 解决方案
1. **Runtime Hook**：创建`pyi_rth_subprocess.py`，在程序启动时自动patch所有subprocess调用
2. **自动添加参数**：所有subprocess.Popen和subprocess.run自动添加：
   - `startupinfo=STARTUPINFO`（隐藏窗口）
   - `creationflags=CREATE_NO_WINDOW`（不创建新窗口）

### 实施
- 已创建runtime hook文件
- 已在build_exe.py中添加`--runtime-hook`参数
- 打包后自动生效，无需修改源代码

## 使用方法

直接运行打包脚本：
```bash
python build_exe.py
```

脚本会自动：
1. 清理models文件夹
2. 排除不需要的模块
3. 应用CMD窗口修复
4. 打包到D盘

## 注意事项

1. **首次打包前备份models文件夹**（如果需要保留训练文件）
2. **测试打包后的程序**：
   - 启动是否正常
   - 是否还有CMD窗口
   - 模型加载是否正常
   - GPU加速是否工作
3. **如果需要更小的体积**，考虑创建CPU版本（不包含CUDA）
