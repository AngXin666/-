# 训练脚本使用指南

## 统一脚本（推荐使用 - 菜单选择）

### 1. 数据准备
```bash
python prepare_dataset.py
```
运行后会显示菜单，选择数据集类型：
- 1. 页面分类器数据
- 2. 个人页详细标注数据
- 3. 个人页区域数据
- 4. 个人页数字数据
- 5. 签到弹窗数据
- 6. 完整分类器数据

### 2. 数据增强
```bash
python augment_dataset.py
```
运行后会显示菜单，选择增强类型：
- 1. 4类数据增强
- 2. 页面分类器数据增强
- 3. 个人页详细标注数据增强
- 4. 个人页区域数据增强

### 3. YOLO训练
```bash
python train_yolo.py
```
运行后会显示菜单，选择模型类型：
- 1. 阶段1模型（核心按钮）
- 2. 个人页区域检测
- 3. 个人页数字识别
- 4. 个人页详细标注

### 4. 分类器训练
```bash
python train_classifier.py
```
运行后会显示菜单，选择分类器类型：
- 1. Keras版本
- 2. PyTorch版本
- 3. 二分类版本
- 4. 4类分类器

### 5. 训练监控
```bash
python monitor.py
```
运行后会显示菜单，选择监控类型：
- 1. 训练进度监控
- 2. 性能监控

## 原始脚本（仍然可用）

如果需要更多控制，可以直接运行原始脚本：

### 数据准备
- `prepare_page_classifier_data.py`
- `prepare_profile_detailed_data.py`
- `prepare_profile_region_data.py`
- `prepare_profile_numbers_dataset.py`
- `prepare_checkin_popup_dataset.py`
- `prepare_full_classifier_dataset.py`

### 数据增强
- `augment_4class_data.py`
- `augment_page_classifier_updated.py`
- `augment_profile_detailed_fixed.py`
- `augment_profile_regions.py`

### YOLO训练
- `train_yolo_stage1.py`
- `train_profile_regions_yolo.py`
- `train_profile_numbers_yolo.py`
- `train_profile_detailed.py`

### 分类器训练
- `train_page_classifier.py`
- `train_page_classifier_pytorch.py`
- `train_page_classifier_binary_v2.py`
- `train_4class_classifier.py`

### 训练监控
- `monitor_improved_training.py`
- `monitor_performance.py`

## 工具脚本

### 数据集分割
```bash
python split_dataset.py
```

### 批处理工具
- `quick_start.bat` - 快速启动
- `clean_and_run.bat` - 清理并运行
- `deep_clean.bat` - 深度清理
- `kill_all.bat` - 终止所有进程
- `backup_templates.bat` - 备份模板
- `restore_templates.bat` - 恢复模板

### GPU支持
- `install_gpu.bat` - 安装GPU支持
- `install_tensorflow_gpu.bat` - 安装TensorFlow GPU
- `setup_cuda_env.bat` - 设置CUDA环境

### PowerShell工具
- `emergency_kill.ps1` - 紧急终止
- `monitor_restart.ps1` - 监控重启
