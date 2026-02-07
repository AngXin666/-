# 训练脚本使用指南

## 统一脚本（推荐使用）

### 1. 数据准备 - prepare_dataset.py
```bash
# 准备页面分类器数据
python prepare_dataset.py --type page_classifier

# 准备个人页详细标注数据
python prepare_dataset.py --type profile_detailed

# 准备个人页区域数据
python prepare_dataset.py --type profile_regions

# 准备个人页数字数据
python prepare_dataset.py --type profile_numbers

# 准备签到弹窗数据
python prepare_dataset.py --type checkin_popup

# 准备完整分类器数据
python prepare_dataset.py --type full_classifier
```

### 2. 数据增强 - augment_dataset.py
```bash
# 4类数据增强
python augment_dataset.py --type 4class

# 页面分类器数据增强
python augment_dataset.py --type page_classifier

# 个人页详细标注数据增强
python augment_dataset.py --type profile_detailed

# 个人页区域数据增强
python augment_dataset.py --type profile_regions
```

### 3. YOLO训练 - train_yolo.py
```bash
# 训练阶段1模型（核心按钮）
python train_yolo.py --type stage1 --epochs 30 --batch 16

# 训练个人页区域检测
python train_yolo.py --type profile_regions

# 训练个人页数字识别
python train_yolo.py --type profile_numbers

# 训练个人页详细标注
python train_yolo.py --type profile_detailed
```

### 4. 分类器训练 - train_classifier.py
```bash
# Keras版本
python train_classifier.py --type keras --epochs 30

# PyTorch版本
python train_classifier.py --type pytorch --epochs 30

# 二分类版本
python train_classifier.py --type binary

# 4类分类器
python train_classifier.py --type 4class
```

### 5. 训练监控 - monitor.py
```bash
# 监控训练进度
python monitor.py --type training

# 监控性能
python monitor.py --type performance
```

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
