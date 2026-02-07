# 训练脚本使用指南

## 📌 重要说明

**训练功能已集成到标注工具中!**

推荐使用标注工具的"训练管理"功能,更加方便快捷:
```bash
python dev_tools/annotation_tool.py
```
点击"🎓 训练管理"按钮即可访问所有训练功能。

---

## 📁 文件说明

### 统一训练脚本(5个)
这些脚本提供了菜单式的训练入口,也可以独立使用:

1. **prepare_dataset.py** - 数据准备
   - 页面分类器数据
   - 个人页详细标注数据
   - 个人页区域数据
   - 个人页数字数据
   - 签到弹窗数据
   - 完整分类器数据

2. **augment_dataset.py** - 数据增强
   - 4类数据增强
   - 页面分类器数据增强
   - 个人页详细标注数据增强
   - 个人页区域数据增强

3. **train_yolo.py** - YOLO训练
   - 阶段1模型(核心按钮)
   - 个人页区域检测
   - 个人页数字识别
   - 个人页详细标注

4. **train_classifier.py** - 分类器训练
   - Keras版本
   - PyTorch版本
   - 二分类版本
   - 4类分类器

5. **monitor.py** - 训练监控
   - 训练进度监控
   - 性能监控

### 工具脚本(7个)

1. **quick_start.bat** - 快速启动主程序
2. **clean_and_run.bat** - 清理缓存并运行
3. **kill_all.bat** - 终止所有相关进程
4. **backup_templates.bat** - 备份模板文件
5. **restore_templates.bat** - 恢复模板文件
6. **install_gpu.bat** - GPU加速安装(保留用于其他电脑)
7. **install_tensorflow_gpu.bat** - TensorFlow GPU安装

---

## 🚀 使用方法

### 方法1: 使用标注工具(推荐)
```bash
python dev_tools/annotation_tool.py
```
点击"🎓 训练管理"按钮,选择要执行的任务。

### 方法2: 独立运行脚本
```bash
# 数据准备
python 脚本/prepare_dataset.py

# 数据增强
python 脚本/augment_dataset.py

# YOLO训练
python 脚本/train_yolo.py

# 分类器训练
python 脚本/train_classifier.py

# 训练监控
python 脚本/monitor.py
```

---

## 📊 训练流程

1. **数据准备** → 准备训练数据集
2. **数据增强** → 增强训练数据(可选)
3. **模型训练** → 训练YOLO或分类器模型
4. **训练监控** → 监控训练进度和性能

---

## 💡 提示

- 所有训练脚本都提供了菜单式选择,操作简单
- 训练功能已完全集成到标注工具中,推荐使用
- GPU安装脚本保留用于其他电脑的环境配置
- 工具脚本提供了常用的辅助功能
