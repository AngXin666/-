# 打包前检查清单

## 1. 文件完整性检查
- [ ] config/yolo_model_registry.json 存在
- [ ] models/page_yolo_mapping.json 存在
- [ ] config/page_state_mapping.json 存在
- [ ] config/page_classes.json 存在
- [ ] models/page_classifier_pytorch_best.pth 存在
- [ ] 所有YOLO模型文件存在（检查注册表中的路径）

## 2. 打包脚本检查
- [ ] build_exe_optimized.py 包含 ('config', 'config')
- [ ] build_exe_optimized.py 包含 ('models', 'models')
- [ ] build_exe_optimized.py 包含文件结构修复逻辑

## 3. 打包步骤
1. 运行: python build_exe_optimized.py
2. 等待打包完成
3. 检查 dist/溪盟商城自动化助手/ 目录

## 4. 打包后检查
- [ ] dist/溪盟商城自动化助手/config/ 目录存在
- [ ] dist/溪盟商城自动化助手/models/ 目录存在
- [ ] config/yolo_model_registry.json 在根目录的config下
- [ ] models/page_yolo_mapping.json 在根目录的models下
- [ ] 所有YOLO模型文件在根目录的models下

## 5. 功能测试
1. 复制 test_packed_models.py 到打包目录
2. 运行测试脚本
3. 检查所有文件是否能正确找到
4. 启动主程序测试模型加载

## 6. 常见问题
- 如果models在_internal下：检查copy_additional_files()函数
- 如果config在_internal下：检查copy_additional_files()函数
- 如果模型加载失败：运行diagnose_model_loading.py诊断