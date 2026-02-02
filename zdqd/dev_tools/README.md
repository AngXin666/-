# 开发工具目录

本目录包含开发和测试过程中使用的各种工具脚本。

## 📁 目录结构

### test_scripts/ - 测试脚本

用于测试各种功能的脚本：

**功能测试**：
- `test_single_account.py` - 单账号完整流程测试
- `test_two_instances_full.py` - 双实例完整流程测试
- `test_multi_instance_real.py` - 多实例真实测试
- `test_checkin_only.py` - 签到功能测试
- `test_checkin_simple.py` - 简单签到测试
- `test_profile_info.py` - 个人信息读取测试

**组件测试**：
- `test_template_matching.py` - 模板匹配测试
- `test_navigate_profile.py` - 导航功能测试
- `test_my_button_click.py` - "我的"按钮点击测试
- `test_performance_quick.py` - 性能快速测试

**登录测试**：
- `test_auto_input_login.py` - 自动输入登录测试
- `test_agreement_and_login_button.py` - 协议和登录按钮测试
- `test_agreement_ocr_multiple.py` - 协议 OCR 多次测试
- `test_login_button_coords.py` - 登录按钮坐标测试
- `test_login_entry_click.py` - 登录入口点击测试
- `test_main_script_login.py` - 主脚本登录测试

**缓存测试**：
- `test_account_cache_match.py` - 账号缓存匹配测试
- `test_account_switch.py` - 账号切换测试
- `test_cache_clear_complete.py` - 缓存清理完整测试
- `test_cache_overwrite.py` - 缓存覆盖测试
- `test_cache_restore_flow.py` - 缓存恢复流程测试
- `test_clear_account_cache.py` - 清理账号缓存测试

**诊断工具**：
- `diagnose_cache_clear.py` - 缓存清理诊断
- `check_page.py` - 页面检查工具

### record_scripts/ - 坐标记录脚本

用于记录屏幕坐标和捕获元素位置的工具：

- `record_coords_simple.py` - 简单坐标记录工具
- `record_mouse_position.py` - 鼠标位置记录工具
- `record_mumu_coords.py` - MuMu 模拟器坐标记录
- `record_my_button.py` - "我的"按钮坐标记录
- `record_checkin_button.py` - 签到按钮坐标记录
- `record_agreement_checkbox.py` - 协议勾选框坐标记录
- `capture_agreement_checkbox.py` - 捕获协议勾选框
- `capture_login_screen.py` - 捕获登录屏幕

### utility_scripts/ - 工具脚本

通用工具脚本：

- `加密模板文件.py` - 加密模板文件工具
- `解密模板文件.py` - 解密模板文件工具

### test_screenshots/ - 测试截图

测试过程中保存的截图：

- `ad_test_screenshots/` - 广告测试截图
- `checkin_test_screenshots/` - 签到测试截图
- `transfer_test_screenshots/` - 转账测试截图
- `wallet_test_screenshots/` - 钱包测试截图

### apk_analysis/ - APK 分析

APK 文件分析相关：

- `app.apk` - 应用 APK 文件
- `extracted/` - 提取的 APK 内容

## 📋 批处理文件

- `test_exe_build.bat` - 测试 EXE 构建
- `测试多模拟器.bat` - 测试多模拟器功能
- `extract_and_analyze_apk.bat` - 提取和分析 APK

## 🔧 使用说明

### 运行测试脚本

```bash
# 单账号测试
python dev_tools/test_scripts/test_single_account.py

# 签到测试
python dev_tools/test_scripts/test_checkin_simple.py

# 多实例测试
python dev_tools/test_scripts/test_multi_instance_real.py
```

### 记录坐标

```bash
# 记录简单坐标
python dev_tools/record_scripts/record_coords_simple.py

# 记录 MuMu 模拟器坐标
python dev_tools/record_scripts/record_mumu_coords.py
```

### 使用工具脚本

```bash
# 加密模板文件
python dev_tools/utility_scripts/加密模板文件.py

# 解密模板文件
python dev_tools/utility_scripts/解密模板文件.py
```

## 📝 注意事项

1. 测试脚本中的账号信息仅用于测试，请勿泄露
2. 坐标记录脚本需要连接模拟器才能使用
3. 部分测试脚本可能已过时，使用前请检查代码
4. 建议在测试环境中运行，避免影响生产数据

---

**最后更新**: 2026-01-23
