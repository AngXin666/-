# PyInstaller 打包完整检查清单

## ✅ 已修复的问题

### 1. multiprocessing 支持（关键）
- ✅ `run.py` - 添加了 `multiprocessing.freeze_support()`
- ✅ `build_exe.py` - 添加了 multiprocessing 相关的 hidden-import
- ✅ `crypto_utils.py` - 已移除 wmic 命令，使用纯 Python 方式

### 2. 文件路径处理
- ✅ `model_manager.py` - 正确使用 `sys.executable` 和 `__file__`
- ✅ `page_state_dynamic.py` - 正确处理打包后的路径
- ✅ `license_manager_simple.py` - 正确处理打包后的路径
- ✅ `auto_model_registry.py` - 正确处理打包后的路径
- ✅ `auto_page_type_registry.py` - 正确处理打包后的路径

### 3. subprocess 调用
- ✅ `emulator_controller.py` - 所有 subprocess 调用都使用了 STARTUPINFO 隐藏窗口
- ✅ `adb_bridge.py` - 所有 subprocess 调用都使用了 STARTUPINFO 隐藏窗口

## ⚠️ 潜在风险点（需要测试）

### 1. ThreadPoolExecutor 使用（多处）
以下文件使用了 ThreadPoolExecutor，可能在打包后导致线程问题：

- `src/ocr_thread_pool.py` - 8个线程的OCR线程池
- `src/model_manager.py` - 模型加载线程池
- `src/login_cache_manager.py` - 缓存加密/解密线程池（8个线程）
- `src/emulator_controller.py` - 模拟器连接线程池（10个线程）
- `src/gui.py` - GUI主线程池（动态数量）
- ✅ `src/user_management_gui.py` - 已改为单线程

**风险评估**：
- 低风险：`model_manager.py`（只在启动时使用一次）
- 低风险：`emulator_controller.py`（短时间使用）
- 中风险：`ocr_thread_pool.py`（长时间运行）
- 中风险：`login_cache_manager.py`（批量操作时使用）
- 高风险：`gui.py`（主线程池，长时间运行）

**建议**：
- 如果测试时发现线程问题，需要在所有 ThreadPoolExecutor 创建时添加：
  ```python
  if getattr(sys, 'frozen', False):
      # 打包后使用更保守的线程数
      max_workers = min(max_workers, 4)
  ```

### 2. 数据文件打包
需要确保以下文件/文件夹正确打包：

- ✅ `config/` - 配置文件夹
- ✅ `models/` - 模型文件夹
- ✅ `config.yaml` - 主配置文件
- ✅ `model_config.json.example` - 模型配置示例
- ✅ `transfer_config.json.example` - 转账配置示例
- ✅ `.env.example` - 环境变量示例

### 3. 运行时创建的目录
打包脚本已创建以下空目录：

- ✅ `data/` - 账号文件目录
- ✅ `login_cache/` - 登录缓存
- ✅ `screenshots/` - 截图
- ✅ `logs/` - 日志
- ✅ `reports/` - 报告
- ✅ `runtime_data/` - 运行时数据
- ✅ `checkin_screenshots/` - 签到截图
- ✅ `no_checkin_screenshots/` - 未签到截图

## 🔍 需要测试的功能

### 高优先级（核心功能）
1. ✅ 程序启动（不会无限重启）
2. ⚠️ 批量添加账号（100+账号，不会卡死）
3. ⚠️ 账号登录和签到
4. ⚠️ 转账功能
5. ⚠️ 模型加载和识别

### 中优先级（辅助功能）
6. ⚠️ 用户管理
7. ⚠️ 转账历史查询
8. ⚠️ 历史结果查询
9. ⚠️ 窗口排列
10. ⚠️ 定时运行

### 低优先级（边缘功能）
11. ⚠️ 注册新模型
12. ⚠️ 流程控制
13. ⚠️ 日志过滤和搜索

## 📋 测试步骤

### 第一阶段：基础功能测试
1. 启动程序，检查是否正常显示界面
2. 检查进程数量（应该只有1个）
3. 检查线程数量（应该在合理范围内，<30）
4. 检查CPU和内存使用（应该正常）

### 第二阶段：批量添加账号测试
1. 打开用户管理
2. 批量添加100个测试账号
3. 监控进程状态（不应该出现第二个进程）
4. 监控CPU使用率（不应该持续100%）
5. 监控线程数（不应该异常增长）
6. 确认添加成功

### 第三阶段：核心功能测试
1. 测试账号登录
2. 测试签到功能
3. 测试转账功能
4. 测试模型识别

## 🚨 已知问题和解决方案

### 问题1：批量添加账号卡死
**原因**：使用了 ThreadPoolExecutor 多线程
**解决方案**：已改为单线程顺序处理

### 问题2：无限重启
**原因**：缺少 `multiprocessing.freeze_support()`
**解决方案**：已在 run.py 中添加

### 问题3：线程数异常高
**原因**：多个地方使用了 ThreadPoolExecutor
**解决方案**：已添加 multiprocessing 支持，如果问题依然存在，需要限制线程数

## 📝 打包命令

```bash
python build_exe.py
```

## 🎯 成功标准

1. 程序启动正常，只有1个进程
2. 批量添加100个账号不卡死，不出现第二个进程
3. CPU使用率正常（不持续100%）
4. 线程数正常（<30个）
5. 所有核心功能正常工作

## 🔧 如果测试失败

### 如果批量添加账号还是卡死：
1. 检查 `src/user_management_gui.py` 是否真的改为单线程
2. 检查是否有其他地方使用了 ThreadPoolExecutor
3. 添加更多日志，定位卡死位置

### 如果还是无限重启：
1. 检查 `run.py` 中的 `freeze_support()` 是否生效
2. 检查是否有其他地方创建了子进程
3. 检查 `close_old_instances()` 是否正确跳过打包后的程序

### 如果线程数异常高：
1. 限制所有 ThreadPoolExecutor 的 max_workers
2. 确保所有线程池正确关闭（使用 `with` 语句或显式 `shutdown()`）
3. 检查是否有线程泄漏

## ✅ 最终检查

- [ ] run.py 包含 `multiprocessing.freeze_support()`
- [ ] build_exe.py 包含 multiprocessing 相关配置
- [ ] user_management_gui.py 改为单线程
- [ ] 所有文件路径正确处理 `sys.frozen`
- [ ] 所有 subprocess 调用使用 STARTUPINFO
- [ ] 所有必需的数据文件已打包
- [ ] 运行时目录已创建
