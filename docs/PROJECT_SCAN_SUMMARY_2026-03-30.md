# 项目全量扫描摘要（2026-03-30）

## 扫描范围

本次扫描覆盖整个项目工作区：  
`D:\360MoveData\Users\Administrator\Desktop\zdqd`

## 全局盘点

- 索引文件总数（`rg --files`）：`610`
- 按顶层目录统计（包含运行产物）：
  - `logs`: 2946
  - `login_cache`: 2843
  - `reports`: 857
  - `runtime_data`: 500
  - `models`: 215
  - `dev_tools`: 177
  - `src`: 158
  - `docs`: 28

## 开发相关范围盘点

- 本次开发检索范围：
  - `src`
  - `docs`
  - `config`
  - `dev_tools`
  - `tools`
  - `training_scripts`
  - `build_scripts`
- 上述开发范围文件数：`330`

## 入口与运行链路

- 主入口：`run.py`
- 批处理入口：`start.bat`
- 运行链路：
  1. 许可证校验
  2. 模拟器与 ADB 初始化
  3. GUI 启动
  4. 账号流程执行

## 主要代码面

- 体量最大的核心文件：
  - `src/gui.py`（约 10520 行）
  - `src/ximeng_automation.py`（约 2511 行）
  - `src/daily_checkin.py`（约 1861 行）
  - `src/profile_reader.py`（大模块）
  - `src/user_management_gui.py`（大模块）

- 核心模块分布：
  - GUI 与编排：`src/gui.py`、`src/orchestrator.py`
  - 业务流程：`src/ximeng_automation.py`
  - 登录与缓存：`src/auto_login.py`、`src/login_cache_manager.py`
  - 签到与转账：`src/daily_checkin.py`、`src/balance_transfer.py`
  - 模拟器与 ADB：`src/emulator_controller.py`、`src/adb_bridge.py`
  - 检测与模型：`src/model_manager.py`、`src/page_detector*.py`
  - 数据持久化：`src/local_db.py`、`src/transfer_history.py`

## Python 符号扫描

- 已对 `src`（`*.py`，排除 backup）执行：
  - `class/def` 全量扫描
  - `async def` 并发关键路径扫描
  - `import` 依赖关系扫描

## 测试与校验脚本

- 根目录测试脚本：
  - `test_cache_lookup.py`
  - `test_launch_instance.py`
  - `test_transfer_config_checkin.py`
- 诊断/检查/修复脚本主要位于：
  - `dev_tools/`
  - `dev_tools/check_scripts/`
  - `tools/`

## 关键观察

- 仓库内运行数据与运维产物占比很高。
- 当前工作树较活跃，后续开发应保持“小步、聚焦、可回滚”的最小改动策略。
