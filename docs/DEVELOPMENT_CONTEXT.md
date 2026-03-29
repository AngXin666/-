# 开发上下文

最后更新：2026-03-30  
项目根目录：`D:\360MoveData\Users\Administrator\Desktop\zdqd`

## 1. 运行入口

- 主入口：`run.py`
- 运行主路径：
  1. 许可证校验（`src/license_manager_simple.py`）
  2. 模拟器与 ADB 初始化（`src/adb_bridge.py`、`src/emulator_controller.py`）
  3. GUI 启动（`src/gui.py`）
  4. 账号流程执行（`src/ximeng_automation.py`）

## 2. 核心模块

- `src/gui.py`：主界面与流程胶水层
- `src/ximeng_automation.py`：端到端业务流程
- `src/auto_login.py`：登录与缓存登录判定
- `src/daily_checkin.py`：签到流程与奖励处理
- `src/balance_transfer.py`：转账流程
- `src/model_manager.py`：模型加载与共享注册
- `src/page_detector_integrated.py` + `src/page_detector_dl.py`：页面状态识别
- `src/local_db.py`：历史与结果持久化

## 3. 当前热点

- 体量热点：
  - `src/gui.py` 约 10520 行
  - `src/ximeng_automation.py` 约 2511 行
  - `src/daily_checkin.py` 约 1861 行
- 高耦合区域：
  - GUI 层混合了 UI、调度与业务逻辑
  - 工作流跨模块分支较多
- 线上修补痕迹重：
  - 大量带日期注释的“修复原因”说明该项目长期在生产态排障

## 4. 工作树状态

- 当前分支：`master`
- 工作树处于活跃/脏状态（有大量 modified/untracked）。
- 开发约束建议：
  - 仅做最小必要改动
  - 不触碰无关 backup/debug 产物
  - 每次编辑前先确认目标文件与范围

## 5. 稳妥开发优先级

1. 先稳关键链路，降低回归
   - 范围：`src/adb_bridge.py`、`src/emulator_controller.py`、`src/auto_login.py`
   - 目标：超时/重试/状态切换可预测

2. 再做 GUI 与流程解耦
   - 从抽离 `_process_account_with_instance` 依赖开始
   - GUI 仅保留 UI 状态与事件

3. 补最小烟雾测试
   - ADB 包装器的超时/重试行为
   - 配置加载与流程模式解析
   - 账号文件加载边界场景

## 6. 实操改动策略

- 一次只做一个窄改动。
- 每次改动明确记录“修改原因”。
- 优先可逆重构：
  - 先抽函数/抽模块
  - 保持旧调用点可用
  - 先不改行为，再做行为修复

## 7. 关键入口位置

- GUI 启动序列：`src/gui.py`（`_start_automation`、`_run_automation_thread`）
- 单账号执行：`src/gui.py`（`_process_account_with_instance`）
- 工作流主函数：`src/ximeng_automation.py`（`run_full_workflow`）
- 启动流处理：`src/ximeng_automation.py`（`handle_startup_flow_integrated`）
