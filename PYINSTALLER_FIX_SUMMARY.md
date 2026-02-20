# PyInstaller 打包问题修复方案

## 问题分析

打包后的程序出现以下问题：
1. 批量添加账号时卡死（CPU 100%）
2. 自动启动第二个进程（无限重启）
3. 线程数异常高（40+和80+个线程）

## 根本原因

**Windows 平台上使用 multiprocessing 和 ThreadPoolExecutor 时，PyInstaller 打包需要特殊处理：**

1. **缺少 `multiprocessing.freeze_support()`** - 这是 Windows 平台必需的
2. **多个地方使用了 ThreadPoolExecutor** - 在打包环境中可能导致线程泄漏
3. **打包脚本可能缺少关键配置** - 需要添加 multiprocessing 相关的 hook

## 修复方案

### 1. run.py - 添加 freeze_support()（已修复）

```python
def main():
    """主函数"""
    
    # ============================================================
    # 【关键修复】PyInstaller 打包后必须调用 freeze_support()
    # 这是 Windows 平台上使用 multiprocessing 的必需配置
    # 否则会导致无限重启和进程卡死问题
    # ============================================================
    if getattr(sys, 'frozen', False):
        # 打包后的程序必须调用 freeze_support()
        import multiprocessing
        multiprocessing.freeze_support()
```

### 2. build_exe.py - 添加 multiprocessing 配置（待修复）

需要在打包脚本中添加：
- `--runtime-hook` 用于 multiprocessing 初始化
- 确保 `concurrent.futures` 正确打包

### 3. 检查所有使用 ThreadPoolExecutor 的地方

项目中使用 ThreadPoolExecutor 的文件：
- `src/ocr_thread_pool.py` - OCR 线程池（8个线程）
- `src/model_manager.py` - 模型加载线程池
- `src/login_cache_manager.py` - 缓存加密/解密线程池（8个线程）
- `src/emulator_controller.py` - 模拟器连接线程池（10个线程）
- `src/gui.py` - GUI 主线程池（动态数量）
- `src/user_management_gui.py` - 批量添加账号（已改为单线程）

### 4. 可能需要的额外修复

如果问题依然存在，可能需要：
1. 在所有 ThreadPoolExecutor 创建时添加 `initializer` 参数
2. 确保所有线程池正确关闭（使用 `with` 语句或显式 `shutdown()`）
3. 添加 PyInstaller 的 runtime hook 来初始化 multiprocessing

## 下一步行动

1. 修复 build_exe.py，添加 multiprocessing 相关配置
2. 重新打包测试
3. 如果问题依然存在，逐个检查 ThreadPoolExecutor 的使用
