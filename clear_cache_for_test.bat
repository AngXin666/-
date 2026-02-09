@echo off
chcp 65001 >nul
echo ============================================================
echo 清理缓存和日志（用于测试快速模式）
echo ============================================================
echo.

echo [1/2] 清理Python缓存...
echo.

REM 清理 src/__pycache__
if exist "src\__pycache__" (
    rd /s /q "src\__pycache__"
    echo ✓ 已清理 src\__pycache__
)

REM 清理所有子目录的 __pycache__
for /d /r "src" %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
echo ✓ 已清理所有 __pycache__ 目录

REM 清理 .pyc 和 .pyo 文件
del /s /q "src\*.pyc" 2>nul
del /s /q "src\*.pyo" 2>nul
echo ✓ 已清理 .pyc 和 .pyo 文件

echo.
echo [2/2] 清理日志文件...
echo.

REM 清理 logs 目录
if exist "logs" (
    del /q "logs\*.log" 2>nul
    echo ✓ 已清理 logs 目录
)

echo.
echo ============================================================
echo ✅ 清理完成！
echo ============================================================
echo.
echo 现在可以运行程序测试快速模式了
echo.
echo 测试要点：
echo 1. 有登录缓存的账号应该显示"快速模式：有缓存，跳过获取资料"
echo 2. 无登录缓存的账号应该显示"快速模式：无缓存，切换为完整流程"
echo 3. 有缓存的账号不应该执行"步骤2: 获取资料"
echo.
pause
