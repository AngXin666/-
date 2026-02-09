@echo off
chcp 65001 >nul
echo ============================================================
echo 清理Python缓存并测试快速模式
echo ============================================================
echo.

echo [1/3] 清理Python缓存...
echo.

REM 清理 __pycache__ 目录
echo 正在清理 __pycache__ 目录...
for /d /r "..\src" %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
echo ✓ 已清理 src/__pycache__

REM 清理 .pyc 文件
echo 正在清理 .pyc 文件...
del /s /q "..\src\*.pyc" 2>nul
echo ✓ 已清理 .pyc 文件

REM 清理 .pyo 文件
echo 正在清理 .pyo 文件...
del /s /q "..\src\*.pyo" 2>nul
echo ✓ 已清理 .pyo 文件

echo.
echo ============================================================
echo [2/3] 检查登录缓存状态
echo ============================================================
echo.

REM 检查 login_cache 目录
if exist "..\login_cache" (
    echo ✓ login_cache 目录存在
    dir /b "..\login_cache" | find /c /v "" > temp_count.txt
    set /p cache_count=<temp_count.txt
    del temp_count.txt
    echo   缓存账号数量: %cache_count%
) else (
    echo ✗ login_cache 目录不存在
)

echo.
echo ============================================================
echo [3/3] 运行快速模式测试
echo ============================================================
echo.

echo 提示：
echo - 快速模式 + 有缓存 → 应该跳过获取资料
echo - 快速模式 + 无缓存 → 应该自动切换为完整流程
echo.

pause

echo.
echo 测试完成！
echo.
echo 请查看日志输出，确认：
echo 1. 有缓存的账号显示"快速模式：有缓存，跳过获取资料"
echo 2. 无缓存的账号显示"快速模式：无缓存，切换为完整流程"
echo 3. 有缓存的账号不会执行"步骤2: 获取资料"
echo.
pause
