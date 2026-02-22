@echo off
chcp 65001 >nul
echo ========================================
echo 清理缓存并启动程序
echo ========================================
echo.

REM 检查是否有Python进程在运行
echo [1/4] 检查运行中的进程...
tasklist /FI "IMAGENAME eq python.exe" 2>NUL | find /I /N "python.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo ⚠️  发现运行中的Python进程
    echo 是否要终止所有Python进程？ [Y/N]
    choice /C YN /N /M "请选择: "
    if errorlevel 2 goto skip_kill
    if errorlevel 1 (
        echo 正在终止Python进程...
        taskkill /F /IM python.exe >nul 2>&1
        timeout /t 2 /nobreak >nul
        echo ✓ Python进程已终止
    )
)
:skip_kill

REM 清理Python缓存
echo.
echo [2/4] 清理Python缓存...
if exist "__pycache__" (
    rd /s /q "__pycache__" 2>nul
    echo ✓ 清理 __pycache__
)
if exist "src\__pycache__" (
    rd /s /q "src\__pycache__" 2>nul
    echo ✓ 清理 src\__pycache__
)
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" 2>nul
echo ✓ Python缓存清理完成

REM 清理应用缓存
echo.
echo [3/4] 清理应用缓存...
if exist "login_cache" (
    echo 清理登录缓存...
    del /q "login_cache\*.json" 2>nul
    echo ✓ 登录缓存已清理
)
if exist "runtime_data" (
    echo 清理运行时数据...
    del /q "runtime_data\*.db-shm" 2>nul
    del /q "runtime_data\*.db-wal" 2>nul
    echo ✓ 运行时数据已清理
)
echo ✓ 应用缓存清理完成

REM 启动程序
echo.
echo [4/4] 启动程序...
echo ========================================
echo.

REM 检查是否存在run.py
if not exist "run.py" (
    echo ❌ 错误: 找不到 run.py 文件
    echo 请确保在正确的目录下运行此脚本
    pause
    exit /b 1
)

REM 启动程序
echo 正在启动程序...
python run.py

REM 如果程序异常退出，暂停以查看错误信息
if errorlevel 1 (
    echo.
    echo ❌ 程序异常退出 (错误代码: %errorlevel%)
    pause
)
