@echo off
chcp 65001 >nul
echo ========================================
echo 快速启动程序
echo ========================================
echo.

REM 检查是否存在run.py
if not exist "run.py" (
    echo ❌ 错误: 找不到 run.py 文件
    echo 请确保在正确的目录下运行此脚本
    pause
    exit /b 1
)

REM 直接启动程序
echo 正在启动程序...
python run.py

REM 如果程序异常退出，暂停以查看错误信息
if errorlevel 1 (
    echo.
    echo ❌ 程序异常退出 (错误代码: %errorlevel%)
    pause
)
